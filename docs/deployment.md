# Deployment

Status: **Phase 17 deliverable — complete. The database path is fully
verified; container execution remains unverified (see §7)**
Last updated: 2026-09-12

---

## 1. Local setup, without Docker

Everything except event persistence works with no database and no container
runtime. This is the fastest way to see the system running.

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r ml/requirements.txt -r requirements-dev.txt
```

```bash
python data-generation/generate.py && python data-generation/diagnostics.py
```

```bash
python ml/pipelines/train.py --mlflow
```

Then, in two terminals:

```bash
PYTHONPATH="backend;ml" python -m uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**. The API docs are at **http://localhost:8000/docs**.

**What works:** every recommendation surface, the storefront, the admin
dashboard, training, evaluation, MLflow, drift detection.

**What does not:** events are accepted and counted but held in memory, so they
are lost on restart. The API reports this honestly at `/health/ready`
(`"database": false`) and logs a warning at start-up.

### Adding a database, still without Docker

`embedded-postgres` ships PostgreSQL 18 with pgvector as a pip wheel and needs
no administrator rights (ADR-016). It is kept in its own requirements file
because it is a large download that neither CI nor any image needs:

```bash
pip install -r requirements-local-db.txt
```

```bash
python scripts/local_postgres.py start && cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py --truncate && python scripts/verify_database.py
```

That takes about a minute and unlocks event persistence, the analytics
endpoints and the database test suite. `stop`, `status` and `destroy` do what
they say.

Measure the serving path afterwards:

```bash
python scripts/benchmark_serving.py
```

### When the local server is up but nothing can reach it

Specific to Windows, and worth knowing because it does not look like what it
is. Symptom: `status` says running, but every client hangs until its connect
timeout, and the test suite reports `no PostgreSQL reachable` after several
minutes instead of several seconds.

The cause is in `.pgdata/server.log`:

```
could not reserve shared memory region (addr=...) error code 487
```

Windows has no `fork`, so PostgreSQL starts each backend by re-executing the
postmaster and re-mapping the shared memory segment at the same address in the
new process. If anything else has taken that address - ASLR, or a DLL injected
by antivirus - the backend cannot start. The postmaster stays healthy and keeps
**accepting** connections it can never service, which is why clients hang
rather than being refused.

Two things in `scripts/local_postgres.py` address it:

- `shared_buffers` is 128MB on Windows and 512MB elsewhere. A smaller segment
  is easier to place, and this failure gets much more likely as it grows.
- `status` and `start` execute a real query rather than checking a pid.
  `pg_ctl status` reports success for a server in this state, so a pid check
  reports a completely unusable database as healthy - the worst kind of health
  check. When the query fails, the log is read and the explanation printed.

The fix is to restart; a fresh process usually gets a usable address:

```bash
python scripts/local_postgres.py stop && python scripts/local_postgres.py start
```

If `stop` also fails, that is the same bug - a clean shutdown needs a backend
it cannot fork. `stop` escalates to `-m immediate` automatically; if even that
fails, end the `postgres.exe` process. Nothing is lost: PostgreSQL recovers
from the WAL, and the dataset reloads in 17 seconds with
`python scripts/seed_database.py --truncate`.

### Required configuration

Copy `.env.example` to `.env` and set at minimum:

```
APP_ENV=development
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(64))">
POSTGRES_PASSWORD=<any value in development>
```

The application **refuses to start** outside development without a real
`JWT_SECRET_KEY` and `POSTGRES_PASSWORD` (`Settings.require_secrets`). Failing
loudly at start-up is better than running with a default secret.

---

## 2. Local setup, with Docker

```bash
docker compose up -d
```

Brings up Postgres (with pgvector), Redis, the API and the frontend. Then:

```bash
cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py --truncate && python scripts/verify_database.py
```

| Service | URL |
| --- | --- |
| Storefront | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |

Add the observability stack:

```bash
docker compose --profile ops up -d
```

| Service | URL | Notes |
| --- | --- | --- |
| MLflow | http://localhost:5000 | |
| Prometheus | http://localhost:9090 | |
| Grafana | http://localhost:3001 | anonymous viewer enabled; admin password from `GRAFANA_PASSWORD` |

Run a training job in a container:

```bash
docker compose --profile training run --rm ml-training python ml/pipelines/train.py --mlflow
```

### Why the compose file is shaped the way it is

**Healthcheck-gated dependencies.** `backend` waits on
`postgres: {condition: service_healthy}`, not merely on container start. Without
that, `up` races: the API tries to connect before Postgres is accepting
connections and crash-loops until it happens to win.

**Read-only artefact mounts.** `ml/artifacts` and `data/synthetic` are mounted
`:ro` into the API. The service *loads* models; it never writes them. A serving
process that can overwrite its own model is one bug away from an
unreproducible incident.

**Profiles.** `ops` and `training` are opt-in so a plain `docker compose up`
starts four containers rather than eight. Developers who only want the
storefront should not pay for Prometheus.

**`POSTGRES_PASSWORD:?` syntax.** Compose fails with a clear message if the
variable is unset, instead of silently starting Postgres with an empty password.

---

## 3. Images

| Image | Base | Size driver | Notes |
| --- | --- | --- | --- |
| `backend` | `python:3.12-slim` | scikit-learn, LightGBM | Multi-stage; no compilers in the runtime layer |
| `ml-training` | `python:3.12-slim` | + XGBoost, MLflow | Separate on purpose |
| `frontend` | `nginx:1.27-alpine` | ~660 kB of assets | No Node in the runtime image |

**Why the training image is separate.** It needs XGBoost, MLflow and optionally
Torch — none of which belong in a latency-sensitive API image. It also runs on a
different schedule and can be resourced independently.

**Why multi-stage.** Wheels are built in a stage that has `build-essential`; the
runtime stage installs from those wheels. The runtime image therefore contains
no compilers and no build headers — smaller, and a much smaller attack surface.

**Both application images run as a non-root user** (uid 10001 / 10002). A
container that does not need root should not have it.

### Frontend bundle

Split so the charting library is not on the shopper's critical path:

| Chunk | Size | Gzipped | Loaded by |
| --- | --- | --- | --- |
| `index` | 100 kB | 36 kB | every page |
| `vendor` | 164 kB | 53 kB | every page |
| `charts` | 400 kB | 109 kB | **admin dashboard only** |

Without the split, every product page would download Recharts.

---

## 4. CI/CD

`.github/workflows/ci.yml`, ordered cheapest-first so a broken commit fails in
under a minute rather than after the eight-minute ML job.

| Job | Does |
| --- | --- |
| `lint` | ruff; mypy (non-blocking) |
| `frontend` | `tsc --noEmit`, production build, upload `dist` |
| `test` | Postgres + Redis services, migrations, **schema-drift check**, full pytest with coverage |
| `ml-pipeline` | Generate data, **run the diagnostic gate**, train, assert the model beats the baseline |
| `docker` | Build all three images with layer caching; validate the compose file |
| `security` | Reject committed `.env` files and private keys; `pip-audit` |

Two jobs are worth calling out:

**The schema-drift check** guards ADR-013. It runs `alembic revision
--autogenerate` against a freshly-migrated database and fails if the generated
migration contains any operations — meaning the models and migrations disagree.
A model change without a migration is caught here rather than at deploy time.

**The diagnostic gate** runs `data-generation/diagnostics.py` on every commit.
If a change to the generator destroys the learnable structure, it fails there
rather than surfacing weeks later as an unexplained metric regression.

`.github/workflows/retrain.yml` runs nightly at 03:00 UTC. It is
**drift-triggered, not unconditional** — retraining on a schedule regardless of
whether anything changed burns compute and, without a gate, random-walks the
production model. Exit code 2 means "a retrain ran and the gate rejected it",
which is a real signal rather than a failure.

---

## 5. Operations runbook

### Deploying a new model

```bash
python ml/pipelines/retrain.py --force
```

The gate decides. On promotion, artefacts are swapped atomically and the
previous version is kept at `ml/artifacts.previous/`.

**Rollback:**

```bash
mv ml/artifacts ml/artifacts.bad && mv ml/artifacts.previous ml/artifacts
```

Then restart the API. Because cache keys embed the model version, the rollback
also rolls the cache back — no flush needed, no cold-cache latency spike.

### Incident triage

| Symptom | First check | Likely cause |
| --- | --- | --- |
| Recommendations look generic | `recommendation_fallback_total` | Ranker artefact failed to load |
| p95 latency climbing | `recommendation_stage_latency_seconds` by stage | Retrieval source slow, or cache cold |
| CTR declining, nothing else alerting | `/admin/drift` | Feature drift — the silent failure |
| Empty rails | `recommendation_empty_total` | Business rules filtering everything (stock?) |
| Events missing from training | `events_dropped_total` | Ingest buffer full |

**`recommendation_fallback_total` is the metric to page on.** When the ranker
fails, latency *improves*, errors stay at zero, and the only other symptom is a
slow CTR decline a week later.

### Scaling

| Stage | Change |
| --- | --- |
| 10k users (now) | Single API process, nightly training |
| 1M users | Move item kNN to a dedicated ANN index; shard the feature store; precompute homepage candidates for the long tail; read replicas for analytics |
| 100M events/day | Kafka ingestion behind the existing `EventSink` interface; Postgres keeps aggregates only; training reads from the lake |

The seams for that evolution exist now: `EventSink`, `CandidateSource`,
`VectorIndex`, `ModelResolver`. Buying the interface is cheap; buying the
infrastructure before it carries load is not.

---

## 6. Security checklist

- [x] No credentials in source; `.env` gitignored and CI-enforced
- [x] Application refuses to start without secrets outside development
- [x] JWT with separate access/refresh types, verified on decode
- [x] Argon2id password hashing, timing-safe verification
- [x] Role hierarchy enforced by dependency
- [x] All input validated by Pydantic; SQL exclusively through bound parameters
- [x] Rate limiting per scope, failing open
- [x] CORS restricted to configured origins
- [x] Containers run as non-root
- [x] `user_id` never accepted from a request body
- [x] The dev token endpoint 404s outside `APP_ENV=development`

---

## 7. Known gaps

Docker Desktop's Linux engine cannot start on the development machine:
`VirtualMachinePlatform` and `Microsoft-Windows-Subsystem-Linux` are disabled in
Windows features, and enabling them requires administrator rights.

Rather than leave the database half of the project unverified, the developer
loop runs PostgreSQL from a pip wheel (ADR-016). **Everything that needed a
database is now verified**:

| Item | Status |
| --- | --- |
| `alembic upgrade head` | **Applied.** Clean on the first attempt |
| Seeding | **561,173 rows in 17.7 s** |
| `scripts/verify_database.py` | **24/24 checks pass** - and it caught a real bug first (database.md section 9) |
| Partition routing | **18 partitions, 7 populated, default empty** |
| Index usage | **Verified by the planner** - partition pruning, partial index, no sequential scans |
| Schema-drift check (ADR-013) | **No drift** |
| Cache hit ratio (NFR-03) | **81.6%** against a `fakeredis` backend |
| Latency (NFR-01) | **cached p95 6.3 ms, cold p99 26.1 ms** |
| Event ingest (NFR-02) | **p99 5.15 ms**, 200/200 rows persisted, sessions registered |
| `docker compose config` | **Validates**, 4 default services resolve |

What genuinely remains unverified, and why:

| Item | Why |
| --- | --- |
| `docker compose up` executing containers | Needs the WSL2 engine |
| Docker image builds | Needs a running daemon |
| Grafana dashboards rendering | Needs Grafana, which needs a container |
| A real Redis server | `fakeredis` implements the same command semantics, so the **hit ratio** is measured faithfully; Redis-specific *latency* is not |

The fix, from an **Administrator** PowerShell, then reboot:

```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart; dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

Nothing in the application changes when that happens: the connection string is
identical, so `docker compose up` substitutes for
`scripts/local_postgres.py start` and the rest of the workflow is unchanged.
