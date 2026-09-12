"""Run PostgreSQL locally without Docker or administrator rights.

    python scripts/local_postgres.py start      # initialise and start
    python scripts/local_postgres.py status
    python scripts/local_postgres.py stop
    python scripts/local_postgres.py destroy    # stop and delete the data directory

**Why this exists.** The intended local substrate is Docker Compose (ADR-015),
which needs WSL2, which needs Windows features that require administrator
rights to enable. On a machine where that is not available, the entire
database-dependent half of the project would be unverifiable - migrations never
applied, seeding never run, indexes never exercised by a real planner.

`embedded-postgres` ships PostgreSQL 18 **with pgvector** as a pip wheel and
runs it from a user-writable directory. That is not a downgrade of the
architecture: it is the same PostgreSQL, the same extension and the same
migration. Only the packaging differs, and the connection string is identical,
so nothing in the application knows or cares.

The Compose definition remains the deployment target. This is the developer
loop when Compose cannot run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / ".pgdata"
PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "recommendations")
DB_USER = os.getenv("POSTGRES_USER", "recsys")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "recsys_dev_password")

#: Matches the tuning in docker-compose.yml, so the developer loop and the
#: deployment target plan queries the same way. A local server with default
#: 128MB shared_buffers would spill to disk on the bulk seed and give
#: misleading timings.
SERVER_SETTINGS = [
    "-c", "shared_buffers=512MB",
    "-c", "work_mem=32MB",
    "-c", "maintenance_work_mem=256MB",
    "-c", "effective_cache_size=2GB",
    "-c", "max_connections=100",
    "-c", "log_min_duration_statement=500",
]


def _bin_dir() -> Path:
    import embedded_postgres

    return Path(embedded_postgres.__file__).parent / "pginstall" / "bin"


def _run(
    executable: str, *args: str, check: bool = True, capture: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    """Invoke one of the bundled PostgreSQL binaries.

    `capture=False` exists for `pg_ctl start`, and the reason is not obvious.
    `pg_ctl` spawns the server as a child that **inherits the captured stdout
    pipe and keeps it open for its entire lifetime**. `subprocess.run` then
    blocks forever waiting for EOF on a pipe held by a daemon that is running
    perfectly well - the server is up, the script never returns, and nothing in
    the output suggests why. Sending the server's handles to DEVNULL detaches
    it; its real output already goes to the log file passed with `-l`.
    """
    command = [str(_bin_dir() / executable), *args]
    if not capture:
        return subprocess.run(
            command,
            check=check,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )


def _connect(dbname: str):
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    conn = psycopg2.connect(
        host="127.0.0.1", port=PORT, user=DB_USER, password=DB_PASSWORD, dbname=dbname
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def _ensure_database() -> None:
    """Create the database and its extensions.

    Done through psycopg2 rather than `psql`. Under scram-sha-256 auth `psql`
    prompts for a password on stdin, and a subprocess captured with
    `capture_output=True` has no terminal to prompt on - so it blocks forever
    rather than failing. psycopg2 takes the password as an argument.
    """
    from psycopg2 import sql

    with _connect("postgres") as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"created database {DB_NAME}")

    # The same extensions the Compose init hook installs, so the two
    # environments are not subtly different.
    with _connect(DB_NAME) as conn, conn.cursor() as cur:
        for extension in ("vector", "pg_trgm", "pg_stat_statements"):
            try:
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
            except Exception as exc:
                print(f"  {extension}: {str(exc).splitlines()[0]}")


def _describe() -> tuple[str, list[str]]:
    with _connect(DB_NAME) as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.execute("SELECT extname FROM pg_extension ORDER BY extname")
        extensions = [row[0] for row in cur.fetchall()]
    return version, extensions


def is_running() -> bool:
    if not DATA_DIR.exists():
        return False
    result = _run("pg_ctl", "-D", str(DATA_DIR), "status", check=False)
    return result.returncode == 0


def start() -> int:
    if is_running():
        print(f"already running on port {PORT}")
        return 0

    if not (DATA_DIR / "PG_VERSION").exists():
        print(f"initialising a new cluster at {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        password_file = DATA_DIR.parent / ".pgpass_init"
        password_file.write_text(DB_PASSWORD, encoding="utf-8")
        try:
            # Deterministic encoding and collation: index ordering and text
            # search must not depend on the developer's host locale.
            _run(
                "initdb",
                "-D", str(DATA_DIR),
                "-U", DB_USER,
                f"--pwfile={password_file}",
                "--encoding=UTF8",
                "--no-locale",
                "-A", "scram-sha-256",
            )
        finally:
            password_file.unlink(missing_ok=True)
        print("cluster initialised")

    log = DATA_DIR / "server.log"
    print(f"starting on port {PORT}")
    _run(
        "pg_ctl",
        "-D", str(DATA_DIR),
        "-l", str(log),
        "-o", " ".join(["-p", str(PORT), *SERVER_SETTINGS]),
        "-w",
        "start",
        capture=False,
    )

    for _ in range(30):
        if _run("pg_isready", "-p", str(PORT), "-U", DB_USER, check=False).returncode == 0:
            break
        time.sleep(1)
    else:
        print(f"server did not become ready; see {log}")
        return 1

    _ensure_database()
    version, extensions = _describe()

    print(f"\nready: {version.split(',')[0]}")
    print(f"  database:   {DB_NAME}")
    print(f"  user:       {DB_USER}")
    print(f"  port:       {PORT}")
    print(f"  extensions: {', '.join(extensions)}")
    print(f"  data:       {DATA_DIR}")
    print("\nnext: cd backend && alembic upgrade head")
    return 0


def stop() -> int:
    if not is_running():
        print("not running")
        return 0
    _run("pg_ctl", "-D", str(DATA_DIR), "-m", "fast", "-w", "stop")
    print("stopped")
    return 0


def status() -> int:
    if is_running():
        print(f"running on port {PORT}, data at {DATA_DIR}")
        return 0
    print("not running")
    return 1


def destroy() -> int:
    import shutil

    stop()
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        print(f"removed {DATA_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status", "destroy"])
    args = parser.parse_args()
    return {"start": start, "stop": stop, "status": status, "destroy": destroy}[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
