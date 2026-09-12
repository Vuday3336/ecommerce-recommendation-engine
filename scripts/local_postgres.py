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
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / ".pgdata"
PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "recommendations")
DB_USER = os.getenv("POSTGRES_USER", "recsys")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "recsys_dev_password")

#: Broadly matches the tuning in docker-compose.yml, so the developer loop and
#: the deployment target plan queries the same way. A local server with the
#: default 128MB `shared_buffers` would spill to disk on the bulk seed and give
#: misleading timings.
#:
#: **`shared_buffers` is deliberately smaller on Windows**, and this is not
#: cosmetic. Windows has no `fork`, so PostgreSQL starts each backend by
#: re-executing the postmaster and re-mapping the shared memory segment at the
#: same address in the child. If anything else has taken that address in the
#: new process - ASLR, or a DLL injected by antivirus or other security
#: software - the mapping fails and the backend dies with:
#:
#:     could not reserve shared memory region (addr=...) error code 487
#:
#: The larger the segment, the harder it is to place, so 512MB makes this
#: likely on exactly the machines this script exists to support. The failure
#: mode is also unusually bad: the postmaster is healthy and still accepts the
#: TCP connection, so clients do not get "connection refused" - they hang until
#: their connect timeout, and a test suite that skips on "no PostgreSQL
#: reachable" takes eight minutes to report nothing.
_WINDOWS = sys.platform == "win32"

SERVER_SETTINGS = [
    "-c", f"shared_buffers={'128MB' if _WINDOWS else '512MB'}",
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
    """Is the postmaster process alive?

    Note what this does *not* tell you: `pg_ctl status` checks a pid, so it
    answers "is the process there", not "can it serve a query". Those come
    apart in practice - see `can_serve`.
    """
    if not DATA_DIR.exists():
        return False
    result = _run("pg_ctl", "-D", str(DATA_DIR), "status", check=False)
    return result.returncode == 0


def can_serve(timeout: int = 5, dbname: str | None = None) -> tuple[bool, str]:
    """Actually execute a query, and say what happened.

    A pid check is not a health check. The failure this exists for is a live
    postmaster that accepts the TCP connection and then cannot fork a backend
    to handle it (the shared-memory reservation failure described above, or a
    server out of connection slots). `pg_ctl status` reports success, `status`
    prints "running", and every client hangs until its own timeout - so the
    tooling confidently says the database is up while nothing can reach it.

    Returning the reason rather than a bare bool matters here: "timeout
    expired" and "authentication failed" need completely different responses
    from whoever is reading the output.
    """
    try:
        import psycopg2
    except ImportError:
        return False, "psycopg2 is not installed"
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            # `postgres` always exists; the application database does not yet
            # on a freshly initialised cluster, and probing for it there would
            # report a healthy server as broken.
            dbname=dbname or DB_NAME,
            connect_timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0]
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    finally:
        conn.close()
    return True, "accepting queries"


def _diagnose_from_log() -> str | None:
    """Turn the one Windows failure that looks like a hang into an explanation."""
    log = DATA_DIR / "server.log"
    if not log.exists():
        return None
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    if any("could not reserve shared memory region" in line for line in tail):
        return (
            "The server log shows repeated 'could not reserve shared memory\n"
            "region ... error code 487'. On Windows PostgreSQL starts each\n"
            "backend by re-mapping shared memory at a fixed address in a new\n"
            "process; when something else already occupies that address the\n"
            "backend cannot start, while the postmaster stays up and keeps\n"
            "accepting connections that then go nowhere.\n\n"
            "Fix: stop the server, then start it again - a fresh process usually\n"
            "gets a usable address. It recurs, so `shared_buffers` is already\n"
            "kept small on Windows to make the region easier to place. If it\n"
            "persists, antivirus or other software injecting DLLs into the\n"
            "postgres process is the usual cause.\n\n"
            "    python scripts/local_postgres.py stop\n"
            "    python scripts/local_postgres.py start\n\n"
            "If `stop` also fails, the shutdown needs a backend it cannot fork\n"
            "either; end the `postgres.exe` process and start again. Nothing is\n"
            "lost - PostgreSQL recovers from the WAL, and the dataset is\n"
            "reproducible with `python scripts/seed_database.py --truncate`."
        )
    return None


def start() -> int:
    if is_running():
        serving, detail = can_serve()
        if serving:
            print(f"already running on port {PORT}, {detail}")
            return 0
        # A live postmaster that cannot serve is the case worth handling
        # loudly, because "already running" would be technically true and
        # completely useless.
        print(f"a server is up on port {PORT} but is NOT serving: {detail}")
        explanation = _diagnose_from_log()
        if explanation:
            print("\n" + explanation)
        return 1

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

    # `pg_isready` only completes a startup packet exchange; it does not prove a
    # backend can be forked to run a query. Proving that here means a failure
    # surfaces as a clear message from the command that was supposed to start
    # the database, rather than as a mysterious timeout in whatever runs next.
    serving, detail = can_serve(timeout=10, dbname="postgres")
    if not serving:
        print(f"\nthe server started but cannot serve queries: {detail}")
        explanation = _diagnose_from_log()
        print("\n" + explanation if explanation else f"\nSee {log}")
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
    """Shut down, escalating if a clean shutdown cannot complete.

    `-m fast` is tried first because it lets the server checkpoint. It can fail
    outright on Windows, though, for the same reason the server was unusable in
    the first place: a clean shutdown needs a backend, and a server that cannot
    fork a backend cannot shut itself down either. `-m immediate` skips that,
    and PostgreSQL recovers from the WAL on the next start - which is the same
    guarantee that makes a power cut survivable.
    """
    if not is_running():
        print("not running")
        return 0

    if _run("pg_ctl", "-D", str(DATA_DIR), "-m", "fast", "-w", "stop", check=False).returncode == 0:
        print("stopped")
        return 0

    print("clean shutdown failed; trying immediate")
    if _run(
        "pg_ctl", "-D", str(DATA_DIR), "-m", "immediate", "-w", "stop", check=False
    ).returncode == 0:
        print("stopped (immediate; the next start will recover from the WAL)")
        return 0

    print(
        "the server will not shut down. It is wedged badly enough that it\n"
        "cannot fork the backend a shutdown needs - end the postgres process\n"
        "and start again. Nothing is lost: PostgreSQL recovers from the WAL,\n"
        "and the dataset is reproducible with\n"
        "`python scripts/seed_database.py --truncate`."
    )
    explanation = _diagnose_from_log()
    if explanation:
        print("\n" + explanation)
    return 1


def status() -> int:
    """Report both facts, because they can disagree.

    A single line saying "running" is what hid a completely unusable server
    behind a healthy-looking pid, so this prints the process state and the
    query state separately and only exits 0 when a query actually succeeded.
    """
    if not is_running():
        print("not running")
        return 1

    serving, detail = can_serve()
    if serving:
        print(f"running on port {PORT}, {detail}")
        print(f"  data: {DATA_DIR}")
        return 0

    print(f"process is up on port {PORT}, but NOT serving: {detail}")
    print(f"  data: {DATA_DIR}")
    explanation = _diagnose_from_log()
    if explanation:
        print("\n" + explanation)
    else:
        print(f"\nSee {DATA_DIR / 'server.log'}")
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
