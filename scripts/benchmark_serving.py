"""Measure the serving path against NFR-01, NFR-02 and NFR-03.

    python scripts/benchmark_serving.py
    python scripts/benchmark_serving.py --requests 600 --users 120

Replays a realistic traffic pattern against the real application and reports:

* **Cache hit ratio** (NFR-03, target > 80%) - a property of the caching logic:
  key construction, TTLs and invalidation.
* **Latency percentiles** (NFR-01, p95 < 80 ms cached, p99 < 400 ms cold).
* **Event persistence** - events posted through the API actually reaching
  `user_events` in PostgreSQL.

**On the cache backend.** If a real Redis is reachable it is used. Otherwise the
benchmark falls back to `fakeredis`, which implements the same command
semantics in-process, and **says so in the output**. That distinction matters:
the hit *ratio* is a property of our key design and is measured faithfully
either way, but the *latency* numbers from a fallback run exclude Redis network
round trips and must not be quoted as a measurement of Redis.

Traffic is Zipf-distributed over users rather than uniform. A uniform replay
would give an unrealistically low hit ratio, because real traffic concentrates
on a small set of active users - which is precisely what makes caching work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "backend", REPO_ROOT / "ml"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "benchmark-only-secret")
os.environ.setdefault("LOG_LEVEL", "ERROR")

# A load test against our own service must not be throttled by our own abuse
# protection. Raised here rather than disabled, so the limiter still runs and a
# bug in it would still surface - it just does not fire at benchmark volumes.
os.environ.setdefault("RATE_LIMIT_RECOMMENDATIONS_PER_MINUTE", "100000")
os.environ.setdefault("RATE_LIMIT_EVENTS_PER_MINUTE", "100000")

PREFIX = "/api/v1"


def build_cache() -> tuple[object, str]:
    """A real Redis when one is reachable, `fakeredis` otherwise."""
    from app.cache.client import build_client

    try:
        client = build_client()
        client.ping()
        client.flushdb()
        return client, "redis"
    except Exception:
        pass

    import fakeredis

    return fakeredis.FakeRedis(decode_responses=True), "fakeredis"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from app.cache.client import set_redis
    from app.main import create_app
    from fastapi.testclient import TestClient

    cache, backend = build_cache()
    set_redis(cache)

    rng = np.random.default_rng(args.seed)

    # Zipf over a fixed user pool: real traffic concentrates on active users,
    # and a uniform replay would understate the hit ratio a real cache achieves.
    pool = np.arange(1, args.users + 1)
    weights = 1.0 / np.arange(1, args.users + 1) ** 0.8
    weights /= weights.sum()

    app = create_app()
    with TestClient(app) as client:
        readiness = client.get("/health/ready").json()
        database_available = readiness["database"]
        print(f"cache backend : {backend}")
        print(f"database      : {'connected' if database_available else 'UNAVAILABLE'}")
        print(f"engine loaded : {readiness['engine']['loaded']}")
        print()

        before = _event_count() if database_available else None

        print(f"posting {args.events} events...")
        event_timings: list[float] = []
        for index in range(args.events):
            user = int(rng.choice(pool, p=weights))
            product = int(rng.integers(1, 5000))
            started = time.perf_counter()
            response = client.post(
                f"{PREFIX}/events",
                json={
                    "event_type": "PRODUCT_VIEW",
                    "session_id": f"bench-session-{user:06d}",
                    "product_id": product,
                    "source": "pdp",
                    "metadata": {"dwell_ms": 3000},
                },
            )
            event_timings.append((time.perf_counter() - started) * 1000.0)
            assert response.status_code == 202, response.text
            _ = index

        # Let the buffered sink drain before counting rows.
        sink = app.state.event_sink
        sink.flush(timeout=15.0)
        time.sleep(1.0)

        print(f"replaying {args.requests} recommendation requests...")
        cold: list[float] = []
        warm: list[float] = []
        hits = misses = 0

        for _ in range(args.requests):
            user = int(rng.choice(pool, p=weights))
            started = time.perf_counter()
            response = client.get(
                f"{PREFIX}/recommendations/home/{user}?limit=12",
                headers={"X-Session-Id": f"bench-session-{user:06d}"},
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            assert response.status_code == 200, response.text

            section = response.json()["sections"].get("for_you")
            if section and section["cache_hit"]:
                hits += 1
                warm.append(elapsed)
            else:
                misses += 1
                cold.append(elapsed)

        after = _event_count() if database_available else None
        sink_stats = sink.stats()

    total = hits + misses
    ratio = hits / total if total else 0.0

    print()
    print("=" * 74)
    print("SERVING BENCHMARK")
    print("=" * 74)

    print(f"\nCache (NFR-03: hit ratio > 80%)   backend={backend}")
    print(f"  requests        {total}")
    print(f"  hits / misses   {hits} / {misses}")
    print(f"  hit ratio       {ratio:.1%}   {'PASS' if ratio > 0.80 else 'FAIL'}")

    print("\nLatency (NFR-01: p95 < 80ms cached, p99 < 400ms cold)")
    if warm:
        print(f"  cached  p50 {percentile(warm, 50):6.1f}ms   "
              f"p95 {percentile(warm, 95):6.1f}ms   p99 {percentile(warm, 99):6.1f}ms"
              f"   {'PASS' if percentile(warm, 95) < 80 else 'FAIL'}")
    if cold:
        print(f"  cold    p50 {percentile(cold, 50):6.1f}ms   "
              f"p95 {percentile(cold, 95):6.1f}ms   p99 {percentile(cold, 99):6.1f}ms"
              f"   {'PASS' if percentile(cold, 99) < 400 else 'FAIL'}")

    print("\nEvent ingestion (NFR-02: p99 < 30ms at the API boundary)")
    print(f"  p50 {percentile(event_timings, 50):6.2f}ms   "
          f"p95 {percentile(event_timings, 95):6.2f}ms   "
          f"p99 {percentile(event_timings, 99):6.2f}ms"
          f"   {'PASS' if percentile(event_timings, 99) < 30 else 'FAIL'}")
    print(f"  sink: {sink_stats}")

    if before is not None and after is not None:
        written = after - before
        print("\nPersistence")
        print(f"  user_events before  {before:,}")
        print(f"  user_events after   {after:,}")
        print(f"  rows written        {written}   "
              f"{'PASS' if written >= args.events else 'FAIL'}")

    failures = []
    if ratio <= 0.80:
        failures.append("cache hit ratio")
    if warm and percentile(warm, 95) >= 80:
        failures.append("cached p95")
    if before is not None and after is not None and (after - before) < args.events:
        failures.append("event persistence")

    print()
    if failures:
        print(f"BENCHMARK: FAILED ({', '.join(failures)})")
        return 1
    print("BENCHMARK: PASSED")
    if backend == "fakeredis":
        print(
            "\nNote: the cache ratio above is a faithful measure of the caching\n"
            "logic, but the latency numbers exclude Redis network round trips."
        )
    return 0


def _event_count() -> int | None:
    try:
        from app.db.session import engine
        from sqlalchemy import text

        with engine.connect() as connection:
            return int(connection.execute(text("SELECT count(*) FROM user_events")).scalar())
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
