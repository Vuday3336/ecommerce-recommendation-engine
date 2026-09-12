"""Generate the synthetic e-commerce dataset.

    python data-generation/generate.py
    python data-generation/generate.py --users 2000 --products 1200 --days 90
    python data-generation/generate.py --out data/synthetic --seed 7

Writes Parquet files plus a manifest describing the run. Nothing is written to
the database here - seeding is a separate step, so the dataset can be
regenerated, inspected and diagnosed without touching Postgres.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config.simulation import SimulationConfig
from generators.behavior import simulate
from generators.catalog import build_catalogue
from generators.users import generate_users, profiles_to_frame

DEFAULT_OUT = REPO_ROOT / "data" / "synthetic"

#: Columns holding dicts or lists. Parquet needs a consistent struct schema and
#: these are genuinely heterogeneous (product attributes differ by category),
#: so they are stored as JSON text and parsed back on load.
JSON_COLUMNS: dict[str, tuple[str, ...]] = {
    "products": ("attributes", "tags"),
    "product_variants": ("options",),
    "users": ("onboarding_categories",),
    "events": ("event_metadata",),
    "latent_user_profiles": ("brand_affinity",),
}


def _encode_json_columns(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = JSON_COLUMNS.get(name, ())
    if not columns:
        return frame
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(json.dumps)
    return out


def _apply_denormalised_counters(
    products: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Fill the catalogue counters that a real store maintains incrementally.

    `products.view_count`, `purchase_count`, `rating_average` and
    `rating_count` are denormalised copies used by the catalogue API. They are
    derived here from the generated events rather than invented, so the
    catalogue a shopper sees is consistent with the behaviour log the models
    train on. (The point-in-time feature snapshots in `product_features` are a
    separate, later computation - see the warning in `app/models/features.py`.)
    """
    products = products.copy()

    views = (
        events.loc[events["event_type"] == "PRODUCT_VIEW", "product_id"]
        .value_counts()
        .rename("view_count_calc")
    )
    purchases = (
        events.loc[events["event_type"] == "PURCHASE", "product_id"]
        .value_counts()
        .rename("purchase_count_calc")
    )

    ratings = events.loc[events["event_type"] == "PRODUCT_RATING"].copy()
    if len(ratings):
        ratings["rating"] = ratings["event_metadata"].map(lambda m: m.get("rating"))
        grouped = ratings.groupby("product_id")["rating"].agg(["mean", "count"])
    else:
        grouped = pd.DataFrame(columns=["mean", "count"])

    products = products.merge(views, left_on="id", right_index=True, how="left")
    products = products.merge(purchases, left_on="id", right_index=True, how="left")
    products = products.merge(grouped, left_on="id", right_index=True, how="left")

    products["view_count"] = products["view_count_calc"].fillna(0).astype("int64")
    products["purchase_count"] = products["purchase_count_calc"].fillna(0).astype("int64")
    # New ratings are added to the pre-window history, not substituted for it.
    products["rating_count"] = (
        products["rating_count"] + products["count"].fillna(0)
    ).astype("int64")

    # Bayesian shrinkage toward the catalogue mean: a product with two ratings
    # must not outrank a product with two hundred just because both happened to
    # be five stars. Without this the popularity baseline would be dominated by
    # noise, and every later model would be compared against a broken baseline.
    catalogue_mean = float(products["rating_average"].mean())
    prior_weight = 12.0
    # The product's intrinsic rating (from its latent quality) stands in for the
    # pre-window review history; observed in-window ratings update it.
    historical_mean = products["rating_average"]
    historical_count = (products["rating_count"] - products["count"].fillna(0)).clip(lower=0)
    observed_mean = products["mean"].fillna(catalogue_mean)
    observed_count = products["count"].fillna(0)
    products["rating_average"] = (
        (
            historical_mean * historical_count
            + observed_mean * observed_count
            + catalogue_mean * prior_weight
        )
        / (historical_count + observed_count + prior_weight)
    ).round(2)

    return products.drop(
        columns=["view_count_calc", "purchase_count_calc", "mean", "count"]
    )


def build_dataset(config: SimulationConfig) -> dict[str, pd.DataFrame]:
    """Run the full generation pipeline in memory."""
    rng = np.random.default_rng(config.seed)

    catalogue = build_catalogue(config, rng)
    users, profiles = generate_users(config, rng, catalogue)
    result = simulate(config, rng, catalogue, profiles)

    products = _apply_denormalised_counters(catalogue.products, result.events)

    return {
        "categories": catalogue.categories,
        "brands": catalogue.brands,
        "products": products,
        "product_variants": catalogue.variants,
        "users": users,
        "user_sessions": result.sessions,
        "events": result.events,
        "orders": result.orders,
        "order_items": result.order_items,
        "latent_user_profiles": profiles_to_frame(profiles),
    }


def write_dataset(
    frames: dict[str, pd.DataFrame], config: SimulationConfig, out_dir: Path, elapsed: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in frames.items():
        _encode_json_columns(name, frame).to_parquet(
            out_dir / f"{name}.parquet", index=False
        )

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "generation_seconds": round(elapsed, 2),
        "config": config.describe(),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "event_type_counts": (
            frames["events"]["event_type"].value_counts().to_dict()
            if len(frames["events"])
            else {}
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--products", type=int, default=5_000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, default=dt.date(2026, 9, 1))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-events",
        type=int,
        default=100_000,
        help="Fail if fewer events are produced (Phase 2 exit criterion)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SimulationConfig(
        seed=args.seed,
        n_users=args.users,
        n_products=args.products,
        simulation_days=args.days,
        end_date=args.end_date,
        min_events=args.min_events,
    )

    print(
        f"Generating: {config.n_users:,} users, {config.n_products:,} products, "
        f"{config.simulation_days} days ({config.start_date} to {config.end_date}), "
        f"seed={config.seed}"
    )
    started = time.perf_counter()
    frames = build_dataset(config)
    elapsed = time.perf_counter() - started

    write_dataset(frames, config, args.out, elapsed)

    print(f"\nGenerated in {elapsed:.1f}s -> {args.out}")
    width = max(len(name) for name in frames)
    for name, frame in frames.items():
        print(f"  {name:<{width}}  {len(frame):>10,} rows")

    events = frames["events"]
    print("\nEvent mix:")
    counts = events["event_type"].value_counts()
    for event_type, count in counts.items():
        print(f"  {event_type:<18} {count:>9,}  ({count / len(events):6.2%})")

    if len(events) < config.min_events:
        print(
            f"\nFAIL: produced {len(events):,} events, "
            f"below the required minimum of {config.min_events:,}."
        )
        return 1

    print(f"\nOK: {len(events):,} events (minimum {config.min_events:,}).")
    print("Next: python data-generation/diagnostics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
