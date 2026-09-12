"""Tests for the synthetic data generator.

These run without a database. They protect two properties that everything
downstream depends on: the dataset is reproducible from a seed (NFR-05), and it
contains the behavioural structure the models are supposed to discover (R-02).

A small population is used so the suite stays fast; the thresholds are relaxed
accordingly, because a 600-user sample is noisier than the full 10,000. The
full-scale gate lives in `data-generation/diagnostics.py`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from config.simulation import SimulationConfig
from config.taxonomy import COMPLEMENTARY_PAIRS, DEPARTMENTS, iter_subcategories
from generate import build_dataset
from generators.catalog import build_catalogue

SMALL = SimulationConfig(seed=11, n_users=600, n_products=700, simulation_days=120)


@pytest.fixture(scope="module")
def dataset() -> dict[str, pd.DataFrame]:
    return build_dataset(SMALL)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_catalogue() -> None:
    first = build_catalogue(SMALL, np.random.default_rng(SMALL.seed))
    second = build_catalogue(SMALL, np.random.default_rng(SMALL.seed))
    pd.testing.assert_frame_equal(first.products, second.products)
    pd.testing.assert_frame_equal(first.variants, second.variants)


def test_different_seed_produces_different_catalogue() -> None:
    first = build_catalogue(SMALL, np.random.default_rng(1))
    second = build_catalogue(SMALL, np.random.default_rng(2))
    assert not first.products["price"].equals(second.products["price"])


def test_full_dataset_is_reproducible() -> None:
    first = build_dataset(SMALL)
    second = build_dataset(SMALL)
    assert len(first["events"]) == len(second["events"])
    pd.testing.assert_frame_equal(
        first["events"].head(500).reset_index(drop=True),
        second["events"].head(500).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------


def test_catalogue_covers_every_subcategory(dataset: dict[str, pd.DataFrame]) -> None:
    expected = {sub.name for _, _, sub in iter_subcategories()}
    assert set(dataset["products"]["subcategory"]) == expected


def test_category_tree_is_three_levels(dataset: dict[str, pd.DataFrame]) -> None:
    categories = dataset["categories"]
    assert set(categories["depth"]) == {0, 1, 2}
    roots = categories[categories["depth"] == 0]
    assert len(roots) == len(DEPARTMENTS)
    assert roots["parent_id"].isna().all()
    # Every non-root points at an existing parent.
    non_roots = categories[categories["depth"] > 0]
    assert non_roots["parent_id"].isin(categories["id"]).all()


def test_products_reference_leaf_categories_only(dataset: dict[str, pd.DataFrame]) -> None:
    categories = dataset["categories"]
    leaves = set(categories.loc[categories["depth"] == 2, "id"])
    assert set(dataset["products"]["category_id"]) <= leaves


def test_brands_stay_within_their_departments(dataset: dict[str, pd.DataFrame]) -> None:
    from config.taxonomy import BRANDS

    allowed = {b.name: set(b.departments) for b in BRANDS}
    brand_names = dataset["brands"].set_index("id")["name"]
    products = dataset["products"]
    for brand_id, department in zip(products["brand_id"], products["department"], strict=True):
        assert department in allowed[brand_names[brand_id]]


def test_prices_and_costs_are_sane(dataset: dict[str, pd.DataFrame]) -> None:
    products = dataset["products"]
    assert (products["price"] > 0).all()
    assert (products["cost"] >= 0).all()
    assert (products["cost"] < products["price"]).all()


def test_ratings_are_in_range(dataset: dict[str, pd.DataFrame]) -> None:
    ratings = dataset["products"]["rating_average"]
    assert ratings.between(1.0, 5.0).all()


def test_order_totals_reconcile(dataset: dict[str, pd.DataFrame]) -> None:
    orders, items = dataset["orders"], dataset["order_items"]
    computed = items.groupby("order_id")["line_total"].sum().round(2)
    stated = orders.set_index("id")["subtotal"]
    joined = pd.concat([computed.rename("computed"), stated.rename("stated")], axis=1).dropna()
    assert ((joined["computed"] - joined["stated"]).abs() < 0.02).all()


def test_events_respect_the_product_scoped_constraint(
    dataset: dict[str, pd.DataFrame],
) -> None:
    """Mirrors the database CHECK constraint on `user_events`."""
    product_scoped = {
        "PRODUCT_VIEW", "PRODUCT_CLICK", "ADD_TO_CART", "REMOVE_FROM_CART",
        "WISHLIST", "PURCHASE", "PRODUCT_SHARE", "PRODUCT_RATING", "PRODUCT_REVIEW",
    }
    events = dataset["events"]
    scoped = events[events["event_type"].isin(product_scoped)]
    assert scoped["product_id"].notna().all()

    unscoped = events[events["event_type"].isin({"SEARCH", "SESSION_START", "SESSION_END"})]
    assert unscoped["product_id"].isna().all()


def test_events_stay_inside_the_simulation_window(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    assert events["occurred_at"].min().date() >= SMALL.start_date
    assert events["occurred_at"].max().date() <= SMALL.end_date


def test_no_product_is_viewed_before_release(dataset: dict[str, pd.DataFrame]) -> None:
    """A view before the release date would make new-product cold start fake."""
    released = dataset["products"].set_index("id")["released_at"]
    views = dataset["events"]
    views = views[views["event_type"] == "PRODUCT_VIEW"]
    release_dates = views["product_id"].map(released)
    assert (views["occurred_at"].dt.date.to_numpy() >= release_dates.to_numpy()).all()


def test_sessions_have_matching_events(dataset: dict[str, pd.DataFrame]) -> None:
    sessions = set(dataset["user_sessions"]["session_key"])
    assert set(dataset["events"]["session_key"]) == sessions


def test_every_session_starts_and_ends(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    starts = (events["event_type"] == "SESSION_START").sum()
    ends = (events["event_type"] == "SESSION_END").sum()
    assert starts == ends == len(dataset["user_sessions"])


# ---------------------------------------------------------------------------
# Learnable structure
# ---------------------------------------------------------------------------


def test_users_have_concentrated_category_taste(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    views = events[events["event_type"] == "PRODUCT_VIEW"]
    departments = dataset["products"].set_index("id")["department"]
    frame = views.assign(department=views["product_id"].map(departments))

    counts = frame.groupby(["user_id", "department"]).size().rename("n").reset_index()
    totals = counts.groupby("user_id")["n"].transform("sum")
    top_share = (counts["n"] / totals).groupby(counts["user_id"]).max()

    # Uniform behaviour over 8 departments would give 0.125.
    assert top_share.median() > 0.4


def test_preferred_categories_convert_better(dataset: dict[str, pd.DataFrame]) -> None:
    profiles = dataset["latent_user_profiles"].set_index("user_id")["top_department"]
    departments = dataset["products"].set_index("id")["department"]
    events = dataset["events"]
    scoped = events[events["event_type"].isin({"PRODUCT_VIEW", "PURCHASE"})].copy()
    scoped["department"] = scoped["product_id"].map(departments)
    scoped["preferred"] = scoped["department"] == scoped["user_id"].map(profiles)

    def rate(preferred: bool) -> float:
        subset = scoped[scoped["preferred"] == preferred]
        views = int((subset["event_type"] == "PRODUCT_VIEW").sum())
        buys = int((subset["event_type"] == "PURCHASE").sum())
        return buys / views if views else 0.0

    assert rate(True) > rate(False) * 1.3


def test_complementary_pairs_are_co_purchased(dataset: dict[str, pd.DataFrame]) -> None:
    subcategory = dataset["products"].set_index("id")["subcategory"]
    items = dataset["order_items"].assign(
        subcategory=dataset["order_items"]["product_id"].map(subcategory)
    )
    baskets = items.groupby("order_id")["subcategory"].agg(set)
    n_orders = len(baskets)
    assert n_orders > 50

    presence: dict[str, int] = {}
    for basket in baskets:
        for sub in basket:
            presence[sub] = presence.get(sub, 0) + 1

    lifts = []
    for left, right, _ in COMPLEMENTARY_PAIRS:
        n_left, n_right = presence.get(left, 0), presence.get(right, 0)
        if n_left < 5 or n_right < 5:
            continue
        both = sum(1 for basket in baskets if left in basket and right in basket)
        lifts.append((both / n_left) / (n_right / n_orders))

    assert lifts, "no complementary pair had enough support to measure"
    assert float(np.median(lifts)) > 2.0


def test_consecutive_views_are_related(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    views = events[events["event_type"] == "PRODUCT_VIEW"].sort_values(
        ["session_key", "occurred_at"]
    )
    subcategory = dataset["products"].set_index("id")["subcategory"]
    labels = views["product_id"].map(subcategory).to_numpy()
    sessions = views["session_key"].to_numpy()

    same_session = sessions[1:] == sessions[:-1]
    same_sub = (labels[1:] == labels[:-1])[same_session]
    shares = views["product_id"].map(subcategory).value_counts(normalize=True).to_numpy()

    assert same_sub.mean() > 5 * float((shares**2).sum())


def test_segments_differ_in_spend(dataset: dict[str, pd.DataFrame]) -> None:
    spend = dataset["orders"].groupby("user_id")["grand_total"].sum()
    users = dataset["users"].assign(spend=dataset["users"]["id"].map(spend).fillna(0.0))
    by_segment = users.groupby("segment")["spend"].mean()
    assert by_segment["high_value"] > by_segment["casual"] * 3


def test_interaction_matrix_is_sparse(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    interactions = events[events["product_id"].notna()]
    pairs = interactions.groupby(["user_id", "product_id"]).size()
    density = len(pairs) / (SMALL.n_users * SMALL.n_products)
    assert density < 0.05


def test_cold_users_and_products_exist(dataset: dict[str, pd.DataFrame]) -> None:
    events = dataset["events"]
    active = set(events["user_id"].dropna().unique())
    assert len(active) < SMALL.n_users, "every user is active; cold start is untestable"

    late = dataset["products"]
    late_released = late[late["released_at"] >= SMALL.start_date]
    assert len(late_released) > 0, "no product is released during the window"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_json_columns_round_trip(dataset: dict[str, pd.DataFrame]) -> None:
    """Attributes and metadata must survive the Parquet JSON encoding."""
    attributes = dataset["products"]["attributes"].iloc[0]
    assert json.loads(json.dumps(attributes)) == attributes

    metadata = dataset["events"]["event_metadata"].iloc[0]
    assert json.loads(json.dumps(metadata)) == metadata


def test_latent_profiles_are_not_part_of_the_loadable_dataset() -> None:
    """The latent taste vectors must never reach the database.

    They are ground truth for the diagnostic only. If they were loaded, any
    model using them would be trained on the answer key and every offline
    metric would be meaningless.
    """
    from scripts.seed_database import LOAD_PLAN

    loaded_files = {file_name for file_name, _, _ in LOAD_PLAN}
    assert "latent_user_profiles" not in loaded_files


def test_users_carry_onboarding_preferences(dataset: dict[str, pd.DataFrame]) -> None:
    onboarding = dataset["users"]["onboarding_categories"]
    assert onboarding.map(len).gt(0).mean() > 0.9
