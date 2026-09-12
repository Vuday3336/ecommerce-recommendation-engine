"""Phase 2 exit gate: is the generated behaviour actually learnable?

Generating 400,000 rows proves nothing. If the behaviour is noise dressed up as
a taxonomy, every model in Phases 5-14 scores at chance, the hybrid cannot beat
the baseline, and the evaluation report becomes fiction. That is risk R-02, and
this script is its mitigation.

Each check states a hypothesis about structure that was deliberately injected,
measures whether it is *recoverable from the behaviour alone*, and compares the
measurement against a threshold. The latent user profiles are read here - and
only here - to confirm that revealed behaviour tracks hidden taste. They are
never loaded into the database and never reach a model.

    python data-generation/diagnostics.py
    python data-generation/diagnostics.py --data data/synthetic

Exit code 0 means the dataset is fit for modelling; 1 means it is not, and
Phase 2 has not passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "synthetic"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.taxonomy import COMPLEMENTARY_PAIRS

PRODUCT_EVENTS = (
    "PRODUCT_VIEW",
    "PRODUCT_CLICK",
    "ADD_TO_CART",
    "WISHLIST",
    "PURCHASE",
)


@dataclass(slots=True)
class Check:
    name: str
    hypothesis: str
    measured: float
    threshold: float
    comparison: str  # ">=" or "<="
    detail: str = ""

    @property
    def passed(self) -> bool:
        if self.comparison == ">=":
            return self.measured >= self.threshold
        return self.measured <= self.threshold

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = (
            f"[{status}] {self.name}\n"
            f"        hypothesis: {self.hypothesis}\n"
            f"        measured:   {self.measured:.4f}  "
            f"(needs {self.comparison} {self.threshold})"
        )
        if self.detail:
            line += f"\n        note:       {self.detail}"
        return line


def gini(values: np.ndarray) -> float:
    """Concentration of a distribution. 0 = perfectly even, 1 = one winner."""
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values.astype(float))
    n = len(sorted_values)
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * sorted_values).sum()) / (n * total) - (n + 1.0) / n)


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "products",
        "categories",
        "users",
        "events",
        "orders",
        "order_items",
        "user_sessions",
        "latent_user_profiles",
    ):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run: python data-generation/generate.py"
            )
        frames[name] = pd.read_parquet(path)
    frames["events"]["event_metadata"] = frames["events"]["event_metadata"].map(json.loads)
    return frames


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_user_concentration(interactions: pd.DataFrame) -> Check:
    """Users must have concentrated, not uniform, category taste."""
    per_user = (
        interactions.groupby(["user_id", "department"]).size().rename("n").reset_index()
    )
    totals = per_user.groupby("user_id")["n"].transform("sum")
    per_user["share"] = per_user["n"] / totals
    top_share = per_user.groupby("user_id")["share"].max()
    engaged = top_share[
        interactions.groupby("user_id").size().reindex(top_share.index).fillna(0) >= 5
    ]
    median = float(engaged.median())
    n_departments = interactions["department"].nunique()
    return Check(
        name="User category concentration",
        hypothesis=(
            "each user's interactions concentrate in a few departments, so "
            "category affinity is a usable signal"
        ),
        measured=median,
        threshold=0.45,
        comparison=">=",
        detail=(
            f"median top-department share among users with 5+ interactions; "
            f"uniform behaviour over {n_departments} departments would give "
            f"{1 / n_departments:.3f}"
        ),
    )


def check_affinity_conversion_lift(
    interactions: pd.DataFrame, profiles: pd.DataFrame
) -> Check:
    """Conversion must be higher inside a user's preferred department."""
    top_department = profiles.set_index("user_id")["top_department"]
    frame = interactions.copy()
    frame["is_preferred"] = (
        frame["department"] == frame["user_id"].map(top_department)
    )

    views = frame[frame["event_type"] == "PRODUCT_VIEW"]
    purchases = frame[frame["event_type"] == "PURCHASE"]

    def rate(preferred: bool) -> float:
        v = int((views["is_preferred"] == preferred).sum())
        p = int((purchases["is_preferred"] == preferred).sum())
        return p / v if v else 0.0

    preferred_rate = rate(True)
    other_rate = rate(False)
    lift = preferred_rate / other_rate if other_rate > 0 else float("inf")

    return Check(
        name="Affinity conversion lift",
        hypothesis=(
            "products in a user's preferred department convert better than "
            "products outside it"
        ),
        measured=lift,
        threshold=1.35,
        comparison=">=",
        detail=(
            f"view-to-purchase inside preferred department {preferred_rate:.4f} "
            f"vs outside {other_rate:.4f}"
        ),
    )


def check_brand_affinity(interactions: pd.DataFrame, profiles: pd.DataFrame) -> Check:
    """Interactions must over-index on each user's affine brands."""
    affinity = profiles.set_index("user_id")["brand_affinity"].map(
        lambda raw: {int(k) for k in json.loads(raw)} if isinstance(raw, str) else set(raw)
    )
    frame = interactions[interactions["user_id"].isin(affinity.index)].copy()
    frame["affine"] = [
        brand in affinity.get(user, set())
        for user, brand in zip(frame["user_id"], frame["brand_id"], strict=True)
    ]
    observed = float(frame["affine"].mean())

    # Baseline: the share of the catalogue those brands represent, averaged
    # over users. Anything above this is genuine brand pull rather than an
    # artefact of some brands simply having more products.
    n_brands = interactions["brand_id"].nunique()
    mean_affine_brands = float(affinity.map(len).mean())
    baseline = mean_affine_brands / n_brands

    return Check(
        name="Brand affinity",
        hypothesis="users over-index on the brands they have an affinity for",
        measured=observed / baseline if baseline > 0 else 0.0,
        threshold=1.8,
        comparison=">=",
        detail=(
            f"{observed:.3f} of interactions are with affine brands vs a "
            f"catalogue-share baseline of {baseline:.3f}"
        ),
    )


def check_price_alignment(
    order_items: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame, profiles: pd.DataFrame
) -> Check:
    """Realised purchase prices must track the latent price target."""
    percentile = products.set_index("id")["price_percentile"]
    items = order_items.merge(
        orders[["id", "user_id"]], left_on="order_id", right_on="id", suffixes=("", "_order")
    )
    items["percentile"] = items["product_id"].map(percentile)
    realised = items.groupby("user_id")["percentile"].mean()

    latent = profiles.set_index("user_id")["price_target_percentile"]
    joined = pd.concat([realised.rename("realised"), latent.rename("latent")], axis=1).dropna()
    correlation = float(joined["realised"].corr(joined["latent"])) if len(joined) > 30 else 0.0

    return Check(
        name="Price preference alignment",
        hypothesis=(
            "the price percentile a user actually buys at tracks their latent "
            "price target, so price-fit is a learnable feature"
        ),
        measured=correlation,
        threshold=0.30,
        comparison=">=",
        detail=f"Pearson correlation over {len(joined):,} purchasing users",
    )


def check_copurchase_lift(
    order_items: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator
) -> Check:
    """Declared complementary pairs must show real co-purchase lift."""
    subcategory = products.set_index("id")["subcategory"]
    items = order_items.copy()
    items["subcategory"] = items["product_id"].map(subcategory)

    baskets = items.groupby("order_id")["subcategory"].agg(set)
    n_orders = len(baskets)
    if n_orders == 0:
        return Check("Co-purchase lift", "", 0.0, 1.0, ">=", "no orders")

    presence: dict[str, int] = {}
    for basket in baskets:
        for sub in basket:
            presence[sub] = presence.get(sub, 0) + 1

    def lift(left: str, right: str) -> float | None:
        n_left = presence.get(left, 0)
        n_right = presence.get(right, 0)
        if n_left < 20 or n_right < 20:
            return None
        both = sum(1 for basket in baskets if left in basket and right in basket)
        p_right_given_left = both / n_left
        p_right = n_right / n_orders
        return p_right_given_left / p_right if p_right > 0 else None

    declared = [
        value
        for left, right, _ in COMPLEMENTARY_PAIRS
        if (value := lift(left, right)) is not None
    ]

    subcategories = sorted(presence)
    random_lifts: list[float] = []
    attempts = 0
    while len(random_lifts) < 200 and attempts < 4000:
        attempts += 1
        a, b = rng.choice(len(subcategories), size=2, replace=False)
        left, right = subcategories[int(a)], subcategories[int(b)]
        if any(left == x and right == y for x, y, _ in COMPLEMENTARY_PAIRS):
            continue
        value = lift(left, right)
        if value is not None:
            random_lifts.append(value)

    declared_median = float(np.median(declared)) if declared else 0.0
    random_median = float(np.median(random_lifts)) if random_lifts else 0.0
    random_p90 = float(np.quantile(random_lifts, 0.90)) if random_lifts else 0.0

    # Lift is already normalised - P(B|A) / P(B) - so it is compared against an
    # absolute threshold rather than against the random baseline. Dividing by
    # the random median would be worse, not better: most unrelated subcategory
    # pairs never co-occur at all, so that denominator is legitimately zero and
    # the ratio is undefined exactly when the signal is strongest. The random
    # distribution is reported as context instead.
    return Check(
        name="Frequently-bought-together signal",
        hypothesis=(
            "subcategory pairs declared complementary are co-purchased far more "
            "often than unrelated pairs, so FBT has ground truth to recover"
        ),
        measured=declared_median,
        threshold=3.0,
        comparison=">=",
        detail=(
            f"median lift {declared_median:.2f} over {len(declared)} declared pairs; "
            f"unrelated pairs ({len(random_lifts)} sampled) have median "
            f"{random_median:.2f} and 90th percentile {random_p90:.2f}"
        ),
    )


def check_covisitation(events: pd.DataFrame, products: pd.DataFrame) -> Check:
    """Consecutive views inside a session must be related, not independent."""
    views = events[events["event_type"] == "PRODUCT_VIEW"].sort_values(
        ["session_key", "occurred_at"]
    )
    subcategory = products.set_index("id")["subcategory"]
    views = views.assign(subcategory=views["product_id"].map(subcategory))

    current = views["subcategory"].to_numpy()
    session = views["session_key"].to_numpy()
    same_session = session[1:] == session[:-1]
    same_subcategory = current[1:] == current[:-1]
    consecutive = same_subcategory[same_session]
    observed = float(consecutive.mean()) if consecutive.size else 0.0

    shares = views["subcategory"].value_counts(normalize=True).to_numpy()
    chance = float((shares**2).sum())

    return Check(
        name="Co-visitation structure",
        hypothesis=(
            "consecutive product views in a session stay related, so the "
            "co-visitation matrix carries signal"
        ),
        measured=observed / chance if chance > 0 else 0.0,
        threshold=6.0,
        comparison=">=",
        detail=(
            f"{observed:.3f} of consecutive view pairs share a subcategory vs "
            f"{chance:.4f} expected under independence"
        ),
    )


def check_segment_separation(
    orders: pd.DataFrame, users: pd.DataFrame, events: pd.DataFrame
) -> Check:
    """Behavioural segments must be behaviourally distinct."""
    spend = orders.groupby("user_id")["grand_total"].sum()
    frame = users[["id", "segment"]].copy()
    frame["spend"] = frame["id"].map(spend).fillna(0.0)
    by_segment = frame.groupby("segment")["spend"].mean()

    high = float(by_segment.get("high_value", 0.0))
    window = float(by_segment.get("window_shopper", 0.0))
    ratio = high / window if window > 0 else float("inf")

    detail = ", ".join(f"{k}={v:,.0f}" for k, v in by_segment.sort_values(ascending=False).items())
    return Check(
        name="Segment separation",
        hypothesis="declared segments differ materially in realised spend",
        measured=ratio,
        threshold=5.0,
        comparison=">=",
        detail=f"mean spend by segment: {detail}",
    )


def check_popularity_skew(interactions: pd.DataFrame, products: pd.DataFrame) -> Check:
    """Popularity must be unequal - that is the bias the system has to fight."""
    counts = (
        interactions[interactions["event_type"] == "PRODUCT_VIEW"]["product_id"]
        .value_counts()
        .reindex(products["id"], fill_value=0)
        .to_numpy()
    )
    value = gini(counts)
    top_1pct = math.ceil(len(counts) * 0.01)
    share = float(np.sort(counts)[::-1][:top_1pct].sum() / max(counts.sum(), 1))
    return Check(
        name="Popularity skew",
        hypothesis=(
            "view popularity is heavily unequal, so popularity bias and "
            "coverage are real problems rather than assumed ones"
        ),
        measured=value,
        threshold=0.45,
        comparison=">=",
        detail=f"Gini over product view counts; top 1% of products take {share:.1%} of views",
    )


def check_matrix_sparsity(interactions: pd.DataFrame, users: pd.DataFrame, products: pd.DataFrame) -> Check:
    """The user-item matrix must be sparse, like a real one."""
    pairs = interactions.groupby(["user_id", "product_id"]).size()
    density = len(pairs) / (len(users) * len(products))
    return Check(
        name="Interaction matrix sparsity",
        hypothesis=(
            "the user-item matrix is sparse, so collaborative filtering faces "
            "the problem it exists to solve"
        ),
        measured=density,
        threshold=0.01,
        comparison="<=",
        detail=(
            f"{len(pairs):,} distinct (user, product) pairs out of "
            f"{len(users) * len(products):,} cells = {density:.5%} dense"
        ),
    )


def check_repeat_purchase(order_items: pd.DataFrame, orders: pd.DataFrame) -> Check:
    """Consumables must actually be re-purchased."""
    items = order_items.merge(
        orders[["id", "user_id"]], left_on="order_id", right_on="id", suffixes=("", "_o")
    )
    per_pair = items.groupby(["user_id", "product_id"]).size()
    repeat_share = float((per_pair > 1).mean()) if len(per_pair) else 0.0
    return Check(
        name="Repeat purchase behaviour",
        hypothesis=(
            "some products are bought repeatedly by the same user, so repeat "
            "rate and recency are predictive"
        ),
        measured=repeat_share,
        threshold=0.01,
        comparison=">=",
        detail=f"{repeat_share:.2%} of (user, product) purchase pairs occur more than once",
    )


def check_cold_start_populations(
    users: pd.DataFrame, products: pd.DataFrame, interactions: pd.DataFrame
) -> Check:
    """There must be genuinely cold users and products to test against."""
    active_users = set(interactions["user_id"].unique())
    touched_products = set(interactions["product_id"].unique())
    cold_users = len(users) - len(active_users)
    cold_products = len(products) - len(touched_products)
    sparse_users = int(interactions.groupby("user_id").size().le(3).sum())

    share = (cold_users + sparse_users) / len(users)
    return Check(
        name="Cold-start population",
        hypothesis=(
            "a meaningful share of users and products have little or no "
            "history, so cold start is exercised rather than assumed"
        ),
        measured=share,
        threshold=0.05,
        comparison=">=",
        detail=(
            f"{cold_users:,} users with no interactions, {sparse_users:,} with 3 or "
            f"fewer, {cold_products:,} products never interacted with"
        ),
    )


def check_temporal_coverage(events: pd.DataFrame) -> Check:
    """Activity must span the window with variation, not sit in a spike."""
    daily = events.set_index("occurred_at").resample("D").size()
    active_days = int((daily > 0).sum())
    coverage = active_days / max(len(daily), 1)
    variation = float(daily.std() / daily.mean()) if daily.mean() > 0 else 0.0
    return Check(
        name="Temporal coverage",
        hypothesis=(
            "events span the whole window, so a temporal train/validation/test "
            "split has data in every fold"
        ),
        measured=coverage,
        threshold=0.98,
        comparison=">=",
        detail=(
            f"{active_days} active days of {len(daily)}; daily coefficient of "
            f"variation {variation:.3f}"
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_interactions(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    frame = events[events["event_type"].isin(PRODUCT_EVENTS)].copy()
    frame = frame[frame["product_id"].notna()]
    frame["product_id"] = frame["product_id"].astype("int64")
    meta = products.set_index("id")[["department", "subcategory", "brand_id"]]
    return frame.join(meta, on="product_id")


def run(data_dir: Path) -> int:
    frames = load(data_dir)
    rng = np.random.default_rng(0)

    products = frames["products"]
    events = frames["events"]
    users = frames["users"]
    orders = frames["orders"]
    order_items = frames["order_items"]
    profiles = frames["latent_user_profiles"]

    interactions = build_interactions(events, products)

    checks = [
        check_user_concentration(interactions),
        check_affinity_conversion_lift(interactions, profiles),
        check_brand_affinity(interactions, profiles),
        check_price_alignment(order_items, products, orders, profiles),
        check_copurchase_lift(order_items, products, rng),
        check_covisitation(events, products),
        check_segment_separation(orders, users, events),
        check_popularity_skew(interactions, products),
        check_matrix_sparsity(interactions, users, products),
        check_repeat_purchase(order_items, orders),
        check_cold_start_populations(users, products, interactions),
        check_temporal_coverage(events),
    ]

    print("=" * 78)
    print("PHASE 2 DATA DIAGNOSTIC")
    print("Does the generated behaviour contain structure a model can recover?")
    print("=" * 78)
    print()

    for check in checks:
        print(check.render())
        print()

    failed = [c for c in checks if not c.passed]
    print("-" * 78)
    print(f"{len(checks) - len(failed)} of {len(checks)} checks passed.")

    print()
    print("Dataset shape")
    print("-" * 78)
    print(f"  users                {len(users):>10,}")
    print(f"  products             {len(products):>10,}")
    print(f"  events               {len(events):>10,}")
    print(f"  sessions             {len(frames['user_sessions']):>10,}")
    print(f"  orders               {len(orders):>10,}")
    print(f"  order items          {len(order_items):>10,}")
    print(
        f"  window               {events['occurred_at'].min():%Y-%m-%d}"
        f" to {events['occurred_at'].max():%Y-%m-%d}"
    )

    views = int((events["event_type"] == "PRODUCT_VIEW").sum())
    purchases = int((events["event_type"] == "PURCHASE").sum())
    sessions = len(frames["user_sessions"])
    converting = int(frames["user_sessions"]["converted"].sum())
    print()
    print("Funnel (note: rates are higher than a typical real storefront, which")
    print("converts around 2-3% of sessions - see docs/database.md section 6)")
    print("-" * 78)
    print(f"  view -> purchase     {purchases / max(views, 1):>10.2%}")
    print(f"  session conversion   {converting / max(sessions, 1):>10.2%}")
    print(f"  items per order      {len(order_items) / max(len(orders), 1):>10.2f}")

    if failed:
        print()
        print("PHASE 2 GATE: FAILED")
        for check in failed:
            print(f"  - {check.name}")
        return 1

    print()
    print("PHASE 2 GATE: PASSED - the dataset carries recoverable structure.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    return run(args.data)


if __name__ == "__main__":
    raise SystemExit(main())
