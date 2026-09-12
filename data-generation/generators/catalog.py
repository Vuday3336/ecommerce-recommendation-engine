"""Catalogue generation: categories, brands, products and variants."""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from config.simulation import SimulationConfig
from config.taxonomy import (
    ATTRIBUTE_VALUES,
    BRANDS,
    DEPARTMENTS,
    TIER_PRICE_MULTIPLIER,
    TIER_RATING_MEAN,
    brands_for_department,
    iter_subcategories,
)

#: Absolute price tiers, used for the `price_band` column. Kept absolute
#: because that is what a shopper perceives and what a business rule filters
#: on. The simulator separately uses a *within-subcategory* price percentile
#: for price-fit, because "expensive for a book" and "expensive for a laptop"
#: are different judgements and conflating them would make price sensitivity
#: an artefact of which department a user shops in.
#: Ratings accrued per day of pre-window catalogue age, before the quality
#: multiplier. Tuned so a two-year-old product carries a few dozen ratings.
PRIOR_RATINGS_PER_DAY = 0.035

PRICE_BAND_BOUNDS: tuple[tuple[str, float], ...] = (
    ("budget", 25.0),
    ("mid", 100.0),
    ("premium", 400.0),
    ("luxury", math.inf),
)

QUALITY_ADJECTIVES: tuple[str, ...] = (
    "Classic", "Pro", "Lite", "Everyday", "Essential", "Signature", "Studio",
    "Trail", "Urban", "Compact", "Ultra", "Core", "Prime", "Heritage",
)

MODEL_SUFFIXES: tuple[str, ...] = (
    "100", "200", "300", "450", "500", "X", "XT", "S", "SE", "Plus", "Max",
    "Mk II", "Mk III", "One", "Two", "Air", "Edge",
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _price_band(price: float) -> str:
    for band, upper in PRICE_BAND_BOUNDS:
        if price < upper:
            return band
    return "luxury"


@dataclass(slots=True)
class Catalogue:
    """Generated catalogue plus the lookup structures the simulator needs."""

    categories: pd.DataFrame
    brands: pd.DataFrame
    products: pd.DataFrame
    variants: pd.DataFrame

    #: subcategory name -> array of product ids
    products_by_subcategory: dict[str, np.ndarray]
    #: department name -> array of product ids
    products_by_department: dict[str, np.ndarray]
    #: brand id -> array of product ids
    products_by_brand: dict[int, np.ndarray]
    #: leaf category id -> (department, category, subcategory)
    leaf_lookup: dict[int, tuple[str, str, str]]

    def __len__(self) -> int:
        return len(self.products)


def generate_categories() -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the three-level category tree.

    Returns the dataframe and a map from subcategory name to its leaf id.
    """
    rows: list[dict] = []
    leaf_ids: dict[str, int] = {}
    next_id = 1

    for department in DEPARTMENTS:
        dept_id = next_id
        next_id += 1
        rows.append(
            {
                "id": dept_id,
                "name": department.name,
                "slug": _slugify(department.name),
                "parent_id": None,
                "depth": 0,
                "path": _slugify(department.name),
            }
        )
        for category in department.categories:
            cat_id = next_id
            next_id += 1
            cat_path = f"{_slugify(department.name)}/{_slugify(category.name)}"
            rows.append(
                {
                    "id": cat_id,
                    "name": category.name,
                    "slug": _slugify(f"{department.name}-{category.name}"),
                    "parent_id": dept_id,
                    "depth": 1,
                    "path": cat_path,
                }
            )
            for sub in category.subcategories:
                sub_id = next_id
                next_id += 1
                rows.append(
                    {
                        "id": sub_id,
                        "name": sub.name,
                        "slug": _slugify(f"{department.name}-{sub.name}"),
                        "parent_id": cat_id,
                        "depth": 2,
                        "path": f"{cat_path}/{_slugify(sub.name)}",
                    }
                )
                leaf_ids[sub.name] = sub_id

    return pd.DataFrame(rows), leaf_ids


def generate_brands() -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    brand_ids: dict[str, int] = {}
    for index, brand in enumerate(BRANDS, start=1):
        brand_ids[brand.name] = index
        rows.append(
            {
                "id": index,
                "name": brand.name,
                "slug": _slugify(brand.name),
                "price_tier": brand.tier,
            }
        )
    return pd.DataFrame(rows), brand_ids


def _allocate_products(rng: np.random.Generator, n_products: int) -> dict[str, int]:
    """Decide how many products each subcategory gets.

    Allocation is proportional to a Dirichlet draw rather than uniform, because
    a real catalogue is lopsided: some subcategories carry hundreds of SKUs and
    others a handful. A uniform catalogue would make popularity and coverage
    metrics behave unrealistically well.
    """
    subs = iter_subcategories()
    weights = rng.dirichlet(np.full(len(subs), 2.2))
    raw = weights * n_products
    counts = np.maximum(np.floor(raw).astype(int), 4)

    # Reconcile to exactly n_products by adjusting the largest allocations.
    diff = n_products - int(counts.sum())
    order = np.argsort(-counts)
    i = 0
    while diff != 0:
        idx = order[i % len(order)]
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        elif counts[idx] > 4:
            counts[idx] -= 1
            diff += 1
        i += 1

    return {sub.name: int(count) for (_, _, sub), count in zip(subs, counts, strict=True)}


def _build_description(
    brand: str,
    subcategory: str,
    department: str,
    attributes: dict[str, str],
    quality: float,
) -> str:
    """Compose product text.

    The description is built from the same facts as the structured attributes,
    so the text embedding and the categorical features describe one product
    rather than two unrelated ones. That coherence is what lets the content
    model beat the popularity baseline on the item-to-item task; random
    lorem-ipsum text would leave the embedding space meaningless.
    """
    attr_phrases = [f"{key.replace('_', ' ')}: {value}" for key, value in attributes.items()]
    quality_phrase = (
        "Built to last with premium materials."
        if quality > 0.72
        else "Reliable everyday quality at a fair price."
        if quality > 0.45
        else "A straightforward, no-frills option."
    )
    return (
        f"{brand} {subcategory.lower()} from our {department.lower()} range. "
        f"{quality_phrase} "
        f"Specifications - {'; '.join(attr_phrases)}."
        if attr_phrases
        else f"{brand} {subcategory.lower()} from our {department.lower()} range. {quality_phrase}"
    )


def generate_products(
    config: SimulationConfig,
    rng: np.random.Generator,
    leaf_ids: dict[str, int],
    brand_ids: dict[str, int],
) -> pd.DataFrame:
    """Generate the product catalogue."""
    allocation = _allocate_products(rng, config.n_products)
    subs = iter_subcategories()

    start = config.start_date
    end = config.end_date
    window_days = (end - start).days

    rows: list[dict] = []
    product_id = 1

    for department_name, category_name, sub in subs:
        department = next(d for d in DEPARTMENTS if d.name == department_name)
        eligible_brands = brands_for_department(department_name)
        count = allocation[sub.name]

        for _ in range(count):
            brand = eligible_brands[rng.integers(len(eligible_brands))]

            # Price: lognormal around department base x subcategory multiplier
            # x brand tier multiplier. Lognormal because prices are positive,
            # right-skewed and multiplicative in exactly this way.
            centre = (
                department.base_price
                * sub.price_multiplier
                * TIER_PRICE_MULTIPLIER[brand.tier]
            )
            price = float(rng.lognormal(mean=math.log(centre), sigma=0.28))
            price = round(max(price, 1.5), 2)
            margin = float(rng.uniform(0.22, 0.58))
            cost = round(price * (1.0 - margin), 2)

            quality = float(np.clip(rng.beta(4.0, 2.4), 0.02, 0.995))
            rating_mean = TIER_RATING_MEAN[brand.tier] + (quality - 0.6) * 1.1
            rating_average = float(np.clip(rng.normal(rating_mean, 0.24), 1.0, 5.0))

            attributes = {
                key: ATTRIBUTE_VALUES[key][rng.integers(len(ATTRIBUTE_VALUES[key]))]
                for key in sub.attributes
                if key in ATTRIBUTE_VALUES
            }

            # Release date: most products predate the window, a configurable
            # slice arrives during it and forms the new-product cold-start set.
            if rng.random() < config.late_release_fraction:
                offset = int(rng.integers(1, max(window_days - 5, 2)))
                released_at = start + dt.timedelta(days=offset)
            else:
                released_at = start - dt.timedelta(days=int(rng.integers(1, 900)))

            adjective = QUALITY_ADJECTIVES[rng.integers(len(QUALITY_ADJECTIVES))]
            suffix = MODEL_SUFFIXES[rng.integers(len(MODEL_SUFFIXES))]
            singular = sub.name[:-1] if sub.name.endswith("s") else sub.name
            name = f"{brand.name} {adjective} {singular} {suffix}"

            rows.append(
                {
                    "id": product_id,
                    "sku": f"SKU-{product_id:06d}",
                    "name": name,
                    "description": _build_description(
                        brand.name, sub.name, department_name, attributes, quality
                    ),
                    "category_id": leaf_ids[sub.name],
                    "brand_id": brand_ids[brand.name],
                    "price": price,
                    "cost": cost,
                    "price_band": _price_band(price),
                    "attributes": attributes,
                    "tags": [
                        _slugify(department_name),
                        _slugify(sub.name),
                        brand.tier,
                    ],
                    "stock_quantity": int(rng.integers(0, 400)),
                    "rating_average": round(rating_average, 2),
                    # Ratings accumulated *before* the simulation window. A
                    # product listed 900 days ago has a review history; starting
                    # every product at zero would leave the catalogue with under
                    # one rating each, which collapses any smoothed quality
                    # feature to the catalogue mean and makes it useless to the
                    # ranker. Rate scales with age and with intrinsic quality,
                    # since better products are both bought and reviewed more.
                    "rating_count": int(
                        rng.poisson(
                            max(
                                0.0,
                                (start - released_at).days
                                * PRIOR_RATINGS_PER_DAY
                                * (0.4 + quality),
                            )
                        )
                    ),
                    "released_at": released_at,
                    # --- simulator-only latent columns -----------------------
                    "department": department_name,
                    "category": category_name,
                    "subcategory": sub.name,
                    "brand_tier": brand.tier,
                    "quality": quality,
                    "repeat_rate": sub.repeat_rate,
                    "repeat_interval_days": sub.repeat_interval_days,
                    "season_phase": department.season_phase,
                    "season_amplitude": department.season_amplitude,
                }
            )
            product_id += 1

    products = pd.DataFrame(rows)

    # Price percentile *within subcategory*: the basis for price fit.
    products["price_percentile"] = products.groupby("subcategory")["price"].rank(pct=True)
    return products


def generate_variants(
    config: SimulationConfig, rng: np.random.Generator, products: pd.DataFrame
) -> pd.DataFrame:
    """Generate purchasable variants.

    Sized products (apparel, footwear) get a size ladder; everything else gets
    a colour or capacity ladder or a single default variant. Variants exist so
    stock and orders are modelled at the grain a real store uses, while every
    model still works at the product grain.
    """
    rows: list[dict] = []
    variant_id = 1

    sizes = ATTRIBUTE_VALUES["size"]
    colours = ATTRIBUTE_VALUES["colour"]

    for row in products.itertuples(index=False):
        attributes: dict = row.attributes
        if "size" in attributes:
            pool = [{"size": s} for s in sizes]
        elif "colour" in attributes:
            pool = [{"colour": c} for c in colours]
        elif "capacity" in attributes:
            pool = [{"capacity": c} for c in ATTRIBUTE_VALUES["capacity"]]
        else:
            pool = [{}]

        n_variants = int(
            rng.integers(
                config.variants_per_product_min,
                min(config.variants_per_product_max, len(pool)) + 1,
            )
        )
        chosen = rng.choice(len(pool), size=min(n_variants, len(pool)), replace=False)

        for order, index in enumerate(sorted(int(i) for i in np.atleast_1d(chosen))):
            options = pool[index]
            label = ", ".join(f"{k}: {v}" for k, v in options.items()) or "Default"
            rows.append(
                {
                    "id": variant_id,
                    "product_id": row.id,
                    "sku": f"{row.sku}-V{order + 1}",
                    "variant_name": label,
                    "options": options,
                    "price_delta": 0.0,
                    "stock_quantity": int(rng.integers(0, 120)),
                }
            )
            variant_id += 1

    return pd.DataFrame(rows)


def build_catalogue(config: SimulationConfig, rng: np.random.Generator) -> Catalogue:
    """Generate the full catalogue and its lookup indexes."""
    categories, leaf_ids = generate_categories()
    brands, brand_ids = generate_brands()
    products = generate_products(config, rng, leaf_ids, brand_ids)
    variants = generate_variants(config, rng, products)

    products_by_subcategory = {
        name: group["id"].to_numpy()
        for name, group in products.groupby("subcategory", sort=False)
    }
    products_by_department = {
        name: group["id"].to_numpy()
        for name, group in products.groupby("department", sort=False)
    }
    products_by_brand = {
        int(brand_id): group["id"].to_numpy()
        for brand_id, group in products.groupby("brand_id", sort=False)
    }
    leaf_lookup = {
        leaf_ids[sub.name]: (dept, cat, sub.name)
        for dept, cat, sub in iter_subcategories()
    }

    return Catalogue(
        categories=categories,
        brands=brands,
        products=products,
        variants=variants,
        products_by_subcategory=products_by_subcategory,
        products_by_department=products_by_department,
        products_by_brand=products_by_brand,
        leaf_lookup=leaf_lookup,
    )


__all__ = [
    "PRICE_BAND_BOUNDS",
    "Catalogue",
    "build_catalogue",
    "generate_brands",
    "generate_categories",
    "generate_products",
    "generate_variants",
]
