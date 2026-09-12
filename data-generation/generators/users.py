"""User generation with latent preference structure.

Every user carries a hidden taste vector that the simulator uses to decide what
they look at and buy, and that the recommendation models must rediscover from
behaviour alone. The latent vectors are written to disk alongside the dataset
so the Phase 2 diagnostic can measure *whether the structure is recoverable* -
but they are never loaded into any feature table or model input. Training on
them would be circular and would make every offline metric meaningless.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from config.simulation import SEGMENT_BEHAVIOUR, SimulationConfig
from config.taxonomy import DEPARTMENTS, brands_for_department

from generators.catalog import Catalogue

FIRST_NAMES: tuple[str, ...] = (
    "Ana", "Ben", "Chloe", "Dev", "Elif", "Farid", "Grace", "Hugo", "Ines",
    "Jonas", "Kira", "Liam", "Maya", "Noor", "Oscar", "Priya", "Quinn", "Rosa",
    "Samir", "Tara", "Umut", "Vera", "Wei", "Xiomara", "Yusuf", "Zara",
    "Adam", "Bella", "Caleb", "Dana", "Eli", "Fiona", "Gabriel", "Hana",
)
LAST_NAMES: tuple[str, ...] = (
    "Alvarez", "Bakker", "Chen", "Dubois", "Eriksen", "Ferreira", "Gupta",
    "Haddad", "Ivanov", "Jensen", "Kowalski", "Lindqvist", "Moreau", "Nakamura",
    "Okafor", "Petrov", "Quintana", "Rossi", "Silva", "Tanaka", "Ueda",
    "Vargas", "Weber", "Xu", "Yilmaz", "Zhang", "Novak", "Murphy", "Kaur",
)
COUNTRIES: tuple[str, ...] = ("US", "GB", "DE", "FR", "ES", "NL", "SE", "CA", "AU", "IN")
COUNTRY_WEIGHTS: tuple[float, ...] = (0.34, 0.14, 0.11, 0.08, 0.06, 0.05, 0.04, 0.07, 0.05, 0.06)
SIGNUP_SOURCES: tuple[str, ...] = ("organic", "paid_search", "social", "referral", "email")
SIGNUP_WEIGHTS: tuple[float, ...] = (0.42, 0.22, 0.16, 0.12, 0.08)
DEVICES: tuple[str, ...] = ("desktop", "mobile", "tablet")
DEVICE_WEIGHTS: tuple[float, ...] = (0.41, 0.51, 0.08)


@dataclass(slots=True)
class UserProfile:
    """Latent state the simulator reads for one user."""

    user_id: int
    segment: str
    signup_date: dt.date
    churn_date: dt.date | None

    #: department name -> preference weight (sums to 1 over chosen departments)
    department_weights: dict[str, float]
    #: subcategory name -> preference weight (sums to 1)
    subcategory_weights: dict[str, float]
    #: brand id -> multiplicative affinity boost
    brand_affinity: dict[int, float]

    #: Where in a subcategory's price distribution this user shops (0 = cheapest).
    price_target_percentile: float
    #: How sharply they are penalised for products away from that target.
    price_sensitivity: float
    exploration: float
    purchase_propensity: float
    sessions_per_week: float
    basket_size: float
    primary_device: str

    #: subcategory name -> array over its product ids, cached selection weights.
    _weight_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def _sample_department_weights(
    rng: np.random.Generator, config: SimulationConfig
) -> dict[str, float]:
    """Concentrate a user's taste on one to three departments."""
    names = [d.name for d in DEPARTMENTS]
    n_choose = int(rng.integers(1, config.preference.max_departments + 1))
    chosen = rng.choice(len(names), size=n_choose, replace=False)
    raw = rng.dirichlet(
        np.full(n_choose, config.preference.department_concentration * 3.0)
    )
    return {names[int(i)]: float(w) for i, w in zip(chosen, raw, strict=True)}


def _sample_subcategory_weights(
    rng: np.random.Generator,
    config: SimulationConfig,
    department_weights: dict[str, float],
) -> dict[str, float]:
    """Spread each department's weight over its subcategories, unevenly."""
    weights: dict[str, float] = {}
    for department_name, dept_weight in department_weights.items():
        department = next(d for d in DEPARTMENTS if d.name == department_name)
        subs = [s.name for c in department.categories for s in c.subcategories]
        raw = rng.dirichlet(
            np.full(len(subs), config.preference.subcategory_concentration)
        )
        for name, w in zip(subs, raw, strict=True):
            weights[name] = float(w) * dept_weight
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _sample_brand_affinity(
    rng: np.random.Generator,
    config: SimulationConfig,
    department_weights: dict[str, float],
    brand_ids: dict[str, int],
) -> dict[int, float]:
    """Pick a few brands the user gravitates to, inside their departments."""
    eligible: list[str] = []
    for department_name in department_weights:
        eligible.extend(b.name for b in brands_for_department(department_name))
    eligible = sorted(set(eligible))
    if not eligible:
        return {}

    n = int(
        rng.integers(
            config.preference.brands_min,
            min(config.preference.brands_max, len(eligible)) + 1,
        )
    )
    chosen = rng.choice(len(eligible), size=n, replace=False)
    affinity: dict[int, float] = {}
    for i in np.atleast_1d(chosen):
        strength = 1.0 + rng.random() * (config.preference.brand_affinity_strength - 1.0)
        affinity[brand_ids[eligible[int(i)]]] = float(strength)
    return affinity


def generate_users(
    config: SimulationConfig, rng: np.random.Generator, catalogue: Catalogue
) -> tuple[pd.DataFrame, dict[int, UserProfile]]:
    """Generate users and their latent profiles."""
    brand_ids = dict(zip(catalogue.brands["name"], catalogue.brands["id"], strict=True))
    leaf_id_by_sub = {
        sub: leaf_id for leaf_id, (_, _, sub) in catalogue.leaf_lookup.items()
    }

    mix = config.users.as_dict()
    segment_names = list(mix)
    segment_probs = np.array([mix[s] for s in segment_names], dtype=float)
    segment_probs /= segment_probs.sum()

    start = config.start_date
    end = config.end_date
    window_days = (end - start).days

    rows: list[dict] = []
    profiles: dict[int, UserProfile] = {}

    for user_id in range(1, config.n_users + 1):
        segment = segment_names[int(rng.choice(len(segment_names), p=segment_probs))]
        behaviour = SEGMENT_BEHAVIOUR[segment]

        # Signup: most users predate the window; a slice joins during it and
        # forms the new-user cold-start population (FR-09).
        if rng.random() < config.late_signup_fraction:
            signup_date = start + dt.timedelta(
                days=int(rng.integers(1, max(window_days - 2, 2)))
            )
        else:
            signup_date = start - dt.timedelta(days=int(rng.integers(1, 1100)))

        # Churn: a per-day hazard, so lapsed users stop generating events partway
        # through rather than thinning out uniformly. This is what makes recency
        # a genuinely predictive feature instead of noise.
        churn_date: dt.date | None = None
        churn_probability = 1.0 - float(
            np.exp(-behaviour.churn_hazard * config.simulation_days)
        )
        if rng.random() < churn_probability:
            churn_offset = int(rng.integers(1, config.simulation_days))
            churn_date = start + dt.timedelta(days=churn_offset)

        department_weights = _sample_department_weights(rng, config)
        subcategory_weights = _sample_subcategory_weights(rng, config, department_weights)
        brand_affinity = _sample_brand_affinity(rng, config, department_weights, brand_ids)

        # Price sensitivity is segment-anchored with individual variation.
        price_sensitivity = float(
            np.clip(rng.normal(behaviour.price_sensitivity, 0.14), 0.02, 0.99)
        )
        # A price-sensitive user targets the cheap end of each subcategory.
        price_target = float(np.clip(rng.beta(2.0, 2.0) * (1.0 - price_sensitivity) + 0.08, 0.03, 0.97))

        primary_device = DEVICES[int(rng.choice(len(DEVICES), p=DEVICE_WEIGHTS))]

        top_subs = sorted(subcategory_weights, key=subcategory_weights.get, reverse=True)[:3]
        onboarding = [leaf_id_by_sub[s] for s in top_subs if s in leaf_id_by_sub]

        first = FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]
        last = LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]

        rows.append(
            {
                "id": user_id,
                "email": f"{first.lower()}.{last.lower()}{user_id}@example.com",
                "full_name": f"{first} {last}",
                "role": "customer",
                "country": COUNTRIES[int(rng.choice(len(COUNTRIES), p=COUNTRY_WEIGHTS))],
                "signup_source": SIGNUP_SOURCES[
                    int(rng.choice(len(SIGNUP_SOURCES), p=SIGNUP_WEIGHTS))
                ],
                "primary_device": primary_device,
                "segment": segment,
                # Declared preference at signup. Deliberately only *correlated*
                # with realised behaviour, not equal to it - stated and revealed
                # preference differ, and a system that assumes otherwise
                # overfits onboarding answers.
                "preferred_price_band": _band_for_percentile(price_target),
                "onboarding_categories": onboarding,
                "created_at": signup_date,
            }
        )

        profiles[user_id] = UserProfile(
            user_id=user_id,
            segment=segment,
            signup_date=signup_date,
            churn_date=churn_date,
            department_weights=department_weights,
            subcategory_weights=subcategory_weights,
            brand_affinity=brand_affinity,
            price_target_percentile=price_target,
            price_sensitivity=price_sensitivity,
            exploration=float(np.clip(rng.normal(behaviour.exploration, 0.06), 0.02, 0.8)),
            purchase_propensity=float(
                np.clip(rng.normal(behaviour.purchase_propensity, 0.10), 0.005, 1.6)
            ),
            sessions_per_week=float(
                max(rng.gamma(shape=3.0, scale=behaviour.sessions_per_week / 3.0), 0.02)
            ),
            basket_size=behaviour.basket_size,
            primary_device=primary_device,
        )

    return pd.DataFrame(rows), profiles


def _band_for_percentile(percentile: float) -> str:
    if percentile < 0.28:
        return "budget"
    if percentile < 0.62:
        return "mid"
    if percentile < 0.87:
        return "premium"
    return "luxury"


def profiles_to_frame(profiles: dict[int, UserProfile]) -> pd.DataFrame:
    """Serialise latent profiles for the diagnostic report.

    Written to `latent_user_profiles.parquet`, which is explicitly *not* loaded
    into the database. It exists only so the Phase 2 gate can ask "does the
    behaviour we generated actually reveal the taste we injected?".
    """
    return pd.DataFrame(
        [
            {
                "user_id": p.user_id,
                "segment": p.segment,
                "signup_date": p.signup_date,
                "churn_date": p.churn_date,
                "top_department": max(p.department_weights, key=p.department_weights.get),
                "top_subcategory": max(p.subcategory_weights, key=p.subcategory_weights.get),
                "n_departments": len(p.department_weights),
                "brand_affinity": p.brand_affinity,
                "price_target_percentile": p.price_target_percentile,
                "price_sensitivity": p.price_sensitivity,
                "exploration": p.exploration,
                "purchase_propensity": p.purchase_propensity,
                "sessions_per_week": p.sessions_per_week,
            }
            for p in profiles.values()
        ]
    )


__all__ = ["UserProfile", "generate_users", "profiles_to_frame"]
