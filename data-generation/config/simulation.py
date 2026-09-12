"""Simulation parameters.

Every number that shapes the generated dataset lives here, so a run is fully
described by this object plus a seed. That is what makes the dataset
reproducible (NFR-05): the same config and seed must produce a byte-identical
catalogue and event log.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class UserMixConfig:
    """Population mix.

    The proportions are chosen to look like a real store rather than a uniform
    sample: most registered users barely engage, a small minority drive most of
    the revenue, and a meaningful slice browses without ever buying. If this
    mix were uniform, every user-level feature would be near-constant and the
    ranking model would have nothing to separate.
    """

    high_value: float = 0.06
    regular: float = 0.19
    casual: float = 0.32
    bargain_hunter: float = 0.12
    window_shopper: float = 0.16
    inactive: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "high_value": self.high_value,
            "regular": self.regular,
            "casual": self.casual,
            "bargain_hunter": self.bargain_hunter,
            "window_shopper": self.window_shopper,
            "inactive": self.inactive,
        }


@dataclass(frozen=True, slots=True)
class SegmentBehaviour:
    """Per-segment behavioural parameters.

    `sessions_per_week` is the Poisson rate; `purchase_propensity` scales the
    view-to-purchase probability; `basket_size` is the mean number of distinct
    products per order; `exploration` is the probability of stepping outside
    the user's preferred categories, which is what stops affinities from being
    perfectly separable and keeps the problem non-trivial.
    """

    sessions_per_week: float
    purchase_propensity: float
    basket_size: float
    exploration: float
    price_sensitivity: float
    churn_hazard: float


#: Session rates are per week and deliberately low. A store with 10k registered
#: users does not see most of them 30 times in six months; inflating the rate
#: would produce a dense interaction matrix that makes collaborative filtering
#: look far easier than it is, and would quietly delete the sparsity problem
#: this whole architecture exists to handle.
#:
#: `churn_hazard` is a per-day hazard: the probability of lapsing at all over
#: the window is 1 - exp(-hazard * days), so over 180 days these values give
#: roughly 9 % churn for high-value users and 76 % for the inactive segment.
SEGMENT_BEHAVIOUR: dict[str, SegmentBehaviour] = {
    "high_value": SegmentBehaviour(0.90, 1.00, 2.6, 0.30, 0.15, 0.0005),
    "regular": SegmentBehaviour(0.40, 0.62, 1.8, 0.24, 0.35, 0.0010),
    "casual": SegmentBehaviour(0.16, 0.34, 1.3, 0.20, 0.50, 0.0020),
    "bargain_hunter": SegmentBehaviour(0.38, 0.45, 1.6, 0.34, 0.88, 0.0015),
    "window_shopper": SegmentBehaviour(0.45, 0.05, 1.1, 0.42, 0.62, 0.0015),
    "inactive": SegmentBehaviour(0.03, 0.18, 1.2, 0.25, 0.55, 0.0080),
}


@dataclass(frozen=True, slots=True)
class FunnelConfig:
    """Base transition probabilities through the browsing funnel.

    These are the *base rates*, before per-user and per-product modulation. The
    realised rates in the generated data will differ, and deliberately so: a
    user browsing a product they have strong affinity for converts far better
    than these numbers, which is precisely the signal the models must find.
    """

    view_to_click: float = 0.34
    click_to_cart: float = 0.30
    cart_to_purchase: float = 0.64
    cart_to_removal: float = 0.17
    view_to_wishlist: float = 0.030
    view_to_share: float = 0.008
    purchase_to_rating: float = 0.28
    rating_to_review: float = 0.35

    #: How strongly an affinity match multiplies conversion. 3.0 means a
    #: perfectly-matched product is three times as likely to convert as a
    #: mismatched one, before price and quality effects.
    affinity_conversion_boost: float = 3.0
    #: How strongly a price mismatch suppresses conversion for a
    #: price-sensitive user.
    price_penalty_strength: float = 2.2
    #: How strongly product quality (rating) lifts conversion.
    quality_boost: float = 0.9


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Within-session browsing dynamics."""

    #: Geometric continuation probability: after each product view, the
    #: session continues with this probability. 0.72 gives a mean of ~3.6
    #: product views per session with a realistic long tail.
    continue_probability: float = 0.72
    max_products_per_session: int = 25

    #: Probability the next product viewed is drawn from the neighbourhood of
    #: the current one (same subcategory, brand or price band) rather than
    #: resampled from the user's global preference. This is what creates the
    #: co-visitation structure that "customers who viewed this also viewed"
    #: (FR-04) has to recover.
    neighbour_continuation: float = 0.62

    mean_dwell_seconds: float = 42.0
    search_probability: float = 0.28

    #: Weekly seasonality: multiplier on session arrival rate by weekday
    #: (Monday = 0). Weekend browsing is higher, midweek purchasing is higher.
    weekday_multiplier: tuple[float, ...] = (
        0.92, 0.95, 1.00, 1.04, 1.12, 1.34, 1.28,
    )
    #: Hour-of-day arrival weights (local). Bimodal: lunchtime and evening.
    hour_weights: tuple[float, ...] = (
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0,
        1.4, 1.6, 1.7, 1.9, 2.2, 2.0, 1.8, 1.7,
        1.8, 2.1, 2.6, 3.0, 3.2, 2.8, 1.9, 0.9,
    )


@dataclass(frozen=True, slots=True)
class PreferenceConfig:
    """Shape of the latent preference vectors."""

    #: Dirichlet concentration over departments. Low values give users who care
    #: about one or two departments; high values give diffuse, unlearnable
    #: taste. 0.35 produces the concentration real e-commerce shows.
    department_concentration: float = 0.35
    #: Number of departments a user can meaningfully shop in.
    max_departments: int = 3
    #: Dirichlet concentration within a chosen department's subcategories.
    subcategory_concentration: float = 0.55
    #: Number of brands a user develops an affinity for.
    brands_min: int = 1
    brands_max: int = 4
    #: How strongly brand affinity multiplies selection probability.
    brand_affinity_strength: float = 2.4


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete description of one dataset."""

    seed: int = 42
    n_users: int = 10_000
    n_products: int = 5_000
    simulation_days: int = 180
    #: Simulation end date. Defaults to "today" at the call site; pinned
    #: explicitly for reproducible runs.
    end_date: dt.date = dt.date(2026, 9, 1)

    #: Hard floor from the Phase 2 exit criterion. The simulator does not
    #: target an event count directly - events emerge from the behavioural
    #: model - but the run fails loudly if the emergent total falls short,
    #: because silently producing a thin dataset would invalidate every later
    #: phase.
    min_events: int = 100_000

    variants_per_product_min: int = 1
    variants_per_product_max: int = 6

    #: Fraction of the catalogue released *during* the simulation window rather
    #: than before it. These products are the new-product cold-start test set
    #: (FR-10): they genuinely have little or no early interaction history.
    late_release_fraction: float = 0.12
    #: Fraction of users who register during the window. Same idea for FR-09.
    late_signup_fraction: float = 0.18

    users: UserMixConfig = field(default_factory=UserMixConfig)
    funnel: FunnelConfig = field(default_factory=FunnelConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)

    @property
    def start_date(self) -> dt.date:
        return self.end_date - dt.timedelta(days=self.simulation_days)

    def describe(self) -> dict[str, object]:
        """Flat summary, written alongside the dataset for provenance."""
        return {
            "seed": self.seed,
            "n_users": self.n_users,
            "n_products": self.n_products,
            "simulation_days": self.simulation_days,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "user_mix": self.users.as_dict(),
            "late_release_fraction": self.late_release_fraction,
            "late_signup_fraction": self.late_signup_fraction,
        }


__all__ = [
    "SEGMENT_BEHAVIOUR",
    "FunnelConfig",
    "PreferenceConfig",
    "SegmentBehaviour",
    "SessionConfig",
    "SimulationConfig",
    "UserMixConfig",
]
