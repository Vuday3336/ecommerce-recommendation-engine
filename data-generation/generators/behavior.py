"""Behavioural simulator: sessions, events, orders.

The generative story, stated plainly so the resulting data can be defended:

* A user's taste is a distribution over subcategories plus a set of brand
  affinities plus a target price percentile. Products are selected by sampling
  a subcategory from that distribution, then a product within it weighted by
  brand affinity, price fit, quality and a global popularity prior.
* Browsing is a Markov chain, not a sequence of independent draws. After each
  view the user either continues to a *neighbouring* product (same subcategory,
  same brand, or a complementary subcategory) or resamples from their global
  taste. The neighbour step is what creates co-visitation structure for
  "customers who viewed this also viewed" to recover.
* Conversion is modulated, not constant. Affinity match, price fit, product
  quality and department seasonality all move the funnel probabilities. A
  well-matched product converts several times better than a mismatched one,
  which is the entire signal the ranking model has to learn.
* Purchases pull complements into the same order using the declared
  complementarity graph, which is the ground truth for frequently-bought-together.
* Consumables are re-purchased on an interval, which is what makes repeat
  purchase and recency genuinely predictive features.
* Users churn on a per-day hazard, so activity thins out at different times
  for different people instead of stopping uniformly at the window edge.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from config.simulation import SimulationConfig
from config.taxonomy import COMPLEMENTARY_PAIRS

from generators.catalog import Catalogue
from generators.users import UserProfile

UTC = dt.UTC

#: Chance per later session that a wishlisted item is revisited. Wishlisting is
#: deferred intent, not a dead end: roughly a fifth of wishlist entries convert
#: in real storefronts, and modelling that is what gives WISHLIST its proper
#: place in the implicit-feedback ordering (ADR-002).
WISHLIST_REVISIT_PROBABILITY = 0.30

#: Multiplier on the affinity match for a product the user deliberately came
#: back to, whether from a wishlist or a repeat-purchase cycle.
HIGH_INTENT_BOOST = 2.2


@dataclass(slots=True)
class SimulationResult:
    sessions: pd.DataFrame
    events: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame

    def summary(self) -> dict[str, int]:
        return {
            "sessions": len(self.sessions),
            "events": len(self.events),
            "orders": len(self.orders),
            "order_items": len(self.order_items),
        }


@dataclass(slots=True)
class _ProductArrays:
    """Column-major view of the catalogue, indexed by `product_id - 1`."""

    price: np.ndarray
    price_percentile: np.ndarray
    brand_id: np.ndarray
    quality: np.ndarray
    popularity_prior: np.ndarray
    released_ordinal: np.ndarray
    subcategory: np.ndarray
    season_phase: np.ndarray
    season_amplitude: np.ndarray
    repeat_rate: np.ndarray
    repeat_interval: np.ndarray


class BehaviourSimulator:
    """Generates the event log for a population against a catalogue."""

    def __init__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
        catalogue: Catalogue,
        profiles: dict[int, UserProfile],
    ) -> None:
        self.config = config
        self.rng = rng
        self.catalogue = catalogue
        self.profiles = profiles

        self._arrays = self._build_arrays(catalogue, rng)
        self._variants_by_product = self._build_variant_index(catalogue)
        self._complements = self._build_complement_index()
        self._subcategory_names = list(catalogue.products_by_subcategory)
        self._global_subcategory_p = self._build_global_subcategory_prior()

        self._events: list[dict] = []
        self._sessions: list[dict] = []
        self._orders: list[dict] = []
        self._order_items: list[dict] = []
        self._next_order_id = 1
        self._next_order_item_id = 1
        self._next_session_id = 1

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_arrays(catalogue: Catalogue, rng: np.random.Generator) -> _ProductArrays:
        products = catalogue.products.sort_values("id")
        n = len(products)

        # A latent global popularity prior with a heavy tail. Real catalogues
        # are extremely unequal, and building that in on purpose is what gives
        # the popularity-bias problem (R-06) something real to bite on: without
        # it, coverage and Gini metrics would look artificially healthy.
        popularity = rng.pareto(a=1.6, size=n) + 1.0
        popularity /= popularity.mean()

        return _ProductArrays(
            price=products["price"].to_numpy(dtype=float),
            price_percentile=products["price_percentile"].to_numpy(dtype=float),
            brand_id=products["brand_id"].to_numpy(dtype=np.int64),
            quality=products["quality"].to_numpy(dtype=float),
            popularity_prior=popularity,
            released_ordinal=np.array(
                [d.toordinal() for d in products["released_at"]], dtype=np.int64
            ),
            subcategory=products["subcategory"].to_numpy(dtype=object),
            season_phase=products["season_phase"].to_numpy(dtype=float),
            season_amplitude=products["season_amplitude"].to_numpy(dtype=float),
            repeat_rate=products["repeat_rate"].to_numpy(dtype=float),
            repeat_interval=products["repeat_interval_days"].to_numpy(dtype=float),
        )

    @staticmethod
    def _build_variant_index(catalogue: Catalogue) -> dict[int, np.ndarray]:
        return {
            int(pid): group["id"].to_numpy()
            for pid, group in catalogue.variants.groupby("product_id", sort=False)
        }

    @staticmethod
    def _build_complement_index() -> dict[str, list[tuple[str, float]]]:
        index: dict[str, list[tuple[str, float]]] = {}
        for left, right, probability in COMPLEMENTARY_PAIRS:
            index.setdefault(left, []).append((right, probability))
        return index

    def _build_global_subcategory_prior(self) -> np.ndarray:
        """Catalogue-size-weighted prior, used for exploration draws."""
        sizes = np.array(
            [len(self.catalogue.products_by_subcategory[s]) for s in self._subcategory_names],
            dtype=float,
        )
        return sizes / sizes.sum()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _subcategory_weights(self, profile: UserProfile, subcategory: str) -> np.ndarray:
        """Cached selection weights over the products of one subcategory."""
        cached = profile._weight_cache.get(subcategory)
        if cached is not None:
            return cached

        ids = self.catalogue.products_by_subcategory[subcategory]
        idx = ids - 1
        arrays = self._arrays

        brand_boost = np.array(
            [profile.brand_affinity.get(int(b), 1.0) for b in arrays.brand_id[idx]],
            dtype=float,
        )
        price_gap = np.abs(arrays.price_percentile[idx] - profile.price_target_percentile)
        price_fit = np.exp(
            -self.config.funnel.price_penalty_strength * profile.price_sensitivity * price_gap
        )
        quality_factor = 0.6 + arrays.quality[idx]

        weights = arrays.popularity_prior[idx] * brand_boost * price_fit * quality_factor
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(len(ids), 1.0 / len(ids))

        profile._weight_cache[subcategory] = weights
        return weights

    def _pick_subcategory(self, profile: UserProfile) -> str:
        if self.rng.random() < profile.exploration or not profile.subcategory_weights:
            index = int(self.rng.choice(len(self._subcategory_names), p=self._global_subcategory_p))
            return self._subcategory_names[index]

        names = list(profile.subcategory_weights)
        probs = np.array([profile.subcategory_weights[n] for n in names], dtype=float)
        probs /= probs.sum()
        return names[int(self.rng.choice(len(names), p=probs))]

    def _pick_product(self, profile: UserProfile, subcategory: str, day: int) -> int | None:
        """Sample a released product from a subcategory."""
        ids = self.catalogue.products_by_subcategory.get(subcategory)
        if ids is None or len(ids) == 0:
            return None

        weights = self._subcategory_weights(profile, subcategory)
        # A product cannot be viewed before it is released. This is what makes
        # new-product cold start a real condition in the data rather than an
        # assumption asserted in a document.
        available = self._arrays.released_ordinal[ids - 1] <= day
        if not available.any():
            return None

        masked = np.where(available, weights, 0.0)
        total = masked.sum()
        if total <= 0:
            return None
        return int(self.rng.choice(ids, p=masked / total))

    def _pick_neighbour(self, profile: UserProfile, product_id: int, day: int) -> int | None:
        """Step to a product related to the one just viewed."""
        arrays = self._arrays
        idx = product_id - 1
        subcategory = str(arrays.subcategory[idx])

        roll = self.rng.random()
        if roll < 0.20:
            complements = self._complements.get(subcategory)
            if complements:
                target = complements[int(self.rng.integers(len(complements)))][0]
                return self._pick_product(profile, target, day)
        if roll < 0.38:
            brand_products = self.catalogue.products_by_brand.get(int(arrays.brand_id[idx]))
            if brand_products is not None and len(brand_products) > 1:
                available = brand_products[arrays.released_ordinal[brand_products - 1] <= day]
                if len(available):
                    return int(self.rng.choice(available))
        return self._pick_product(profile, subcategory, day)

    # ------------------------------------------------------------------
    # Modulation
    # ------------------------------------------------------------------

    def _affinity(self, profile: UserProfile, subcategory: str) -> float:
        """0..1 measure of how well a subcategory matches the user's taste."""
        weight = profile.subcategory_weights.get(subcategory, 0.0)
        if not profile.subcategory_weights:
            return 0.0
        peak = max(profile.subcategory_weights.values())
        return float(min(weight / peak, 1.0)) if peak > 0 else 0.0

    def _seasonality(self, product_index: int, day_of_year: int) -> float:
        arrays = self._arrays
        phase = arrays.season_phase[product_index]
        amplitude = arrays.season_amplitude[product_index]
        return float(
            1.0 + amplitude * math.cos(2.0 * math.pi * (day_of_year / 365.25 - phase))
        )

    def _price_fit(self, profile: UserProfile, product_index: int) -> float:
        gap = abs(
            self._arrays.price_percentile[product_index] - profile.price_target_percentile
        )
        return float(
            math.exp(
                -self.config.funnel.price_penalty_strength * profile.price_sensitivity * gap
            )
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        start = self.config.start_date
        end = self.config.end_date
        session_config = self.config.session

        weekday_multiplier = np.array(session_config.weekday_multiplier, dtype=float)
        hour_p = np.array(session_config.hour_weights, dtype=float)
        hour_p /= hour_p.sum()

        for profile in self.profiles.values():
            active_from = max(profile.signup_date, start)
            active_to = min(profile.churn_date or end, end)
            active_days = (active_to - active_from).days
            if active_days <= 0:
                continue

            expected = profile.sessions_per_week * (active_days / 7.0)
            n_sessions = int(self.rng.poisson(expected))
            if n_sessions == 0:
                continue

            # Session days are sampled with weekday seasonality rather than
            # uniformly, so day-of-week effects exist for the trending model
            # and the drift monitor to see.
            offsets = self.rng.integers(0, active_days, size=n_sessions)
            days = [active_from + dt.timedelta(days=int(o)) for o in offsets]
            keep = self.rng.random(n_sessions) < (
                weekday_multiplier[[d.weekday() for d in days]] / weekday_multiplier.max()
            )
            session_days = sorted(d for d, k in zip(days, keep, strict=True) if k)

            purchased_at: dict[int, dt.date] = {}
            wishlist: dict[int, dt.date] = {}
            for session_day in session_days:
                hour = int(self.rng.choice(24, p=hour_p))
                minute = int(self.rng.integers(0, 60))
                started = dt.datetime(
                    session_day.year, session_day.month, session_day.day,
                    hour, minute, tzinfo=UTC,
                )
                self._simulate_session(profile, started, purchased_at, wishlist)

        return SimulationResult(
            sessions=pd.DataFrame(self._sessions),
            events=pd.DataFrame(self._events),
            orders=pd.DataFrame(self._orders),
            order_items=pd.DataFrame(self._order_items),
        )

    def _emit(
        self,
        *,
        occurred_at: dt.datetime,
        event_type: str,
        user_id: int,
        session_key: str,
        product_id: int | None,
        source: str,
        device_type: str,
        metadata: dict | None = None,
    ) -> None:
        self._events.append(
            {
                "occurred_at": occurred_at,
                "event_type": event_type,
                "user_id": user_id,
                "session_key": session_key,
                "product_id": product_id,
                "source": source,
                "device_type": device_type,
                "recommendation_id": None,
                "event_metadata": metadata or {},
            }
        )

    def _simulate_session(
        self,
        profile: UserProfile,
        started: dt.datetime,
        purchased_at: dict[int, dt.date],
        wishlist: dict[int, dt.date],
    ) -> None:
        config = self.config
        funnel = config.funnel
        session_config = config.session
        arrays = self._arrays

        # Derived from the seeded generator, not `uuid.uuid4()`. A random UUID
        # would make every run produce different session keys, and since the
        # key appears in the event log the dataset would stop being
        # reproducible from its seed (NFR-05).
        session_key = uuid.UUID(bytes=self.rng.bytes(16), version=4).hex
        session_id = self._next_session_id
        self._next_session_id += 1

        device = (
            profile.primary_device
            if self.rng.random() < 0.82
            else ("mobile" if profile.primary_device != "mobile" else "desktop")
        )
        day_ordinal = started.date().toordinal()
        day_of_year = started.timetuple().tm_yday
        clock = started
        event_start = len(self._events)

        self._emit(
            occurred_at=clock,
            event_type="SESSION_START",
            user_id=profile.user_id,
            session_key=session_key,
            product_id=None,
            source="app",
            device_type=device,
            metadata={"entry": "direct" if self.rng.random() < 0.6 else "search_engine"},
        )

        source_surface = "homepage"
        if self.rng.random() < session_config.search_probability:
            clock += dt.timedelta(seconds=float(self.rng.exponential(12.0)))
            subcategory = self._pick_subcategory(profile)
            self._emit(
                occurred_at=clock,
                event_type="SEARCH",
                user_id=profile.user_id,
                session_key=session_key,
                product_id=None,
                source="search",
                device_type=device,
                metadata={"query": subcategory.lower(), "results": int(self.rng.integers(4, 60))},
            )
            source_surface = "search"

        # --- deliberate revisits ---------------------------------------------
        # Two mechanisms bring a user back to a specific product rather than to
        # a fresh browse, and both matter downstream.
        #
        # Repeat purchase: a consumable whose interval has elapsed. This is what
        # makes repeat rate and recency genuinely predictive features.
        #
        # Deferred wishlist intent: a wishlisted item the user comes back to
        # buy. Without this, wishlisting would be a dead end in the data and the
        # calibrated implicit weight for WISHLIST would land *below* a plain
        # view (ADR-002) - which contradicts both the brief's stated ordering
        # and how wishlists actually behave.
        due = [
            pid
            for pid, when in purchased_at.items()
            if (started.date() - when).days >= arrays.repeat_interval[pid - 1]
            and self.rng.random() < arrays.repeat_rate[pid - 1]
        ]
        revisited_wishlist = [
            pid
            for pid, when in wishlist.items()
            if (started.date() - when).days >= 1
            and self.rng.random() < WISHLIST_REVISIT_PROBABILITY
        ]

        # Items the user already signalled intent for convert far better than a
        # cold browse, so they carry a boost through the funnel.
        high_intent: set[int] = set(due) | set(revisited_wishlist)
        revisit_queue = due + revisited_wishlist
        current: int | None = None
        cart: list[int] = []
        purchases: list[int] = []
        views = 0

        while views < session_config.max_products_per_session:
            if current is None:
                # Work through deliberate revisits first, then browse. Taking
                # only the head of the queue would mean a user returning for
                # two wishlisted items silently forgets one of them.
                if revisit_queue:
                    current = revisit_queue.pop(0)
                else:
                    subcategory = self._pick_subcategory(profile)
                    current = self._pick_product(profile, subcategory, day_ordinal)
            if current is None:
                break

            index = current - 1
            subcategory = str(arrays.subcategory[index])
            affinity = self._affinity(profile, subcategory)
            price_fit = self._price_fit(profile, index)
            season = self._seasonality(index, day_of_year)
            quality = float(arrays.quality[index])

            clock += dt.timedelta(
                seconds=float(self.rng.exponential(session_config.mean_dwell_seconds))
            )
            views += 1
            self._emit(
                occurred_at=clock,
                event_type="PRODUCT_VIEW",
                user_id=profile.user_id,
                session_key=session_key,
                product_id=current,
                source=source_surface,
                device_type=device,
                metadata={"dwell_ms": int(self.rng.exponential(24_000)) + 800},
            )
            source_surface = "pdp"

            match = 1.0 + funnel.affinity_conversion_boost * affinity
            if current in high_intent:
                match *= HIGH_INTENT_BOOST
            quality_lift = 1.0 + funnel.quality_boost * (quality - 0.5)

            p_click = min(funnel.view_to_click * match * 0.6, 0.95)
            if self.rng.random() < p_click:
                clock += dt.timedelta(seconds=float(self.rng.exponential(8.0)))
                self._emit(
                    occurred_at=clock,
                    event_type="PRODUCT_CLICK",
                    user_id=profile.user_id,
                    session_key=session_key,
                    product_id=current,
                    source="pdp",
                    device_type=device,
                )

                p_cart = min(
                    funnel.click_to_cart
                    * match
                    * price_fit
                    * quality_lift
                    * season
                    * profile.purchase_propensity,
                    0.92,
                )
                if self.rng.random() < p_cart:
                    clock += dt.timedelta(seconds=float(self.rng.exponential(15.0)))
                    cart.append(current)
                    self._emit(
                        occurred_at=clock,
                        event_type="ADD_TO_CART",
                        user_id=profile.user_id,
                        session_key=session_key,
                        product_id=current,
                        source="pdp",
                        device_type=device,
                        metadata={"quantity": 1},
                    )
            elif self.rng.random() < funnel.view_to_wishlist * match:
                clock += dt.timedelta(seconds=float(self.rng.exponential(6.0)))
                wishlist.setdefault(current, clock.date())
                self._emit(
                    occurred_at=clock,
                    event_type="WISHLIST",
                    user_id=profile.user_id,
                    session_key=session_key,
                    product_id=current,
                    source="pdp",
                    device_type=device,
                )
            elif self.rng.random() < funnel.view_to_share:
                self._emit(
                    occurred_at=clock,
                    event_type="PRODUCT_SHARE",
                    user_id=profile.user_id,
                    session_key=session_key,
                    product_id=current,
                    source="pdp",
                    device_type=device,
                    metadata={"channel": "link"},
                )

            if not revisit_queue and self.rng.random() >= session_config.continue_probability:
                break
            current = (
                self._pick_neighbour(profile, current, day_ordinal)
                if not revisit_queue
                and self.rng.random() < session_config.neighbour_continuation
                else None
            )

        # --- checkout --------------------------------------------------------
        for product_id in list(cart):
            if self.rng.random() < self.config.funnel.cart_to_removal:
                clock += dt.timedelta(seconds=float(self.rng.exponential(20.0)))
                self._emit(
                    occurred_at=clock,
                    event_type="REMOVE_FROM_CART",
                    user_id=profile.user_id,
                    session_key=session_key,
                    product_id=product_id,
                    source="cart",
                    device_type=device,
                )
                continue
            # Purchase propensity is applied once, at the add-to-cart step.
            # Applying it again here would square it, so a casual user with
            # propensity 0.34 would convert at 0.12 of the base rate and the
            # segment differences would stop being interpretable.
            if self.rng.random() < self.config.funnel.cart_to_purchase:
                purchases.append(product_id)

        purchases = self._pull_complements(profile, purchases, day_ordinal)

        if purchases:
            clock += dt.timedelta(seconds=float(self.rng.exponential(45.0)))
            self._record_order(profile, session_id, session_key, device, purchases, clock)
            for product_id in dict.fromkeys(purchases):
                self._emit(
                    occurred_at=clock,
                    event_type="PURCHASE",
                    user_id=profile.user_id,
                    session_key=session_key,
                    product_id=product_id,
                    source="checkout",
                    device_type=device,
                    metadata={"quantity": 1},
                )
                purchased_at[product_id] = clock.date()
                wishlist.pop(product_id, None)
                self._maybe_rate(profile, product_id, clock, session_key, device)

        clock += dt.timedelta(seconds=float(self.rng.exponential(30.0)))
        self._emit(
            occurred_at=clock,
            event_type="SESSION_END",
            user_id=profile.user_id,
            session_key=session_key,
            product_id=None,
            source="app",
            device_type=device,
        )

        self._sessions.append(
            {
                "id": session_id,
                "session_key": session_key,
                "user_id": profile.user_id,
                "device_type": device,
                "started_at": started,
                "ended_at": clock,
                "event_count": len(self._events) - event_start,
                "converted": bool(purchases),
            }
        )

    def _pull_complements(
        self, profile: UserProfile, purchases: list[int], day: int
    ) -> list[int]:
        """Add complementary products to a basket, per the complementarity graph."""
        if not purchases:
            return purchases
        arrays = self._arrays
        extended = list(purchases)
        for product_id in purchases:
            subcategory = str(arrays.subcategory[product_id - 1])
            for target, probability in self._complements.get(subcategory, ()):
                if self.rng.random() >= probability:
                    continue
                companion = self._pick_product(profile, target, day)
                if companion is not None and companion not in extended:
                    extended.append(companion)
        return extended

    def _maybe_rate(
        self,
        profile: UserProfile,
        product_id: int,
        purchased_at: dt.datetime,
        session_key: str,
        device: str,
    ) -> None:
        """Emit a delayed rating (and sometimes a review).

        Ratings arrive days after the purchase, which is both realistic and
        important: a rating that lands at purchase time would leak the outcome
        into any feature window that includes the purchase.
        """
        funnel = self.config.funnel
        if self.rng.random() >= funnel.purchase_to_rating:
            return

        index = product_id - 1
        subcategory = str(self._arrays.subcategory[index])
        affinity = self._affinity(profile, subcategory)
        quality = float(self._arrays.quality[index])

        centre = 2.4 + 1.7 * quality + 0.9 * affinity
        value = int(np.clip(round(self.rng.normal(centre, 0.7)), 1, 5))

        rated_at = purchased_at + dt.timedelta(days=float(self.rng.uniform(1.0, 21.0)))
        if rated_at.date() > self.config.end_date:
            return

        self._emit(
            occurred_at=rated_at,
            event_type="PRODUCT_RATING",
            user_id=profile.user_id,
            session_key=session_key,
            product_id=product_id,
            source="email",
            device_type=device,
            metadata={"rating": value},
        )
        if self.rng.random() < funnel.rating_to_review:
            self._emit(
                occurred_at=rated_at + dt.timedelta(minutes=float(self.rng.uniform(1, 30))),
                event_type="PRODUCT_REVIEW",
                user_id=profile.user_id,
                session_key=session_key,
                product_id=product_id,
                source="email",
                device_type=device,
                metadata={"rating": value, "length": int(self.rng.integers(40, 600))},
            )

    def _record_order(
        self,
        profile: UserProfile,
        session_id: int,
        session_key: str,
        device: str,
        purchases: list[int],
        placed_at: dt.datetime,
    ) -> None:
        order_id = self._next_order_id
        self._next_order_id += 1

        # Collapse repeats into one line with a higher quantity. A user can
        # cart the same product twice in a session, but a real order has one
        # line per variant with a quantity - which is exactly what the
        # `uq_order_items_order_id_variant_id` constraint encodes. Emitting two
        # lines instead is rejected by the database, and rightly so.
        from collections import Counter

        basket = Counter(purchases)

        subtotal = 0.0
        for product_id, repeats in basket.items():
            price = float(self._arrays.price[product_id - 1])
            quantity = repeats + int(self.rng.random() < 0.14)
            discount = round(price * quantity * float(self.rng.choice([0.0, 0.0, 0.0, 0.1, 0.2])), 2)
            line_total = round(price * quantity - discount, 2)
            subtotal += line_total

            variants = self._variants_by_product.get(product_id)
            variant_id = int(self.rng.choice(variants)) if variants is not None and len(variants) else None

            self._order_items.append(
                {
                    "id": self._next_order_item_id,
                    "order_id": order_id,
                    "product_id": product_id,
                    "variant_id": variant_id,
                    "quantity": quantity,
                    "unit_price": price,
                    "discount": discount,
                    "line_total": line_total,
                    "source_recommendation_id": None,
                }
            )
            self._next_order_item_id += 1

        shipping = 0.0 if subtotal > 60 else 5.95
        self._orders.append(
            {
                "id": order_id,
                "user_id": profile.user_id,
                "session_id": session_id,
                "order_number": f"ORD-{order_id:08d}",
                "status": "delivered" if self.rng.random() < 0.93 else "returned",
                "subtotal": round(subtotal, 2),
                "discount_total": 0.0,
                "shipping_total": shipping,
                "grand_total": round(subtotal + shipping, 2),
                "currency": "USD",
                "placed_at": placed_at,
            }
        )


def simulate(
    config: SimulationConfig,
    rng: np.random.Generator,
    catalogue: Catalogue,
    profiles: dict[int, UserProfile],
) -> SimulationResult:
    return BehaviourSimulator(config, rng, catalogue, profiles).run()


__all__ = ["BehaviourSimulator", "SimulationResult", "simulate"]
