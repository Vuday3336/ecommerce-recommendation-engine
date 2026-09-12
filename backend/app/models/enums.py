"""Domain enumerations shared by ORM models, API schemas and the ML package.

These are plain `str` enums so they serialise directly to JSON and compare
cleanly against database values. Each is materialised as a native PostgreSQL
ENUM type, which costs 4 bytes per row instead of a variable-length string and
makes an invalid value a database error rather than a silent data-quality bug.

Adding a new event type (FR-21) means adding a member here plus an
`ALTER TYPE ... ADD VALUE` migration. No table is rewritten and no column
changes shape, which is the property that makes the event system extensible.
"""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    CUSTOMER = "customer"
    ANALYST = "analyst"
    ADMIN = "admin"


class UserSegment(StrEnum):
    """Behavioural segment assigned by the simulator and recomputed by the
    feature pipeline. Used for cold-start priors and dashboard slicing."""

    NEW = "new"
    CASUAL = "casual"
    REGULAR = "regular"
    HIGH_VALUE = "high_value"
    BARGAIN_HUNTER = "bargain_hunter"
    WINDOW_SHOPPER = "window_shopper"
    INACTIVE = "inactive"


class DeviceType(StrEnum):
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    """The 12 baseline behavioural events (FR-20)."""

    PRODUCT_VIEW = "PRODUCT_VIEW"
    PRODUCT_CLICK = "PRODUCT_CLICK"
    SEARCH = "SEARCH"
    ADD_TO_CART = "ADD_TO_CART"
    REMOVE_FROM_CART = "REMOVE_FROM_CART"
    WISHLIST = "WISHLIST"
    PURCHASE = "PURCHASE"
    PRODUCT_SHARE = "PRODUCT_SHARE"
    PRODUCT_RATING = "PRODUCT_RATING"
    PRODUCT_REVIEW = "PRODUCT_REVIEW"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"


#: Events that carry a product reference. `SEARCH`, `SESSION_START` and
#: `SESSION_END` do not, which is why `user_events.product_id` is nullable.
PRODUCT_SCOPED_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.PRODUCT_VIEW,
        EventType.PRODUCT_CLICK,
        EventType.ADD_TO_CART,
        EventType.REMOVE_FROM_CART,
        EventType.WISHLIST,
        EventType.PURCHASE,
        EventType.PRODUCT_SHARE,
        EventType.PRODUCT_RATING,
        EventType.PRODUCT_REVIEW,
    }
)


class OrderStatus(StrEnum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class RecommendationSurface(StrEnum):
    """Where a recommendation was shown. Attribution is meaningless without
    this: a click on the homepage rail and a click on the product page carry
    very different intent, and mixing them makes CTR uninterpretable."""

    HOME_FOR_YOU = "home_for_you"
    HOME_BECAUSE_VIEWED = "home_because_you_viewed"
    HOME_TRENDING = "home_trending"
    HOME_CONTINUE_SHOPPING = "home_continue_shopping"
    HOME_FREQUENTLY_BOUGHT = "home_frequently_bought_together"
    PDP_SIMILAR = "pdp_similar"
    PDP_FREQUENTLY_BOUGHT = "pdp_frequently_bought_together"
    PDP_ALSO_VIEWED = "pdp_also_viewed"
    SEARCH_RANKING = "search_ranking"
    RECENTLY_VIEWED = "recently_viewed"


class RecommendationSource(StrEnum):
    """Which candidate generator produced the item. Recorded per item so the
    dashboard can report recall and conversion contribution per source, which
    is how a source earns or loses its slot in the candidate mix."""

    POPULARITY = "popularity"
    TRENDING = "trending"
    CONTENT = "content"
    COLLABORATIVE = "collaborative"
    COVISITATION = "covisitation"
    FREQUENTLY_BOUGHT = "frequently_bought_together"
    CATEGORY_AFFINITY = "category_affinity"
    BRAND_AFFINITY = "brand_affinity"
    RECENTLY_VIEWED = "recently_viewed"
    EXPLORATION = "exploration"


class ServingStrategy(StrEnum):
    """Which rung of the degradation ladder served the request
    (architecture.md section 4.2). Logged on every response so `fallback rate`
    is a measurable health metric rather than an invisible failure."""

    RANKER = "ranker"
    HYBRID = "hybrid"
    COLLABORATIVE_CONTENT = "collaborative_content"
    CONTENT_TRENDING = "content_trending"
    CATEGORY_POPULAR = "category_popular"
    GLOBAL_TRENDING = "global_trending"
    STATIC_FALLBACK = "static_fallback"


class ModelStage(StrEnum):
    """Mirrors the MLflow registry stage (ADR-010)."""

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelKind(StrEnum):
    POPULARITY = "popularity"
    CONTENT = "content"
    COLLABORATIVE_ALS = "collaborative_als"
    COLLABORATIVE_BPR = "collaborative_bpr"
    HYBRID = "hybrid"
    RANKER = "ranker"
    EMBEDDING = "embedding"


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class PriceBand(StrEnum):
    """Coarse price tier. The simulator gives each user a preferred band and
    the feature pipeline derives price sensitivity from realised behaviour."""

    BUDGET = "budget"
    MID = "mid"
    PREMIUM = "premium"
    LUXURY = "luxury"


__all__ = [
    "PRODUCT_SCOPED_EVENTS",
    "DeviceType",
    "EventType",
    "ExperimentStatus",
    "ModelKind",
    "ModelStage",
    "OrderStatus",
    "PriceBand",
    "RecommendationSource",
    "RecommendationSurface",
    "ServingStrategy",
    "UserRole",
    "UserSegment",
]
