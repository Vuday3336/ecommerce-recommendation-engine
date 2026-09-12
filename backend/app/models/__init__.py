"""ORM model package.

Importing this module registers every table on `Base.metadata`. Alembic's
`env.py` and the DDL renderer both rely on that, so a new model file must be
imported here or it will silently never be migrated.
"""

from app.db.base import Base
from app.models.catalog import Brand, Category, Product, ProductVariant
from app.models.enums import (
    DeviceType,
    EventType,
    ExperimentStatus,
    ModelKind,
    ModelStage,
    OrderStatus,
    PriceBand,
    RecommendationSource,
    RecommendationSurface,
    ServingStrategy,
    UserRole,
    UserSegment,
)
from app.models.events import UserEvent, UserProductInteraction
from app.models.features import (
    EMBEDDING_DIM,
    ProductEmbedding,
    ProductFeature,
    UserFeature,
)
from app.models.mlops import Experiment, ExperimentAssignment, ModelVersion
from app.models.orders import Order, OrderItem
from app.models.recommendations import (
    Recommendation,
    RecommendationClick,
    RecommendationConversion,
    RecommendationImpression,
)
from app.models.users import User, UserSession

__all__ = [
    "EMBEDDING_DIM",
    "Base",
    "Brand",
    "Category",
    "DeviceType",
    "EventType",
    "Experiment",
    "ExperimentAssignment",
    "ExperimentStatus",
    "ModelKind",
    "ModelStage",
    "ModelVersion",
    "Order",
    "OrderItem",
    "OrderStatus",
    "PriceBand",
    "Product",
    "ProductEmbedding",
    "ProductFeature",
    "ProductVariant",
    "Recommendation",
    "RecommendationClick",
    "RecommendationConversion",
    "RecommendationImpression",
    "RecommendationSource",
    "RecommendationSurface",
    "ServingStrategy",
    "User",
    "UserEvent",
    "UserFeature",
    "UserProductInteraction",
    "UserRole",
    "UserSegment",
    "UserSession",
]
