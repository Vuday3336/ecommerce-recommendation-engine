"""Dataset loading.

The engine reads from an abstract `DataSource`, not from Postgres directly.
Two implementations exist: `ParquetSource` (the generated dataset, used for
training and offline evaluation) and `PostgresSource` (the live database, used
by the batch jobs that run against production data).

That split is not architectural decoration. It is what lets the entire ML stack
be trained, evaluated and regression-tested with no database at all, which
keeps CI fast and makes the ML package genuinely reusable from a notebook.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

#: Events that represent a user expressing interest in a product. `SEARCH`,
#: `SESSION_START` and `SESSION_END` carry no product and are excluded.
INTERACTION_EVENTS: tuple[str, ...] = (
    "PRODUCT_VIEW",
    "PRODUCT_CLICK",
    "ADD_TO_CART",
    "REMOVE_FROM_CART",
    "WISHLIST",
    "PURCHASE",
    "PRODUCT_SHARE",
    "PRODUCT_RATING",
    "PRODUCT_REVIEW",
)


@dataclass(slots=True)
class Dataset:
    """Everything the models train on.

    Frames are kept separate rather than pre-joined because different models
    need different joins, and materialising one wide table would waste memory
    on columns most models never touch.
    """

    events: pd.DataFrame
    products: pd.DataFrame
    categories: pd.DataFrame
    brands: pd.DataFrame
    users: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    sessions: pd.DataFrame

    def __post_init__(self) -> None:
        if "occurred_at" in self.events.columns:
            self.events = self.events.sort_values("occurred_at", kind="stable")

    @property
    def time_span(self) -> tuple[dt.datetime, dt.datetime]:
        return (
            self.events["occurred_at"].min().to_pydatetime(),
            self.events["occurred_at"].max().to_pydatetime(),
        )

    @property
    def interactions(self) -> pd.DataFrame:
        """Product-scoped events only, with catalogue metadata attached."""
        frame = self.events[
            self.events["event_type"].isin(INTERACTION_EVENTS)
            & self.events["product_id"].notna()
        ].copy()
        frame["product_id"] = frame["product_id"].astype("int64")
        frame["user_id"] = frame["user_id"].astype("int64")

        meta = self.products.set_index("id")[
            ["category_id", "brand_id", "price", "price_band"]
        ]
        return frame.join(meta, on="product_id")

    def summary(self) -> dict[str, int]:
        return {
            "events": len(self.events),
            "products": len(self.products),
            "users": len(self.users),
            "orders": len(self.orders),
            "order_items": len(self.order_items),
            "sessions": len(self.sessions),
        }


class DataSource(Protocol):
    def load(self) -> Dataset: ...


class ParquetSource:
    """Reads the generated dataset from Parquet."""

    FILES: ClassVar[dict[str, str]] = {
        "events": "events.parquet",
        "products": "products.parquet",
        "categories": "categories.parquet",
        "brands": "brands.parquet",
        "users": "users.parquet",
        "orders": "orders.parquet",
        "order_items": "order_items.parquet",
        "sessions": "user_sessions.parquet",
    }

    #: Columns stored as JSON text by the generator (see `generate.py`).
    JSON_COLUMNS: ClassVar[dict[str, tuple[str, ...]]] = {
        "products": ("attributes", "tags"),
        "users": ("onboarding_categories",),
        "events": ("event_metadata",),
    }

    def __init__(self, data_dir: Path, *, decode_json: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.decode_json = decode_json

    def load(self) -> Dataset:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"No dataset at {self.data_dir}. "
                "Run: python data-generation/generate.py"
            )

        frames: dict[str, pd.DataFrame] = {}
        for name, filename in self.FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            frame = pd.read_parquet(path)
            if self.decode_json:
                for column in self.JSON_COLUMNS.get(name, ()):
                    if column in frame.columns and frame[column].dtype == object:
                        first = frame[column].iloc[0] if len(frame) else None
                        if isinstance(first, str):
                            frame[column] = frame[column].map(json.loads)
            frames[name] = frame

        return Dataset(**frames)


class PostgresSource:
    """Reads the same shapes from the live database.

    Column names are aliased to match `ParquetSource` exactly, so every model
    downstream is indifferent to which source produced the data. Where the two
    disagree the fix belongs here, not in a model.
    """

    QUERIES: ClassVar[dict[str, str]] = {
        "events": """
            SELECT occurred_at, event_type::text AS event_type, user_id,
                   session_key, product_id, source, device_type::text AS device_type,
                   event_metadata
            FROM user_events
            WHERE (%(start)s IS NULL OR occurred_at >= %(start)s)
              AND (%(end)s IS NULL OR occurred_at < %(end)s)
        """,
        "products": """
            SELECT id, sku, name, description, category_id, brand_id,
                   price::float8 AS price, cost::float8 AS cost,
                   price_band::text AS price_band, attributes, tags,
                   stock_quantity, rating_average, rating_count,
                   view_count, purchase_count, released_at, is_active
            FROM products
        """,
        "categories": "SELECT id, name, slug, parent_id, depth, path FROM categories",
        "brands": "SELECT id, name, slug, price_tier::text AS price_tier FROM brands",
        "users": """
            SELECT id, email, full_name, country, signup_source,
                   primary_device::text AS primary_device,
                   segment::text AS segment,
                   preferred_price_band::text AS preferred_price_band,
                   onboarding_categories, created_at
            FROM users
        """,
        "orders": """
            SELECT id, user_id, session_id, order_number, status::text AS status,
                   subtotal::float8 AS subtotal, grand_total::float8 AS grand_total,
                   placed_at
            FROM orders
        """,
        "order_items": """
            SELECT id, order_id, product_id, variant_id, quantity,
                   unit_price::float8 AS unit_price, line_total::float8 AS line_total
            FROM order_items
        """,
        "sessions": """
            SELECT id, session_key, user_id, device_type::text AS device_type,
                   started_at, ended_at, event_count, converted
            FROM user_sessions
        """,
    }

    def __init__(
        self,
        engine: object,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> None:
        self.engine = engine
        self.start = start
        self.end = end

    def load(self) -> Dataset:
        params = {"start": self.start, "end": self.end}
        frames = {
            name: pd.read_sql(
                sql, self.engine, params=params if name == "events" else None
            )
            for name, sql in self.QUERIES.items()
        }
        return Dataset(**frames)


def load_dataset(
    data_dir: Path | str | None = None, *, source: DataSource | None = None
) -> Dataset:
    """Convenience loader used by pipelines and tests."""
    if source is not None:
        return source.load()
    from recsys.config.settings import DEFAULT_DATA_DIR

    return ParquetSource(Path(data_dir) if data_dir else DEFAULT_DATA_DIR).load()


__all__ = [
    "INTERACTION_EVENTS",
    "DataSource",
    "Dataset",
    "ParquetSource",
    "PostgresSource",
    "load_dataset",
]
