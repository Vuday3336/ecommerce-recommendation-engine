"""Initial schema: catalogue, users, events, features, recommendations, MLOps.

Hand-reviewed. The table and index operations were generated from the ORM
metadata, then three things were added that autogenerate cannot express and
that the schema genuinely needs:

* the ``vector`` extension, required before any ``VECTOR`` column exists;
* explicit up-front creation of the native enum types, so a type used by
  several tables (``price_band`` is on four) is created once rather than once
  per table, which would fail with "type already exists";
* range partitions for ``user_events`` plus a maintenance function that creates
  future months, since Alembic models the parent table but not its partitions.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-08
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: Native PostgreSQL enum types, created before any table references them.
ENUM_TYPES: dict[str, tuple[str, ...]] = {
    "device_type": (
        'desktop',
        'mobile',
        'tablet',
        'unknown',
    ),
    "event_type": (
        'PRODUCT_VIEW',
        'PRODUCT_CLICK',
        'SEARCH',
        'ADD_TO_CART',
        'REMOVE_FROM_CART',
        'WISHLIST',
        'PURCHASE',
        'PRODUCT_SHARE',
        'PRODUCT_RATING',
        'PRODUCT_REVIEW',
        'SESSION_START',
        'SESSION_END',
    ),
    "experiment_status": (
        'draft',
        'running',
        'paused',
        'completed',
        'aborted',
    ),
    "model_kind": (
        'popularity',
        'content',
        'collaborative_als',
        'collaborative_bpr',
        'hybrid',
        'ranker',
        'embedding',
    ),
    "model_stage": (
        'none',
        'staging',
        'production',
        'archived',
    ),
    "order_status": (
        'pending',
        'paid',
        'shipped',
        'delivered',
        'cancelled',
        'returned',
    ),
    "price_band": (
        'budget',
        'mid',
        'premium',
        'luxury',
    ),
    "recommendation_source": (
        'popularity',
        'trending',
        'content',
        'collaborative',
        'covisitation',
        'frequently_bought_together',
        'category_affinity',
        'brand_affinity',
        'recently_viewed',
        'exploration',
    ),
    "recommendation_surface": (
        'home_for_you',
        'home_because_you_viewed',
        'home_trending',
        'home_continue_shopping',
        'home_frequently_bought_together',
        'pdp_similar',
        'pdp_frequently_bought_together',
        'pdp_also_viewed',
        'search_ranking',
        'recently_viewed',
    ),
    "serving_strategy": (
        'ranker',
        'hybrid',
        'collaborative_content',
        'content_trending',
        'category_popular',
        'global_trending',
        'static_fallback',
    ),
    "user_role": (
        'customer',
        'analyst',
        'admin',
    ),
    "user_segment": (
        'new',
        'casual',
        'regular',
        'high_value',
        'bargain_hunter',
        'window_shopper',
        'inactive',
    ),
}

#: Months of ``user_events`` partitions created up front. The synthetic dataset
#: backfills 180 days, so history must be covered; the forward months keep
#: ingestion working until the maintenance function next runs.
PARTITION_MONTHS_BACK = 13
PARTITION_MONTHS_FORWARD = 3


def _create_enum_types() -> None:
    bind = op.get_bind()
    for name, values in ENUM_TYPES.items():
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)


def _drop_enum_types() -> None:
    bind = op.get_bind()
    for name, values in ENUM_TYPES.items():
        postgresql.ENUM(*values, name=name).drop(bind, checkfirst=True)


def _create_event_partitions() -> None:
    """Create monthly partitions and the function that adds future ones.

    A DEFAULT partition is included as a safety net: an event whose timestamp
    falls outside every declared range is stored rather than rejected. The
    trade-off is real and worth stating - once a row for month M lands in the
    default partition, PostgreSQL refuses to attach a partition for M until
    that row is moved. ``ensure_user_events_partition`` exists so scheduled
    maintenance keeps declared ranges ahead of ingestion and the default stays
    empty; ``scripts/verify_database.py`` asserts that it is.
    """
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ensure_user_events_partition(target date)
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        DECLARE
            month_start date := date_trunc('month', target)::date;
            month_end   date := (date_trunc('month', target) + interval '1 month')::date;
            part_name   text := 'user_events_' || to_char(month_start, 'YYYY_MM');
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
                EXECUTE format(
                    'CREATE TABLE %I PARTITION OF user_events '
                    'FOR VALUES FROM (%L) TO (%L)',
                    part_name, month_start, month_end
                );
            END IF;
        END;
        $$;
        """
    )
    op.execute(
        f"""
        DO $$
        DECLARE
            i int;
        BEGIN
            FOR i IN -{PARTITION_MONTHS_BACK}..{PARTITION_MONTHS_FORWARD} LOOP
                PERFORM ensure_user_events_partition(
                    (date_trunc('month', now()) + (i || ' months')::interval)::date
                );
            END LOOP;
        END;
        $$;
        """
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS user_events_default "
        "PARTITION OF user_events DEFAULT"
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _create_enum_types()

    op.create_table('brands',
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('slug', sa.String(length=140), nullable=False),
    sa.Column('price_tier', postgresql.ENUM('budget', 'mid', 'premium', 'luxury', name='price_band', create_type=False), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_brands')),
    sa.UniqueConstraint('slug', name='uq_brands_slug')
    )
    op.create_index(op.f('ix_brands_created_at'), 'brands', ['created_at'], unique=False)
    op.create_table('categories',
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('slug', sa.String(length=140), nullable=False),
    sa.Column('parent_id', sa.BigInteger(), nullable=True),
    sa.Column('depth', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False, comment="Materialised ancestor path, e.g. 'apparel/footwear/running-shoes'"),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('depth >= 0 AND depth <= 4', name=op.f('ck_categories_depth_range')),
    sa.CheckConstraint('parent_id IS NULL OR parent_id <> id', name=op.f('ck_categories_no_self_parent')),
    sa.ForeignKeyConstraint(['parent_id'], ['categories.id'], name=op.f('fk_categories_parent_id_categories'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_categories')),
    sa.UniqueConstraint('slug', name='uq_categories_slug')
    )
    op.create_index(op.f('ix_categories_created_at'), 'categories', ['created_at'], unique=False)
    op.create_index('ix_categories_parent_id', 'categories', ['parent_id'], unique=False)
    op.create_index('ix_categories_path', 'categories', ['path'], unique=False)
    op.create_table('experiments',
    sa.Column('key', sa.String(length=80), nullable=False, comment='Stable identifier; also the hash salt for assignment (ADR-011)'),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('hypothesis', sa.Text(), server_default='', nullable=False),
    sa.Column('status', postgresql.ENUM('draft', 'running', 'paused', 'completed', 'aborted', name='experiment_status', create_type=False), server_default='draft', nullable=False),
    sa.Column('variants', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False, comment='[{"name": "control", "allocation": 0.5, "config": {...}}, ...]'),
    sa.Column('primary_metric', sa.String(length=60), server_default='ctr', nullable=False),
    sa.Column('guardrail_metrics', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('traffic_allocation', sa.Float(), server_default=sa.text('1.0'), nullable=False, comment='Fraction of eligible traffic entering the experiment at all'),
    sa.Column('minimum_sample_size', sa.Integer(), server_default=sa.text('1000'), nullable=False, comment='Results are not declared before this many users per variant'),
    sa.Column('starts_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('ends_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ends_at IS NULL OR starts_at IS NULL OR ends_at > starts_at', name=op.f('ck_experiments_experiment_window_ordered')),
    sa.CheckConstraint('minimum_sample_size > 0', name=op.f('ck_experiments_minimum_sample_size_positive')),
    sa.CheckConstraint('traffic_allocation > 0 AND traffic_allocation <= 1', name=op.f('ck_experiments_traffic_allocation_range')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_experiments')),
    sa.UniqueConstraint('key', name='uq_experiments_key')
    )
    op.create_index(op.f('ix_experiments_created_at'), 'experiments', ['created_at'], unique=False)
    op.create_index('ix_experiments_status', 'experiments', ['status'], unique=False)
    op.create_table('model_versions',
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('version', sa.String(length=40), nullable=False),
    sa.Column('kind', postgresql.ENUM('popularity', 'content', 'collaborative_als', 'collaborative_bpr', 'hybrid', 'ranker', 'embedding', name='model_kind', create_type=False), nullable=False),
    sa.Column('stage', postgresql.ENUM('none', 'staging', 'production', 'archived', name='model_stage', create_type=False), server_default='none', nullable=False),
    sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
    sa.Column('mlflow_model_uri', sa.String(length=500), nullable=True),
    sa.Column('artifact_path', sa.String(length=500), nullable=True),
    sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('training_started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('training_finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('training_data_from', sa.DateTime(timezone=True), nullable=True),
    sa.Column('training_data_to', sa.DateTime(timezone=True), nullable=True),
    sa.Column('training_rows', sa.BigInteger(), nullable=True),
    sa.Column('promoted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('promoted_by', sa.String(length=120), nullable=True),
    sa.Column('promotion_notes', sa.Text(), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('training_finished_at IS NULL OR training_started_at IS NULL OR training_finished_at >= training_started_at', name=op.f('ck_model_versions_training_time_ordered')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_versions')),
    sa.UniqueConstraint('name', 'version', name='uq_model_versions_name_version')
    )
    op.create_index(op.f('ix_model_versions_created_at'), 'model_versions', ['created_at'], unique=False)
    op.create_index('ix_model_versions_kind_stage', 'model_versions', ['kind', 'stage'], unique=False)
    op.create_index('uq_model_versions_one_production_per_name', 'model_versions', ['name'], unique=True, postgresql_where=sa.text("stage = 'production'"))
    op.create_table('user_events',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False, comment='Business timestamp and partition key'),
    sa.Column('event_type', postgresql.ENUM('PRODUCT_VIEW', 'PRODUCT_CLICK', 'SEARCH', 'ADD_TO_CART', 'REMOVE_FROM_CART', 'WISHLIST', 'PURCHASE', 'PRODUCT_SHARE', 'PRODUCT_RATING', 'PRODUCT_REVIEW', 'SESSION_START', 'SESSION_END', name='event_type', create_type=False), nullable=False),
    sa.Column('user_id', sa.BigInteger(), nullable=True),
    sa.Column('session_key', sa.String(length=64), nullable=False),
    sa.Column('product_id', sa.BigInteger(), nullable=True),
    sa.Column('source', sa.String(length=64), server_default='unknown', nullable=False, comment='Surface that produced the event: homepage, pdp, search, rail id'),
    sa.Column('device_type', postgresql.ENUM('desktop', 'mobile', 'tablet', 'unknown', name='device_type', create_type=False), server_default='unknown', nullable=False),
    sa.Column('recommendation_id', sa.BigInteger(), nullable=True),
    sa.Column('event_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("event_type NOT IN ('PRODUCT_VIEW','PRODUCT_CLICK','ADD_TO_CART','REMOVE_FROM_CART','WISHLIST','PURCHASE','PRODUCT_SHARE','PRODUCT_RATING','PRODUCT_REVIEW') OR product_id IS NOT NULL", name=op.f('ck_user_events_product_scoped_events_have_product')),
    sa.PrimaryKeyConstraint('id', 'occurred_at', name=op.f('pk_user_events')),
    postgresql_partition_by='RANGE (occurred_at)'
    )
    op.create_index('ix_user_events_occurred_at', 'user_events', [sa.literal_column('occurred_at DESC')], unique=False)
    op.create_index('ix_user_events_product_id_event_type', 'user_events', ['product_id', 'event_type'], unique=False)
    op.create_index('ix_user_events_recommendation_id', 'user_events', ['recommendation_id'], unique=False)
    op.create_index('ix_user_events_session_key', 'user_events', ['session_key'], unique=False)
    op.create_index('ix_user_events_user_id_occurred_at', 'user_events', ['user_id', sa.literal_column('occurred_at DESC')], unique=False)
    op.create_table('users',
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=True),
    sa.Column('full_name', sa.String(length=200), nullable=False),
    sa.Column('role', postgresql.ENUM('customer', 'analyst', 'admin', name='user_role', create_type=False), server_default='customer', nullable=False),
    sa.Column('country', sa.String(length=2), server_default='US', nullable=False),
    sa.Column('signup_source', sa.String(length=40), server_default='organic', nullable=False),
    sa.Column('primary_device', postgresql.ENUM('desktop', 'mobile', 'tablet', 'unknown', name='device_type', create_type=False), server_default='desktop', nullable=False),
    sa.Column('segment', postgresql.ENUM('new', 'casual', 'regular', 'high_value', 'bargain_hunter', 'window_shopper', 'inactive', name='user_segment', create_type=False), server_default='new', nullable=False),
    sa.Column('preferred_price_band', postgresql.ENUM('budget', 'mid', 'premium', 'luxury', name='price_band', create_type=False), server_default='mid', nullable=False),
    sa.Column('onboarding_categories', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False, comment='Category ids picked at signup; seeds the new-user cold start (FR-09)'),
    sa.Column('lifetime_orders', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('lifetime_spend', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('last_active_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('public_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('char_length(country) = 2', name=op.f('ck_users_country_is_iso2')),
    sa.CheckConstraint('lifetime_orders >= 0', name=op.f('ck_users_lifetime_orders_non_negative')),
    sa.CheckConstraint('lifetime_spend >= 0', name=op.f('ck_users_lifetime_spend_non_negative')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name='uq_users_email'),
    sa.UniqueConstraint('public_id', name=op.f('uq_users_public_id'))
    )
    op.create_index(op.f('ix_users_created_at'), 'users', ['created_at'], unique=False)
    op.create_index('ix_users_last_active_at', 'users', [sa.literal_column('last_active_at DESC')], unique=False)
    op.create_index('ix_users_segment', 'users', ['segment'], unique=False)
    op.create_table('experiment_assignments',
    sa.Column('experiment_id', sa.BigInteger(), nullable=False),
    sa.Column('user_id', sa.BigInteger(), nullable=True),
    sa.Column('session_key', sa.String(length=64), nullable=True),
    sa.Column('variant', sa.String(length=40), nullable=False),
    sa.Column('bucket', sa.Integer(), nullable=False, comment='0-9999 hash bucket, stored for auditability'),
    sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('first_exposure_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.CheckConstraint('bucket >= 0 AND bucket < 10000', name=op.f('ck_experiment_assignments_bucket_range')),
    sa.CheckConstraint('user_id IS NOT NULL OR session_key IS NOT NULL', name=op.f('ck_experiment_assignments_assignment_has_subject')),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], name=op.f('fk_experiment_assignments_experiment_id_experiments'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_experiment_assignments_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_experiment_assignments')),
    sa.UniqueConstraint('experiment_id', 'user_id', name='uq_experiment_assignments_experiment_id_user_id')
    )
    op.create_index('ix_experiment_assignments_experiment_variant', 'experiment_assignments', ['experiment_id', 'variant'], unique=False)
    op.create_index('ix_experiment_assignments_user_id', 'experiment_assignments', ['user_id'], unique=False)
    op.create_table('products',
    sa.Column('sku', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=300), nullable=False),
    sa.Column('description', sa.Text(), server_default='', nullable=False),
    sa.Column('category_id', sa.BigInteger(), nullable=False),
    sa.Column('brand_id', sa.BigInteger(), nullable=False),
    sa.Column('price', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('cost', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('price_band', postgresql.ENUM('budget', 'mid', 'premium', 'luxury', name='price_band', create_type=False), nullable=False),
    sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False, comment='Category-specific attributes (colour, size, material, ...)'),
    sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('stock_quantity', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('rating_average', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('rating_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('view_count', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('purchase_count', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('released_at', sa.DateTime(timezone=True), nullable=False, comment='Catalogue entry date; drives the new-product cold-start path'),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('public_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('cost >= 0', name=op.f('ck_products_cost_non_negative')),
    sa.CheckConstraint('price > 0', name=op.f('ck_products_price_positive')),
    sa.CheckConstraint('rating_average >= 0 AND rating_average <= 5', name=op.f('ck_products_rating_range')),
    sa.CheckConstraint('rating_count >= 0', name=op.f('ck_products_rating_count_non_negative')),
    sa.CheckConstraint('stock_quantity >= 0', name=op.f('ck_products_stock_non_negative')),
    sa.ForeignKeyConstraint(['brand_id'], ['brands.id'], name=op.f('fk_products_brand_id_brands'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_products_category_id_categories'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_products')),
    sa.UniqueConstraint('public_id', name=op.f('uq_products_public_id')),
    sa.UniqueConstraint('sku', name='uq_products_sku')
    )
    op.create_index('ix_products_active_brand', 'products', ['brand_id'], unique=False, postgresql_where=sa.text('is_active AND stock_quantity > 0'))
    op.create_index('ix_products_active_category', 'products', ['category_id', 'price'], unique=False, postgresql_where=sa.text('is_active AND stock_quantity > 0'))
    op.create_index('ix_products_attributes', 'products', ['attributes'], unique=False, postgresql_using='gin')
    op.create_index(op.f('ix_products_created_at'), 'products', ['created_at'], unique=False)
    op.create_index('ix_products_purchase_count', 'products', [sa.literal_column('purchase_count DESC')], unique=False)
    op.create_index('ix_products_released_at', 'products', [sa.literal_column('released_at DESC')], unique=False)
    op.create_index('ix_products_search_tsv', 'products', [sa.literal_column("to_tsvector('english', name || ' ' || description)")], unique=False, postgresql_using='gin')
    op.create_table('user_features',
    sa.Column('user_id', sa.BigInteger(), nullable=False),
    sa.Column('as_of_ts', sa.DateTime(timezone=True), nullable=False, comment='Cut-off used to build this row; no input event is at or after it'),
    sa.Column('total_purchases', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('total_spending', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('avg_order_value', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('customer_lifetime_value', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False, comment='Historical margin plus a simple frequency/recency projection'),
    sa.Column('total_events', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('distinct_products_viewed', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('session_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('avg_session_duration_s', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('active_days', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('view_to_cart_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('cart_to_purchase_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('view_to_purchase_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('purchase_frequency_days', sa.Float(), nullable=True, comment='Mean days between orders; NULL for users with fewer than 2 orders'),
    sa.Column('days_since_last_event', sa.Float(), nullable=True),
    sa.Column('days_since_last_purchase', sa.Float(), nullable=True),
    sa.Column('days_since_first_seen', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('avg_purchase_price', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('median_purchase_price', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('price_sensitivity', sa.Float(), server_default=sa.text('0'), nullable=False, comment='0 = indifferent to price, 1 = strongly prefers the cheap end'),
    sa.Column('dominant_price_band', postgresql.ENUM('budget', 'mid', 'premium', 'luxury', name='price_band', create_type=False), server_default='mid', nullable=False),
    sa.Column('category_affinity', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('brand_affinity', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('top_categories', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('top_brands', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('taste_embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('price_sensitivity >= 0 AND price_sensitivity <= 1', name=op.f('ck_user_features_price_sensitivity_range')),
    sa.CheckConstraint('total_purchases >= 0', name=op.f('ck_user_features_total_purchases_non_negative')),
    sa.CheckConstraint('view_to_purchase_rate >= 0 AND view_to_purchase_rate <= 1', name=op.f('ck_user_features_view_to_purchase_rate_range')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_features_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', name=op.f('pk_user_features'))
    )
    op.create_index('ix_user_features_as_of_ts', 'user_features', [sa.literal_column('as_of_ts DESC')], unique=False)
    op.create_index(op.f('ix_user_features_created_at'), 'user_features', ['created_at'], unique=False)
    op.create_table('user_sessions',
    sa.Column('session_key', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.BigInteger(), nullable=True),
    sa.Column('device_type', postgresql.ENUM('desktop', 'mobile', 'tablet', 'unknown', name='device_type', create_type=False), server_default='unknown', nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('event_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('converted', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ended_at IS NULL OR ended_at >= started_at', name=op.f('ck_user_sessions_session_time_ordered')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_sessions_user_id_users'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_sessions')),
    sa.UniqueConstraint('session_key', name='uq_user_sessions_session_key')
    )
    op.create_index(op.f('ix_user_sessions_created_at'), 'user_sessions', ['created_at'], unique=False)
    op.create_index('ix_user_sessions_user_id_started_at', 'user_sessions', ['user_id', sa.literal_column('started_at DESC')], unique=False)
    op.create_table('orders',
    sa.Column('user_id', sa.BigInteger(), nullable=False),
    sa.Column('session_id', sa.BigInteger(), nullable=True),
    sa.Column('order_number', sa.String(length=32), nullable=False),
    sa.Column('status', postgresql.ENUM('pending', 'paid', 'shipped', 'delivered', 'cancelled', 'returned', name='order_status', create_type=False), server_default='paid', nullable=False),
    sa.Column('subtotal', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('discount_total', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('shipping_total', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('grand_total', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('currency', sa.String(length=3), server_default='USD', nullable=False),
    sa.Column('placed_at', sa.DateTime(timezone=True), nullable=False, comment='Business timestamp; all temporal logic reads this, not created_at'),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('public_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('discount_total >= 0', name=op.f('ck_orders_discount_non_negative')),
    sa.CheckConstraint('grand_total >= 0', name=op.f('ck_orders_grand_total_non_negative')),
    sa.CheckConstraint('subtotal >= 0', name=op.f('ck_orders_subtotal_non_negative')),
    sa.ForeignKeyConstraint(['session_id'], ['user_sessions.id'], name=op.f('fk_orders_session_id_user_sessions'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_orders_user_id_users'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_orders')),
    sa.UniqueConstraint('order_number', name='uq_orders_order_number'),
    sa.UniqueConstraint('public_id', name=op.f('uq_orders_public_id'))
    )
    op.create_index(op.f('ix_orders_created_at'), 'orders', ['created_at'], unique=False)
    op.create_index('ix_orders_placed_at', 'orders', [sa.literal_column('placed_at DESC')], unique=False)
    op.create_index('ix_orders_session_id', 'orders', ['session_id'], unique=False)
    op.create_index('ix_orders_status', 'orders', ['status'], unique=False)
    op.create_index('ix_orders_user_id_placed_at', 'orders', ['user_id', sa.literal_column('placed_at DESC')], unique=False)
    op.create_table('product_embeddings',
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=False),
    sa.Column('model_name', sa.String(length=120), nullable=False),
    sa.Column('model_version', sa.String(length=40), nullable=False),
    sa.Column('source_hash', sa.String(length=64), nullable=False),
    sa.Column('dim', sa.Integer(), server_default=sa.text('384'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('dim = 384', name=op.f('ck_product_embeddings_dim_matches_model')),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_product_embeddings_product_id_products'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('product_id', name=op.f('pk_product_embeddings'))
    )
    op.create_index(op.f('ix_product_embeddings_created_at'), 'product_embeddings', ['created_at'], unique=False)
    op.create_index('ix_product_embeddings_hnsw_cosine', 'product_embeddings', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('ix_product_embeddings_model', 'product_embeddings', ['model_name', 'model_version'], unique=False)
    op.create_table('product_features',
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('as_of_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('view_count_total', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('view_count_30d', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('view_count_7d', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('cart_count_30d', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('purchase_count_total', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('purchase_count_30d', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('distinct_buyers', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('revenue_30d', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('view_to_cart_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('conversion_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('repeat_purchase_rate', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('popularity_score', sa.Numeric(precision=10, scale=6), server_default=sa.text('0'), nullable=False, comment='Normalised blend of purchases, views and revenue over the window'),
    sa.Column('trending_score', sa.Numeric(precision=10, scale=6), server_default=sa.text('0'), nullable=False, comment='Time-decayed velocity; short half-life so it reacts within hours'),
    sa.Column('novelty_score', sa.Numeric(precision=10, scale=6), server_default=sa.text('0'), nullable=False, comment='Inverse log popularity; used by the diversity re-ranker'),
    sa.Column('quality_score', sa.Numeric(precision=10, scale=6), server_default=sa.text('0'), nullable=False, comment='Bayesian-smoothed rating, so a single 5-star review cannot top the chart'),
    sa.Column('days_since_release', sa.Float(), server_default=sa.text('0'), nullable=False),
    sa.Column('is_cold', sa.Boolean(), server_default=sa.text('true'), nullable=False, comment='Too few interactions for collaborative signal; routes to content path'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('conversion_rate >= 0 AND conversion_rate <= 1', name=op.f('ck_product_features_conversion_rate_range')),
    sa.CheckConstraint('popularity_score >= 0', name=op.f('ck_product_features_popularity_non_negative')),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_product_features_product_id_products'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('product_id', name=op.f('pk_product_features'))
    )
    op.create_index(op.f('ix_product_features_created_at'), 'product_features', ['created_at'], unique=False)
    op.create_index('ix_product_features_is_cold', 'product_features', ['is_cold'], unique=False)
    op.create_index('ix_product_features_popularity', 'product_features', [sa.literal_column('popularity_score DESC')], unique=False)
    op.create_index('ix_product_features_trending', 'product_features', [sa.literal_column('trending_score DESC')], unique=False)
    op.create_table('product_variants',
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('sku', sa.String(length=72), nullable=False),
    sa.Column('variant_name', sa.String(length=160), nullable=False),
    sa.Column('options', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('price_delta', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('stock_quantity', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('stock_quantity >= 0', name=op.f('ck_product_variants_stock_non_negative')),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_product_variants_product_id_products'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_product_variants')),
    sa.UniqueConstraint('sku', name='uq_product_variants_sku')
    )
    op.create_index(op.f('ix_product_variants_created_at'), 'product_variants', ['created_at'], unique=False)
    op.create_index('ix_product_variants_product_id', 'product_variants', ['product_id'], unique=False)
    op.create_table('recommendations',
    sa.Column('request_id', sa.UUID(), nullable=False, comment='Groups the items returned by a single API call'),
    sa.Column('user_id', sa.BigInteger(), nullable=True),
    sa.Column('session_key', sa.String(length=64), nullable=False),
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('surface', postgresql.ENUM('home_for_you', 'home_because_you_viewed', 'home_trending', 'home_continue_shopping', 'home_frequently_bought_together', 'pdp_similar', 'pdp_frequently_bought_together', 'pdp_also_viewed', 'search_ranking', 'recently_viewed', name='recommendation_surface', create_type=False), nullable=False),
    sa.Column('source', postgresql.ENUM('popularity', 'trending', 'content', 'collaborative', 'covisitation', 'frequently_bought_together', 'category_affinity', 'brand_affinity', 'recently_viewed', 'exploration', name='recommendation_source', create_type=False), nullable=False, comment='Candidate generator that produced this item'),
    sa.Column('strategy', postgresql.ENUM('ranker', 'hybrid', 'collaborative_content', 'content_trending', 'category_popular', 'global_trending', 'static_fallback', name='serving_strategy', create_type=False), nullable=False, comment='Degradation-ladder rung that served the request'),
    sa.Column('position', sa.SmallInteger(), nullable=False, comment='Zero-based slot; needed for position bias'),
    sa.Column('score', sa.Numeric(precision=10, scale=6), nullable=False),
    sa.Column('score_components', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('explanation', sa.Text(), server_default='', nullable=False, comment='Human-readable reason (FR-11)'),
    sa.Column('explanation_evidence', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False, comment='Machine-readable support: anchor products, SHAP contributions'),
    sa.Column('model_version_id', sa.BigInteger(), nullable=True),
    sa.Column('experiment_id', sa.BigInteger(), nullable=True),
    sa.Column('variant', sa.String(length=40), nullable=True),
    sa.Column('candidate_pool_size', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('latency_ms', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('cache_hit', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('served_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.CheckConstraint('latency_ms >= 0', name=op.f('ck_recommendations_latency_non_negative')),
    sa.CheckConstraint('position >= 0', name=op.f('ck_recommendations_position_non_negative')),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], name=op.f('fk_recommendations_experiment_id_experiments'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], name=op.f('fk_recommendations_model_version_id_model_versions'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_recommendations_product_id_products'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_recommendations_user_id_users'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recommendations')),
    sa.UniqueConstraint('request_id', 'position', name='uq_recommendations_request_id_position')
    )
    op.create_index('ix_recommendations_experiment_variant', 'recommendations', ['experiment_id', 'variant'], unique=False)
    op.create_index('ix_recommendations_model_version_id', 'recommendations', ['model_version_id'], unique=False)
    op.create_index('ix_recommendations_product_id_served_at', 'recommendations', ['product_id', sa.literal_column('served_at DESC')], unique=False)
    op.create_index('ix_recommendations_request_id', 'recommendations', ['request_id'], unique=False)
    op.create_index('ix_recommendations_surface_served_at', 'recommendations', ['surface', sa.literal_column('served_at DESC')], unique=False)
    op.create_index('ix_recommendations_user_id_served_at', 'recommendations', ['user_id', sa.literal_column('served_at DESC')], unique=False)
    op.create_table('user_product_interactions',
    sa.Column('user_id', sa.BigInteger(), nullable=False),
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('view_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('click_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('cart_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('cart_removal_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('wishlist_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('purchase_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('share_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('rating', sa.Float(), nullable=True),
    sa.Column('total_quantity', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('total_spend', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('implicit_weight', sa.Numeric(precision=10, scale=6), server_default=sa.text('0'), nullable=False, comment='Confidence for ALS; derived from event mix and recency (ADR-002)'),
    sa.Column('recency_weight', sa.Numeric(precision=10, scale=6), server_default=sa.text('1'), nullable=False),
    sa.Column('first_interaction_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_interaction_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('implicit_weight >= 0', name=op.f('ck_user_product_interactions_implicit_weight_non_negative')),
    sa.CheckConstraint('last_interaction_at >= first_interaction_at', name=op.f('ck_user_product_interactions_interaction_ordered')),
    sa.CheckConstraint('purchase_count >= 0', name=op.f('ck_user_product_interactions_purchase_count_non_negative')),
    sa.CheckConstraint('rating IS NULL OR (rating >= 1 AND rating <= 5)', name=op.f('ck_user_product_interactions_rating_range')),
    sa.CheckConstraint('view_count >= 0', name=op.f('ck_user_product_interactions_view_count_non_negative')),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_user_product_interactions_product_id_products'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_product_interactions_user_id_users'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('user_id', 'product_id', name=op.f('pk_user_product_interactions'))
    )
    op.create_index('ix_user_product_interactions_last_interaction_at', 'user_product_interactions', [sa.literal_column('last_interaction_at DESC')], unique=False)
    op.create_index('ix_user_product_interactions_product_id', 'user_product_interactions', ['product_id'], unique=False)
    op.create_index('ix_user_product_interactions_user_weight', 'user_product_interactions', ['user_id', sa.literal_column('implicit_weight DESC')], unique=False)
    op.create_table('order_items',
    sa.Column('order_id', sa.BigInteger(), nullable=False),
    sa.Column('product_id', sa.BigInteger(), nullable=False),
    sa.Column('variant_id', sa.BigInteger(), nullable=True),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('unit_price', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('discount', sa.Numeric(precision=12, scale=2), server_default=sa.text('0'), nullable=False),
    sa.Column('line_total', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('source_recommendation_id', sa.BigInteger(), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('line_total >= 0', name=op.f('ck_order_items_line_total_non_negative')),
    sa.CheckConstraint('quantity > 0', name=op.f('ck_order_items_quantity_positive')),
    sa.CheckConstraint('unit_price >= 0', name=op.f('ck_order_items_unit_price_non_negative')),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], name=op.f('fk_order_items_order_id_orders'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['product_id'], ['products.id'], name=op.f('fk_order_items_product_id_products'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['source_recommendation_id'], ['recommendations.id'], name=op.f('fk_order_items_source_recommendation_id_recommendations'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['variant_id'], ['product_variants.id'], name=op.f('fk_order_items_variant_id_product_variants'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_order_items')),
    sa.UniqueConstraint('order_id', 'variant_id', name='uq_order_items_order_id_variant_id')
    )
    op.create_index(op.f('ix_order_items_created_at'), 'order_items', ['created_at'], unique=False)
    op.create_index('ix_order_items_order_id', 'order_items', ['order_id'], unique=False)
    op.create_index('ix_order_items_product_id', 'order_items', ['product_id'], unique=False)
    op.create_index('ix_order_items_source_recommendation_id', 'order_items', ['source_recommendation_id'], unique=False, postgresql_where=sa.text('source_recommendation_id IS NOT NULL'))
    op.create_index('ix_order_items_variant_id', 'order_items', ['variant_id'], unique=False)
    op.create_table('recommendation_clicks',
    sa.Column('recommendation_id', sa.BigInteger(), nullable=False),
    sa.Column('clicked_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('dwell_ms', sa.Integer(), nullable=True, comment='Time on the destination page; a proxy for click quality'),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], name=op.f('fk_recommendation_clicks_recommendation_id_recommendations'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recommendation_clicks'))
    )
    op.create_index('ix_recommendation_clicks_clicked_at', 'recommendation_clicks', [sa.literal_column('clicked_at DESC')], unique=False)
    op.create_index('ix_recommendation_clicks_recommendation_id', 'recommendation_clicks', ['recommendation_id'], unique=False)
    op.create_table('recommendation_impressions',
    sa.Column('recommendation_id', sa.BigInteger(), nullable=False),
    sa.Column('shown_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('visible_ms', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('viewport_position', sa.SmallInteger(), nullable=True),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.CheckConstraint('visible_ms >= 0', name=op.f('ck_recommendation_impressions_visible_ms_non_negative')),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], name=op.f('fk_recommendation_impressions_recommendation_id_recommendations'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recommendation_impressions')),
    sa.UniqueConstraint('recommendation_id', name='uq_recommendation_impressions_recommendation_id')
    )
    op.create_index('ix_recommendation_impressions_shown_at', 'recommendation_impressions', [sa.literal_column('shown_at DESC')], unique=False)
    op.create_table('recommendation_conversions',
    sa.Column('recommendation_id', sa.BigInteger(), nullable=False),
    sa.Column('order_id', sa.BigInteger(), nullable=False),
    sa.Column('order_item_id', sa.BigInteger(), nullable=True),
    sa.Column('converted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('quantity', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('revenue', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('attribution_window_hours', sa.Integer(), server_default=sa.text('168'), nullable=False),
    sa.Column('attribution_model', sa.String(length=32), server_default='last_click', nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.CheckConstraint('quantity > 0', name=op.f('ck_recommendation_conversions_quantity_positive')),
    sa.CheckConstraint('revenue >= 0', name=op.f('ck_recommendation_conversions_revenue_non_negative')),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], name=op.f('fk_recommendation_conversions_order_id_orders'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['order_item_id'], ['order_items.id'], name=op.f('fk_recommendation_conversions_order_item_id_order_items'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], name=op.f('fk_recommendation_conversions_recommendation_id_recommendations'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recommendation_conversions')),
    sa.UniqueConstraint('recommendation_id', 'order_item_id', name='uq_recommendation_conversions_recommendation_id_order_item_id')
    )
    op.create_index('ix_recommendation_conversions_converted_at', 'recommendation_conversions', [sa.literal_column('converted_at DESC')], unique=False)
    op.create_index('ix_recommendation_conversions_order_id', 'recommendation_conversions', ['order_id'], unique=False)
    op.create_index('ix_recommendation_conversions_order_item_id', 'recommendation_conversions', ['order_item_id'], unique=False)
    op.create_index('ix_recommendation_conversions_recommendation_id', 'recommendation_conversions', ['recommendation_id'], unique=False)

    _create_event_partitions()


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS ensure_user_events_partition(date)")
    op.drop_table('recommendation_conversions')
    op.drop_table('recommendation_impressions')
    op.drop_table('recommendation_clicks')
    op.drop_table('order_items')
    op.drop_table('user_product_interactions')
    op.drop_table('recommendations')
    op.drop_table('product_variants')
    op.drop_table('product_features')
    op.drop_table('product_embeddings')
    op.drop_table('orders')
    op.drop_table('user_sessions')
    op.drop_table('user_features')
    op.drop_table('products')
    op.drop_table('experiment_assignments')
    op.drop_table('users')
    op.drop_table('user_events')
    op.drop_table('model_versions')
    op.drop_table('experiments')
    op.drop_table('categories')
    op.drop_table('brands')
    _drop_enum_types()
