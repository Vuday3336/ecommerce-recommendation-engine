"""Model registry mirror, experiments and experiment assignments."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin
from app.db.types import JSONB, pg_enum
from app.models.enums import ExperimentStatus, ModelKind, ModelStage

if TYPE_CHECKING:
    from app.models.users import User


class ModelVersion(IdMixin, TimestampMixin, Base):
    """Serving-side mirror of one MLflow registered model version (ADR-010).

    MLflow remains the system of record for lineage, parameters and artefacts.
    This table exists because every served recommendation is stamped with a
    `model_version_id`, and the analytics that join those rows to model
    metadata must be plain SQL. Reaching into the MLflow REST API from a
    dashboard query would couple reporting to MLflow uptime and make
    "revenue by model version" a two-system operation.

    The mirror is written by the promotion step, so it cannot silently diverge:
    a promotion that fails to update this table fails outright.
    """

    __tablename__ = "model_versions"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    kind: Mapped[ModelKind] = mapped_column(pg_enum(ModelKind, "model_kind"), nullable=False)
    stage: Mapped[ModelStage] = mapped_column(
        pg_enum(ModelStage, "model_stage"),
        nullable=False,
        server_default=ModelStage.NONE.value,
    )

    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_model_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    #: Snapshot of the offline evaluation that justified this version, so the
    #: promotion decision is auditable without re-reading MLflow.
    metrics: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    params: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    training_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_finished_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_data_from: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_data_to: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_rows: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    promoted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    promotion_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        CheckConstraint(
            "training_finished_at IS NULL OR training_started_at IS NULL "
            "OR training_finished_at >= training_started_at",
            name="training_time_ordered",
        ),
        # At most one Production version per model name. This is the constraint
        # that makes "which model is live?" unambiguous; enforcing it in the
        # database rather than in promotion code means a race between two
        # concurrent promotions fails loudly instead of leaving two live models.
        Index(
            "uq_model_versions_one_production_per_name",
            "name",
            unique=True,
            postgresql_where=text("stage = 'production'"),
        ),
        Index("ix_model_versions_kind_stage", "kind", "stage"),
    )


class Experiment(IdMixin, TimestampMixin, Base):
    """An online A/B test.

    `variants` is JSONB rather than a child table: a variant definition is a
    small, write-once config blob (name, allocation, engine settings) that is
    always read as a whole with its experiment. A child table would add a join
    to the assignment hot path for no query we actually run.
    """

    __tablename__ = "experiments"

    key: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        comment="Stable identifier; also the hash salt for assignment (ADR-011)",
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    status: Mapped[ExperimentStatus] = mapped_column(
        pg_enum(ExperimentStatus, "experiment_status"),
        nullable=False,
        server_default=ExperimentStatus.DRAFT.value,
    )

    variants: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment='[{"name": "control", "allocation": 0.5, "config": {...}}, ...]',
    )
    primary_metric: Mapped[str] = mapped_column(
        String(60), nullable=False, server_default="ctr"
    )
    guardrail_metrics: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    traffic_allocation: Mapped[float] = mapped_column(
        nullable=False,
        server_default=text("1.0"),
        comment="Fraction of eligible traffic entering the experiment at all",
    )
    minimum_sample_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1000"),
        comment="Results are not declared before this many users per variant",
    )

    starts_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ends_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    assignments: Mapped[list[ExperimentAssignment]] = relationship(
        back_populates="experiment"
    )

    __table_args__ = (
        UniqueConstraint("key", name="uq_experiments_key"),
        CheckConstraint(
            "traffic_allocation > 0 AND traffic_allocation <= 1",
            name="traffic_allocation_range",
        ),
        CheckConstraint("minimum_sample_size > 0", name="minimum_sample_size_positive"),
        CheckConstraint(
            "ends_at IS NULL OR starts_at IS NULL OR ends_at > starts_at",
            name="experiment_window_ordered",
        ),
        Index("ix_experiments_status", "status"),
    )


class ExperimentAssignment(IdMixin, Base):
    """A user's bucket for one experiment.

    Assignment is *computed* by hashing (ADR-011), not looked up - the hot path
    never reads this table. It is written asynchronously on first exposure so
    that analysis has an exposure timestamp, and so a sample-ratio-mismatch
    check has real observed counts to test against the intended allocation.
    """

    __tablename__ = "experiment_assignments"

    experiment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    session_key: Mapped[str | None] = mapped_column(String(64), nullable=True)

    variant: Mapped[str] = mapped_column(String(40), nullable=False)
    bucket: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="0-9999 hash bucket, stored for auditability"
    )
    assigned_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    first_exposure_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    experiment: Mapped[Experiment] = relationship(back_populates="assignments")
    user: Mapped[User | None] = relationship()

    __table_args__ = (
        CheckConstraint("bucket >= 0 AND bucket < 10000", name="bucket_range"),
        CheckConstraint(
            "user_id IS NOT NULL OR session_key IS NOT NULL",
            name="assignment_has_subject",
        ),
        UniqueConstraint(
            "experiment_id", "user_id", name="uq_experiment_assignments_experiment_id_user_id"
        ),
        Index("ix_experiment_assignments_experiment_variant", "experiment_id", "variant"),
        Index("ix_experiment_assignments_user_id", "user_id"),
    )
