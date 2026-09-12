"""Configuration for the recommendation engine.

Every tunable lives here as a typed dataclass. Two reasons this matters beyond
tidiness: a training run is fully described by this object plus a seed, which
is what makes results reproducible (NFR-05); and the exact configuration is
logged to MLflow with each run, so "which weights produced this model?" is
answerable months later instead of being archaeology.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "synthetic"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Temporal split boundaries (ADR-007).

    Expressed as fractions of the observed time span rather than fixed dates,
    so the same config works on a 90-day sample and a 180-day full run.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    #: Remainder is test. Stated explicitly so the three always sum to 1.

    #: Users must have at least this many training interactions to be scored.
    #: Not a quality filter - it is how the evaluation avoids reporting a
    #: meaningless number for a user the model was never given a chance to
    #: learn. Cold users are evaluated separately, by segment.
    min_train_interactions: int = 3
    #: Items with fewer than this many training interactions are still
    #: recommendable (that is the point of content-based retrieval) but are
    #: excluded from collaborative factorisation, where they only add noise.
    min_item_interactions: int = 2

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction


@dataclass(frozen=True, slots=True)
class WeightingConfig:
    """Implicit-feedback weighting (ADR-002).

    `calibrate` is the switch between deriving weights from observed conversion
    odds and using the priors below. The priors exist so the pipeline runs on a
    dataset too small to calibrate against; every real run calibrates.
    """

    calibrate: bool = True
    #: Window for "did this event lead to a purchase of that product?"
    conversion_window_days: int = 30
    #: Fallback ordering if calibration cannot run. Deliberately not the final
    #: numbers - the calibrated values replace these and are logged.
    prior_weights: dict[str, float] = field(
        default_factory=lambda: {
            "PURCHASE": 10.0,
            "ADD_TO_CART": 5.0,
            "WISHLIST": 4.0,
            "PRODUCT_RATING": 3.0,
            "PRODUCT_REVIEW": 3.0,
            "PRODUCT_SHARE": 2.5,
            "PRODUCT_CLICK": 1.5,
            "PRODUCT_VIEW": 1.0,
            "REMOVE_FROM_CART": -1.5,
        }
    )
    #: Half-life in days for recency decay on interaction weight. 45 days is
    #: roughly the median repeat-purchase interval in this catalogue, so an
    #: interaction has decayed to half its strength by the time a typical user
    #: would naturally reconsider the category.
    recency_half_life_days: float = 45.0
    #: Cap on a single (user, item) weight, so one obsessive user cannot
    #: dominate an item's factor vector.
    max_weight: float = 60.0


@dataclass(frozen=True, slots=True)
class PopularityConfig:
    #: Prior weight for Bayesian smoothing of conversion rates. Without it, a
    #: product viewed twice and bought once outranks everything at a 50%
    #: conversion rate.
    smoothing_prior: float = 20.0
    trending_half_life_hours: float = 6.0
    trending_window_hours: int = 24


@dataclass(frozen=True, slots=True)
class ContentConfig:
    """Content model settings.

    `backend` selects the text representation. `tfidf_svd` is the default
    because it is deterministic, trains in seconds, needs no model download and
    no GPU, and on a catalogue whose descriptions are template-generated it
    captures essentially the same structure a transformer would. The
    `sentence_transformer` backend implements the same interface and is used
    when semantic nuance in free-text descriptions actually matters - the
    choice is measured in the evaluation report, not asserted.
    """

    backend: str = "tfidf_svd"
    embedding_dim: int = 384
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tfidf_features: int = 30_000
    ngram_range: tuple[int, int] = (1, 2)
    #: Weight of structured attributes versus free text when composing the
    #: final product vector. Category and brand are strong, reliable signals;
    #: description text adds nuance but is noisier.
    text_weight: float = 0.6
    attribute_weight: float = 0.4


@dataclass(frozen=True, slots=True)
class CollaborativeConfig:
    """ALS settings (ADR-003)."""

    factors: int = 64
    regularization: float = 0.10
    iterations: int = 20
    #: `c_ui = 1 + alpha * r_ui`. Higher alpha means the model trusts observed
    #: interactions more relative to the unobserved zeros.
    #:
    #: Selected by grid search on the *validation* fold, not the test fold.
    #: The value matters more than it looks: the calibrated weights already
    #: average about 25, so the textbook alpha of 40 would give confidences
    #: near 1000 and drive the factorisation to fit heavy users almost
    #: exclusively. Moving from 24 to 2 lifted validation NDCG@10 from 0.038 to
    #: 0.051 - a 35% relative gain from one number, and a good illustration of
    #: why alpha must be tuned against the weight scale it multiplies rather
    #: than copied from a paper that used binary interactions.
    alpha: float = 2.0
    use_native: bool = True
    calculate_training_loss: bool = False
    random_state: int = 42

    #: BPR is trained for comparison only (ADR-003).
    bpr_factors: int = 96
    bpr_learning_rate: float = 0.01
    bpr_regularization: float = 0.01
    bpr_iterations: int = 120


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Adaptive blend weights (ADR-014).

    Weights are a function of how much history a user has. A user with two
    clicks must not be scored mostly by a collaborative model that knows
    nothing about them; a user with two hundred should be.

    Each tuple is (collaborative, content, trending, affinity, popularity) and
    must sum to 1.
    """

    #: Selected by randomised simplex search on the *validation* fold, scored
    #: per segment. The raw per-segment argmax was
    #:   cold (0.07, 0.07, 0.00, 0.05, 0.81)
    #:   sparse (0.38, 0.11, 0.03, 0.29, 0.19)
    #:   warm (0.10, 0.20, 0.10, 0.15, 0.45)
    #:   rich (0.55, 0.18, 0.08, 0.14, 0.05)
    #: which confirms the ADR-014 hypothesis at the extremes - popularity falls
    #: from 0.81 to 0.05 and collaborative rises from 0.07 to 0.55 as history
    #: deepens - but is noisy in the middle, where each segment holds only a few
    #: hundred evaluation users. The shipped values are therefore constrained to
    #: be monotone in history depth rather than taken as the raw argmax. That is
    #: a deliberate bias-variance trade: it costs a little validation NDCG in the
    #: warm bucket and avoids fitting the search to segment sampling noise.
    cold: tuple[float, float, float, float, float] = (0.05, 0.10, 0.05, 0.05, 0.75)
    sparse: tuple[float, float, float, float, float] = (0.30, 0.14, 0.04, 0.22, 0.30)
    warm: tuple[float, float, float, float, float] = (0.42, 0.18, 0.06, 0.16, 0.18)
    rich: tuple[float, float, float, float, float] = (0.55, 0.18, 0.08, 0.14, 0.05)

    #: Interaction-count boundaries between the four regimes.
    sparse_threshold: int = 1
    warm_threshold: int = 5
    rich_threshold: int = 20

    def weights_for(self, interaction_count: int) -> tuple[float, ...]:
        if interaction_count < self.sparse_threshold:
            return self.cold
        if interaction_count < self.warm_threshold:
            return self.sparse
        if interaction_count < self.rich_threshold:
            return self.warm
        return self.rich


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """Stage-1 retrieval budget per source (ADR-001)."""

    total: int = 300
    collaborative: int = 150
    content: int = 100
    covisitation: int = 80
    frequently_bought: int = 40
    category_affinity: int = 60
    brand_affinity: int = 40
    trending: int = 40
    popularity: int = 40
    recently_viewed: int = 20
    #: Fraction of slots reserved for cold items, so a new product can earn
    #: collaborative signal instead of being locked out by the popularity
    #: feedback loop (R-06).
    exploration_fraction: float = 0.05


@dataclass(frozen=True, slots=True)
class RankerConfig:
    """LightGBM LambdaRank settings (ADR-005)."""

    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (5, 10, 20)
    num_leaves: int = 63
    learning_rate: float = 0.06
    n_estimators: int = 400
    min_child_samples: int = 30
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 40
    random_state: int = 42
    n_jobs: int = -1

    #: Graded relevance labels. LambdaRank uses gains of 2^label - 1, so the
    #: spacing here decides how much harder the model works to lift a purchase
    #: above a click.
    label_purchase: int = 3
    label_cart: int = 2
    label_click: int = 1
    label_none: int = 0


@dataclass(frozen=True, slots=True)
class DiversityConfig:
    """Post-ranking re-rank (ADR-014)."""

    enabled: bool = True
    #: MMR trade-off: 1.0 is pure relevance, 0.0 is pure diversity.
    mmr_lambda: float = 0.72
    max_per_category: int = 4
    max_per_brand: int = 3
    #: Exponent on the inverse-popularity discount. 0 disables it.
    popularity_discount: float = 0.15


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    k_values: tuple[int, ...] = (5, 10, 20)
    primary_k: int = 10
    #: Segment boundaries for history-depth reporting (R-05). Aggregate metrics
    #: hide the fact that most real traffic is sparse.
    history_segments: tuple[tuple[str, int, int], ...] = (
        ("cold", 0, 1),
        ("sparse", 1, 5),
        ("warm", 5, 20),
        ("rich", 20, 10**9),
    )
    max_eval_users: int | None = 4000
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class RecsysConfig:
    """Root configuration."""

    seed: int = 42
    data_dir: Path = DEFAULT_DATA_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    as_of: dt.datetime | None = None

    split: SplitConfig = field(default_factory=SplitConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    popularity: PopularityConfig = field(default_factory=PopularityConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    collaborative: CollaborativeConfig = field(default_factory=CollaborativeConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe view for MLflow parameter logging."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dt.datetime):
                return value.isoformat()
            if isinstance(value, tuple):
                return list(value)
            return value

        raw = asdict(self)
        flat: dict[str, Any] = {}
        for section, value in raw.items():
            if isinstance(value, dict):
                for key, inner in value.items():
                    flat[f"{section}.{key}"] = convert(inner)
            else:
                flat[section] = convert(value)
        return flat


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_DATA_DIR",
    "REPO_ROOT",
    "CandidateConfig",
    "CollaborativeConfig",
    "ContentConfig",
    "DiversityConfig",
    "EvaluationConfig",
    "HybridConfig",
    "PopularityConfig",
    "RankerConfig",
    "RecsysConfig",
    "SplitConfig",
    "WeightingConfig",
]
