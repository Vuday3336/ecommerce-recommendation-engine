"""Data-drift detection (FR-18).

Models degrade silently. Nothing errors when the input distribution moves: the
service stays fast, returns 200s, and quietly gets worse. Drift detection is
the only thing that turns that into a signal before a business metric moves
weeks later.

Two complementary statistics, because they answer different questions:

* **PSI (Population Stability Index)** — a binned, symmetric divergence.
  Interpretable, has well-established industry thresholds, and works on both
  numeric and categorical features. This is the number people act on.
* **KS (Kolmogorov-Smirnov)** — a distribution-free test on the empirical CDF.
  More sensitive than PSI to a shift in shape that leaves the bin masses
  similar, and it comes with a p-value.

PSI is used for alerting and KS for corroboration. Relying on the p-value alone
would be a mistake: with hundreds of thousands of rows, a KS test rejects the
null for effects far too small to matter, so significance and importance come
apart. PSI's magnitude thresholds are what keep alerting tied to impact.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Industry-standard PSI bands. Below 0.10 is noise; 0.10-0.25 warrants a look;
#: above 0.25 the population has genuinely moved.
PSI_WARN = 0.10
PSI_ALERT = 0.25
KS_ALERT_P = 0.01

#: Smoothing for empty bins. Without it, a bin present in the reference and
#: absent in the current window makes PSI infinite - a single missing category
#: would fire an alert on its own.
EPSILON = 1e-6

DEFAULT_BINS = 10


class DriftSeverity(StrEnum):
    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"


@dataclass(slots=True)
class FeatureDrift:
    """Drift measured for one feature."""

    feature: str
    psi: float
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    severity: DriftSeverity = DriftSeverity.NONE
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_n: int = 0
    current_n: int = 0

    @property
    def alerting(self) -> bool:
        return self.severity is DriftSeverity.ALERT

    def summary(self) -> str:
        parts = [f"{self.feature}: PSI={self.psi:.4f} [{self.severity.value}]"]
        if self.ks_p_value is not None:
            parts.append(f"KS={self.ks_statistic:.4f} (p={self.ks_p_value:.2e})")
        if self.reference_mean is not None and self.current_mean is not None:
            parts.append(f"mean {self.reference_mean:.4g} -> {self.current_mean:.4g}")
        return "; ".join(parts)


@dataclass(slots=True)
class DriftReport:
    """Drift across every monitored feature."""

    features: list[FeatureDrift] = field(default_factory=list)
    computed_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    reference_window: tuple[str, str] | None = None
    current_window: tuple[str, str] | None = None

    @property
    def alerting(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.severity is DriftSeverity.ALERT]

    @property
    def warning(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.severity is DriftSeverity.WARNING]

    @property
    def should_retrain(self) -> bool:
        """Whether drift justifies kicking off a retrain.

        One alerting feature is not enough. A single feature can move for a
        benign reason - a merchandising change, a new category launch - and
        retraining on every such event means retraining constantly. Two
        alerting features, or one alert plus three warnings, is a population
        shift rather than a local change.
        """
        return len(self.alerting) >= 2 or (
            len(self.alerting) >= 1 and len(self.warning) >= 3
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "computed_at": self.computed_at.isoformat(),
            "reference_window": self.reference_window,
            "current_window": self.current_window,
            "should_retrain": self.should_retrain,
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 6),
                    "ks_statistic": round(f.ks_statistic, 6) if f.ks_statistic else None,
                    "ks_p_value": f.ks_p_value,
                    "severity": f.severity.value,
                    "reference_mean": f.reference_mean,
                    "current_mean": f.current_mean,
                }
                for f in self.features
            ],
        }

    def summary(self) -> str:
        lines = [
            f"Drift report ({len(self.features)} features, "
            f"{len(self.alerting)} alerting, {len(self.warning)} warning)"
        ]
        for feature in sorted(self.features, key=lambda f: -f.psi):
            lines.append(f"  {feature.summary()}")
        lines.append(
            f"  -> retrain recommended: {'yes' if self.should_retrain else 'no'}"
        )
        return "\n".join(lines)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    categorical: bool = False,
) -> float:
    """PSI between a reference and a current sample.

        PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

    Bin edges come from the **reference** quantiles, not from the combined
    data. Recomputing edges each run would move the goalposts with the data and
    could report zero drift for a distribution that had shifted wholesale.
    """
    reference = np.asarray(reference)
    current = np.asarray(current)
    reference = reference[~pd.isna(reference)]
    current = current[~pd.isna(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    if categorical:
        categories = np.union1d(np.unique(reference), np.unique(current))
        reference_share = np.array(
            [(reference == c).sum() / len(reference) for c in categories]
        )
        current_share = np.array(
            [(current == c).sum() / len(current) for c in categories]
        )
    else:
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(reference.astype(float), quantiles))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        reference_counts, _ = np.histogram(reference.astype(float), bins=edges)
        current_counts, _ = np.histogram(current.astype(float), bins=edges)
        reference_share = reference_counts / max(reference_counts.sum(), 1)
        current_share = current_counts / max(current_counts.sum(), 1)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    from scipy import stats

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 2 or len(current) < 2:
        return 0.0, 1.0
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def classify(psi: float, *, warn: float = PSI_WARN, alert: float = PSI_ALERT) -> DriftSeverity:
    if psi >= alert:
        return DriftSeverity.ALERT
    if psi >= warn:
        return DriftSeverity.WARNING
    return DriftSeverity.NONE


class DriftDetector:
    """Compares a current window against a stored reference snapshot.

    The reference is the distribution the production model was *trained* on,
    not last week's data. Comparing against a rolling recent window would let
    slow drift pass unnoticed: each week looks like the last, while the
    cumulative distance from training grows without bound.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        *,
        numeric_features: tuple[str, ...] = (),
        categorical_features: tuple[str, ...] = (),
        psi_warn: float = PSI_WARN,
        psi_alert: float = PSI_ALERT,
        bins: int = DEFAULT_BINS,
    ) -> None:
        self.reference = reference
        self.numeric_features = numeric_features or tuple(
            c for c in reference.select_dtypes("number").columns
        )
        self.categorical_features = categorical_features
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.bins = bins

    def compare(self, current: pd.DataFrame) -> DriftReport:
        results: list[FeatureDrift] = []

        for feature in self.numeric_features:
            if feature not in current.columns or feature not in self.reference.columns:
                continue
            reference_values = self.reference[feature].to_numpy()
            current_values = current[feature].to_numpy()

            psi = population_stability_index(
                reference_values, current_values, bins=self.bins
            )
            statistic, p_value = ks_test(reference_values, current_values)
            results.append(
                FeatureDrift(
                    feature=feature,
                    psi=psi,
                    ks_statistic=statistic,
                    ks_p_value=p_value,
                    severity=classify(psi, warn=self.psi_warn, alert=self.psi_alert),
                    reference_mean=float(np.nanmean(reference_values.astype(float))),
                    current_mean=float(np.nanmean(current_values.astype(float))),
                    reference_n=len(reference_values),
                    current_n=len(current_values),
                )
            )

        for feature in self.categorical_features:
            if feature not in current.columns or feature not in self.reference.columns:
                continue
            psi = population_stability_index(
                self.reference[feature].to_numpy(),
                current[feature].to_numpy(),
                categorical=True,
            )
            results.append(
                FeatureDrift(
                    feature=feature,
                    psi=psi,
                    severity=classify(psi, warn=self.psi_warn, alert=self.psi_alert),
                    reference_n=len(self.reference),
                    current_n=len(current),
                )
            )

        return DriftReport(features=results)

    def export_metrics(self, report: DriftReport) -> None:
        """Publish the report to Prometheus so Grafana can alert on it."""
        try:
            from app.core import metrics
        except ImportError:
            return
        for feature in report.features:
            metrics.drift_score.labels(feature=feature.feature, method="psi").set(feature.psi)
            if feature.ks_statistic is not None:
                metrics.drift_score.labels(feature=feature.feature, method="ks").set(
                    feature.ks_statistic
                )
            if feature.alerting:
                metrics.drift_alerts_total.labels(feature=feature.feature).inc()


#: Features worth monitoring, and why each one is on the list.
MONITORED_USER_FEATURES: tuple[str, ...] = (
    "total_events",          # overall engagement level
    "total_purchases",       # conversion behaviour
    "avg_purchase_price",    # basket value, catches pricing and mix changes
    "price_sensitivity",     # audience composition
    "days_since_last_event", # recency profile, catches a traffic-source change
    "category_concentration",# how focused shoppers are
    "view_to_purchase_rate", # funnel efficiency
)

MONITORED_PRODUCT_FEATURES: tuple[str, ...] = (
    "price",
    "popularity_score",
    "conversion_rate",
    "view_count_30d",
    "quality_score",
    "days_since_release",
)


__all__ = [
    "DEFAULT_BINS",
    "KS_ALERT_P",
    "MONITORED_PRODUCT_FEATURES",
    "MONITORED_USER_FEATURES",
    "PSI_ALERT",
    "PSI_WARN",
    "DriftDetector",
    "DriftReport",
    "DriftSeverity",
    "FeatureDrift",
    "classify",
    "ks_test",
    "population_stability_index",
]
