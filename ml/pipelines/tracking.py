"""MLflow tracking and model registry (ADR-010).

Logs parameters, metrics, artefacts and the dataset fingerprint for every
training run, then registers the ranker as a model version. Promotion to
Production goes through the gate in `recsys.registry.promotion` - never
automatically, and never without the comparison being recorded.

MLflow is the system of record for lineage. It is deliberately *not* on the
serving path: the API resolves models from a local artefact directory and the
`model_versions` table, so MLflow being down can prevent a model *change* but
never a recommendation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from pipelines.train import TrainedArtifacts

DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "recommendation-engine")
DEFAULT_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "recsys-ranker")

#: Metrics copied from the winning model onto the run itself, so runs can be
#: sorted and compared in the MLflow UI without opening the artefact table.
HEADLINE_METRICS: tuple[str, ...] = (
    "ndcg@10",
    "recall@10",
    "precision@10",
    "map@10",
    "hit_rate@10",
    "mrr",
    "coverage",
    "diversity",
    "novelty",
    "personalisation",
)


def dataset_fingerprint(artifacts: TrainedArtifacts) -> str:
    """Stable hash of the data a run trained on.

    Without it, two runs with identical parameters and different metrics are
    inexplicable. With it, "the data changed" is a checkable claim rather than
    a guess - which is exactly the question that comes up when a retrained
    model unexpectedly regresses.
    """
    summary = artifacts.split.summary()
    payload = json.dumps(
        {
            "train_rows": summary["train_rows"],
            "validation_rows": summary["validation_rows"],
            "test_rows": summary["test_rows"],
            "train_end": summary["train_end"],
            "validation_end": summary["validation_end"],
            "train_users": summary["train_users"],
            "train_items": summary["train_items"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def log_training_run(
    artifacts: TrainedArtifacts,
    artifact_dir: Path,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    model_name: str = DEFAULT_MODEL_NAME,
    register: bool = True,
    tracking_uri: str | None = None,
) -> str:
    """Log one training run and register the ranker. Returns the run id."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        # A local SQLite backend, so the pipeline runs with no tracking server.
        # Not a plain `file:` store: MLflow 3 put the filesystem backend into
        # maintenance mode and raises on it, and SQLite is the documented
        # local replacement. It also supports the model registry, which the
        # file store never did - so promotion works locally too.
        store = (artifact_dir.parent / "mlflow.db").resolve()
        store.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
        mlflow.set_registry_uri(f"sqlite:///{store.as_posix()}")

    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(_flatten_params(artifacts.config.to_dict()))
        mlflow.set_tags(
            {
                "dataset_fingerprint": dataset_fingerprint(artifacts),
                "train_end": artifacts.split.train_end.isoformat(),
                "models_trained": ",".join(sorted(artifacts.models)),
                "weight_calibration": "derived" if artifacts.calibration.calibrated else "prior",
            }
        )

        for event_type, weight in artifacts.calibration.weights.items():
            mlflow.log_metric(f"weight/{event_type}", float(weight))

        for result in artifacts.evaluations:
            for metric, value in result.metrics.items():
                # MLflow metric names allow a limited character set; '@' is not
                # in it, so `ndcg@10` becomes `ndcg_at_10`.
                mlflow.log_metric(
                    f"{_safe(result.model)}/{_safe(metric)}", float(value)
                )
            for segment, values in result.per_segment.items():
                for metric, value in values.items():
                    mlflow.log_metric(
                        f"{_safe(result.model)}/{segment}/{_safe(metric)}", float(value)
                    )

        best = _best_model(artifacts)
        if best is not None:
            for metric in HEADLINE_METRICS:
                if metric in best.metrics:
                    mlflow.log_metric(f"best/{_safe(metric)}", float(best.metrics[metric]))
            mlflow.set_tag("best_model", best.model)

        for name in (
            "evaluation.csv",
            "evaluation_by_segment.csv",
            "weight_calibration.csv",
            "metrics.json",
            "config.json",
        ):
            path = artifact_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))

        ranker = artifacts.models.get("ranker")
        if ranker is not None and register:
            _register_ranker(mlflow, ranker, model_name, artifact_dir)

        return run.info.run_id


def _register_ranker(mlflow: Any, ranker: Any, model_name: str, artifact_dir: Path) -> None:
    """Log and register the LightGBM ranker."""
    try:
        import mlflow.lightgbm

        mlflow.lightgbm.log_model(
            ranker.model.booster_,
            name="ranker",
            registered_model_name=model_name,
        )
    except Exception:
        # A registry that refuses the flavour must not fail the training run;
        # the artefact on disk is what the serving path actually loads.
        logger.exception("failed to register the ranker; the artefact is still on disk")
        path = artifact_dir / "ranker.joblib"
        if path.exists():
            mlflow.log_artifact(str(path))


def _best_model(artifacts: TrainedArtifacts):
    scored = [r for r in artifacts.evaluations if "ndcg@10" in r.metrics]
    return max(scored, key=lambda r: r.metrics["ndcg@10"]) if scored else None


def _flatten_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow parameters must be short scalars."""
    out: dict[str, str] = {}
    for key, value in params.items():
        text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        out[_safe(key)[:250]] = text[:500]
    return out


def _safe(name: str) -> str:
    return (
        name.replace("@", "_at_")
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("-", "_")
    )


def resolve_production_model(
    model_name: str = DEFAULT_MODEL_NAME, *, tracking_uri: str | None = None
) -> dict[str, Any] | None:
    """Look up the version currently aliased to Production.

    Returns None rather than raising when the registry is unreachable: the
    caller falls back to the last-known-good version pinned in
    `model_versions`, which is what keeps MLflow off the serving critical path.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        production = [v for v in versions if getattr(v, "current_stage", "") == "Production"]
        chosen = production[0] if production else (versions[0] if versions else None)
        if chosen is None:
            return None
        return {
            "name": chosen.name,
            "version": chosen.version,
            "run_id": chosen.run_id,
            "source": chosen.source,
            "stage": getattr(chosen, "current_stage", "None"),
        }
    except Exception:
        logger.warning("MLflow registry unavailable; using the pinned local model")
        return None


__all__ = [
    "DEFAULT_EXPERIMENT",
    "DEFAULT_MODEL_NAME",
    "HEADLINE_METRICS",
    "dataset_fingerprint",
    "log_training_run",
    "resolve_production_model",
]
