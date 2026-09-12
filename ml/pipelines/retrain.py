"""Automated retraining pipeline (FR-19).

    python ml/pipelines/retrain.py --check-drift     # decide only
    python ml/pipelines/retrain.py --force           # retrain regardless
    python ml/pipelines/retrain.py                   # drift-triggered

The loop the brief asks for, closed:

    drift detected (or scheduled)
        -> retrain
            -> evaluate against the incumbent
                -> promotion gate
                    -> promote and swap artefacts, or reject and keep serving

**The gate is the point.** A retrained model does not become production because
it finished training. It is promoted only if it beats the incumbent on NDCG@10
and hit-rate@10 *and* does not collapse catalogue coverage. Automatic
retraining without a gate is worse than no automatic retraining: it replaces a
known-good model with an unvalidated one on a schedule.

Artefacts are swapped atomically - written to a temporary directory, then
renamed - so a crash mid-write cannot leave the serving path pointing at a
half-written model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ml"))

from recsys.config.settings import RecsysConfig  # noqa: E402
from recsys.drift.detector import (  # noqa: E402
    MONITORED_PRODUCT_FEATURES,
    MONITORED_USER_FEATURES,
    DriftDetector,
    DriftReport,
)
from recsys.registry.promotion import GateDecision, evaluate_gate  # noqa: E402

from pipelines.train import TrainingPipeline, save_artifacts  # noqa: E402

logger = logging.getLogger("recsys.retrain")

#: Metric the gate and the reporting treat as primary.
PRIMARY_MODEL = "two-stage-ranker"


@dataclass
class RetrainOutcome:
    """What one retraining run decided and did."""

    triggered: bool
    reason: str
    drift: DriftReport | None = None
    decision: GateDecision | None = None
    promoted: bool = False
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    incumbent_metrics: dict[str, float] = field(default_factory=dict)
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "promoted": self.promoted,
            "started_at": self.started_at.isoformat(),
            "drift": self.drift.to_dict() if self.drift else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "candidate_metrics": self.candidate_metrics,
            "incumbent_metrics": self.incumbent_metrics,
        }


def load_incumbent_metrics(artifact_dir: Path) -> dict[str, float]:
    """Metrics of the model currently serving.

    Read from the artefact directory rather than from MLflow, deliberately:
    the gate must be able to run when the tracking server is down, and the
    artefact directory *is* what is serving.
    """
    path = Path(artifact_dir) / "evaluation.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    row = frame[frame["model"] == PRIMARY_MODEL]
    if row.empty:
        row = frame.nlargest(1, "ndcg@10") if "ndcg@10" in frame.columns else frame.head(1)
    if row.empty:
        return {}
    return {
        column: float(row.iloc[0][column])
        for column in row.columns
        if column not in {"model", "users", "seconds"} and pd.notna(row.iloc[0][column])
    }


def check_drift(artifact_dir: Path, config: RecsysConfig) -> DriftReport:
    """Compare current feature distributions with the training reference.

    The reference is the feature snapshot the serving model was trained on;
    the current window is features rebuilt from the most recent data. Comparing
    against a rolling recent window instead would hide slow drift entirely.
    """
    artifact_dir = Path(artifact_dir)
    reference_users = artifact_dir / "user_features.parquet"
    reference_products = artifact_dir / "product_features.parquet"
    if not reference_users.exists():
        logger.warning("no reference features at %s; cannot assess drift", artifact_dir)
        return DriftReport()

    pipeline = TrainingPipeline(config)
    dataset = pipeline.extract()
    split = pipeline.split(dataset)
    _, _, current_users, current_products, _, _ = pipeline.build_features(dataset, split)

    report = DriftReport(
        reference_window=(
            "training snapshot",
            str(pd.read_parquet(reference_users).shape[0]) + " users",
        ),
        current_window=(split.train_end.isoformat(), split.test.iloc[-1]["occurred_at"].isoformat())
        if len(split.test)
        else None,
    )

    user_detector = DriftDetector(
        pd.read_parquet(reference_users), numeric_features=MONITORED_USER_FEATURES
    )
    report.features.extend(user_detector.compare(current_users).features)

    if reference_products.exists():
        product_detector = DriftDetector(
            pd.read_parquet(reference_products),
            numeric_features=MONITORED_PRODUCT_FEATURES,
        )
        report.features.extend(product_detector.compare(current_products).features)

    return report


def swap_artifacts(staging: Path, production: Path) -> None:
    """Atomically replace the serving artefacts.

    Write to staging, move the old directory aside, move staging in, then drop
    the old one. A plain overwrite would leave the serving path reading a
    directory that is half old and half new if the process died mid-copy.
    """
    staging, production = Path(staging), Path(production)
    backup = production.with_name(production.name + ".previous")

    if backup.exists():
        shutil.rmtree(backup)
    if production.exists():
        production.rename(backup)
    staging.rename(production)
    logger.info("artefacts swapped: %s -> %s (previous kept at %s)", staging, production, backup)


def run(
    config: RecsysConfig,
    *,
    artifact_dir: Path,
    force: bool = False,
    check_only: bool = False,
    eval_users: int = 1500,
    ranker_users: int = 2500,
) -> RetrainOutcome:
    """Assess drift, retrain if warranted, and promote only if the gate passes."""
    artifact_dir = Path(artifact_dir)

    drift = None
    if not force:
        drift = check_drift(artifact_dir, config)
        logger.info("\n%s", drift.summary())
        if not drift.should_retrain:
            return RetrainOutcome(
                triggered=False,
                reason=(
                    f"drift below the retraining threshold "
                    f"({len(drift.alerting)} alerting, {len(drift.warning)} warning)"
                ),
                drift=drift,
            )
        reason = f"drift detected on {len(drift.alerting)} features"
    else:
        reason = "forced by the operator"

    if check_only:
        return RetrainOutcome(triggered=True, reason=reason + " (check only)", drift=drift)

    logger.info("retraining: %s", reason)
    incumbent = load_incumbent_metrics(artifact_dir)

    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run(
        train_ranker=True, train_bpr=False, eval_users=eval_users, ranker_users=ranker_users
    )

    candidate_result = next(
        (r for r in artifacts.evaluations if r.model == PRIMARY_MODEL), None
    ) or max(artifacts.evaluations, key=lambda r: r.metrics.get("ndcg@10", 0.0))
    candidate = candidate_result.metrics

    decision = evaluate_gate(candidate, incumbent or None)
    logger.info("\n%s", decision.summary())

    outcome = RetrainOutcome(
        triggered=True,
        reason=reason,
        drift=drift,
        decision=decision,
        candidate_metrics=candidate,
        incumbent_metrics=incumbent,
    )

    if decision.should_promote:
        staging = artifact_dir.with_name(artifact_dir.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        save_artifacts(artifacts, staging)
        (staging / "promotion.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        swap_artifacts(staging, artifact_dir)
        outcome.promoted = True
        logger.info("promoted: the new model is now serving")
    else:
        # The candidate is kept for inspection but does not serve. Discarding
        # it would make "why was this rejected?" unanswerable.
        rejected = artifact_dir.with_name(artifact_dir.name + ".rejected")
        if rejected.exists():
            shutil.rmtree(rejected)
        save_artifacts(artifacts, rejected)
        (rejected / "promotion.json").write_text(
            json.dumps(outcome.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.warning(
            "rejected: the incumbent keeps serving. Candidate kept at %s", rejected
        )

    return outcome


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "ml" / "artifacts")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="retrain regardless of drift")
    parser.add_argument("--check-drift", action="store_true", help="assess drift and stop")
    parser.add_argument("--eval-users", type=int, default=1500)
    parser.add_argument("--ranker-users", type=int, default=2500)
    parser.add_argument("--out", type=Path, default=None, help="write the outcome as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    config = RecsysConfig(data_dir=args.data) if args.data else RecsysConfig()
    outcome = run(
        config,
        artifact_dir=args.artifacts,
        force=args.force,
        check_only=args.check_drift,
        eval_users=args.eval_users,
        ranker_users=args.ranker_users,
    )

    print("\n" + "=" * 78)
    print("RETRAINING OUTCOME")
    print("=" * 78)
    print(f"  triggered : {outcome.triggered}")
    print(f"  reason    : {outcome.reason}")
    if outcome.decision:
        print(f"  gate      : {outcome.decision.outcome.value}")
        for line in outcome.decision.passed:
            print(f"    PASS  {line}")
        for line in outcome.decision.failed:
            print(f"    FAIL  {line}")
    print(f"  promoted  : {outcome.promoted}")

    if args.out:
        args.out.write_text(
            json.dumps(outcome.to_dict(), indent=2, default=str), encoding="utf-8"
        )

    # Non-zero when a retrain ran and the gate rejected it, so a scheduler can
    # alert on "we tried and the model got worse" rather than treating it as a
    # normal no-op.
    if outcome.triggered and outcome.decision and not outcome.promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
