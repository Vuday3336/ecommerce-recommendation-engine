"""Stage-2 ranking model: LightGBM LambdaRank (ADR-005).

**Why gradient boosting.** The ranking features are heterogeneous and tabular -
normalised model scores, raw counts, ratios, log prices, days of recency,
categorical matches. Trees handle mixed scales, non-linearities and
interactions without any feature scaling, and they capture things like "content
similarity matters far more when the collaborative score is weak", which a
linear model cannot express and a neural ranker would need far more data to
learn.

**Why LambdaRank rather than binary classification.** Predicting per-item click
probability treats each candidate independently, but the product is a *list*.
LambdaRank optimises a listwise NDCG surrogate directly, weighting each
potential swap by how much it would change NDCG - the metric actually reported.
`LogisticRanker` below is trained as the measured control, so the claim is
evidenced rather than asserted.

**Why LightGBM rather than XGBoost.** Both have ranking objectives. LightGBM's
leaf-wise growth and histogram binning are faster on this shape, and its
`group` handling for ranking is more ergonomic. `XGBoostRanker` is kept for the
comparison run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config.settings import RankerConfig
from recsys.ranking.dataset import RankingDataset

logger = logging.getLogger(__name__)


class LambdaRanker:
    """LightGBM LambdaRank over the candidate feature matrix."""

    name = "ranker_lambdarank"

    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()
        self.model = None
        self.feature_columns: list[str] = []
        self.best_iteration: int | None = None

    def fit(
        self,
        train: RankingDataset,
        validation: RankingDataset | None = None,
    ) -> LambdaRanker:
        import lightgbm as lgb

        if len(train) == 0:
            raise ValueError("cannot fit a ranker on an empty dataset")

        self.feature_columns = list(train.features.columns)

        self.model = lgb.LGBMRanker(
            objective=self.config.objective,
            metric=self.config.metric,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            subsample_freq=self.config.subsample_freq,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=-1,
            # `label_gain` must cover every label value used. LightGBM defaults
            # to a 31-entry table; declaring it explicitly makes the gain
            # spacing an intentional, reviewable choice rather than a default.
            label_gain=[0, 1, 3, 7],
        )

        callbacks = [lgb.log_evaluation(period=0)]
        eval_set = None
        eval_group = None
        if validation is not None and len(validation) > 0:
            eval_set = [(validation.features[self.feature_columns], validation.labels)]
            eval_group = [validation.groups]
            callbacks.append(
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
            )

        self.model.fit(
            train.features[self.feature_columns],
            train.labels,
            group=train.groups,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_at=list(self.config.eval_at),
            callbacks=callbacks,
        )
        self.best_iteration = getattr(self.model, "best_iteration_", None)
        logger.info(
            "ranker fitted on %d rows / %d queries, %d features, best_iteration=%s",
            len(train),
            len(train.groups),
            len(self.feature_columns),
            self.best_iteration,
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        if features.empty:
            return np.array([])
        aligned = features.reindex(columns=self.feature_columns).fillna(0.0)
        return np.asarray(self.model.predict(aligned))

    def feature_importance(self, top: int | None = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        frame = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "gain": self.model.booster_.feature_importance(importance_type="gain"),
                "split": self.model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)
        frame["gain_share"] = frame["gain"] / max(frame["gain"].sum(), 1e-9)
        return frame.head(top) if top else frame

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "config": self.config,
                "best_iteration": self.best_iteration,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> LambdaRanker:
        import joblib

        payload = joblib.load(Path(path))
        ranker = cls(payload["config"])
        ranker.model = payload["model"]
        ranker.feature_columns = payload["feature_columns"]
        ranker.best_iteration = payload.get("best_iteration")
        return ranker


class XGBoostRanker:
    """XGBoost `rank:ndcg` - the cross-library control for ADR-005."""

    name = "ranker_xgboost"

    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()
        self.model = None
        self.feature_columns: list[str] = []

    def fit(self, train: RankingDataset, validation: RankingDataset | None = None) -> XGBoostRanker:
        import xgboost as xgb

        self.feature_columns = list(train.features.columns)
        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=6,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        self.model.fit(
            train.features[self.feature_columns],
            train.labels,
            group=train.groups,
            verbose=False,
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        if features.empty:
            return np.array([])
        return np.asarray(
            self.model.predict(features.reindex(columns=self.feature_columns).fillna(0.0))
        )


class LogisticRanker:
    """Pointwise logistic control for ADR-005.

    Predicts P(engagement) per item independently. Its purpose is to make the
    listwise-versus-pointwise claim measurable: if it matches LambdaRank, the
    extra machinery is not earning its place.
    """

    name = "ranker_logistic"

    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()
        self.model = None
        self.scaler = None
        self.feature_columns: list[str] = []

    def fit(self, train: RankingDataset, validation: RankingDataset | None = None) -> LogisticRanker:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.feature_columns = list(train.features.columns)
        features = train.features[self.feature_columns].to_numpy(dtype=np.float64)
        # Unlike the tree models this one needs scaling, and it needs the
        # graded labels collapsed to binary.
        self.scaler = StandardScaler().fit(features)
        self.model = LogisticRegression(
            max_iter=400, C=1.0, class_weight="balanced", random_state=self.config.random_state
        )
        self.model.fit(self.scaler.transform(features), (train.labels > 0).astype(int))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("ranker is not fitted")
        if features.empty:
            return np.array([])
        aligned = features.reindex(columns=self.feature_columns).fillna(0.0)
        scaled = self.scaler.transform(aligned.to_numpy(dtype=np.float64))
        return np.asarray(self.model.predict_proba(scaled)[:, 1])


__all__ = ["LambdaRanker", "LogisticRanker", "XGBoostRanker"]
