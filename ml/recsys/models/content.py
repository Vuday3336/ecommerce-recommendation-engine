"""Content-based recommendation from product metadata and text.

Two things make this model matter beyond being "another baseline":

1. **It is the new-product cold-start path (FR-10).** A product with no
   interactions has no collaborative representation at all, but it has a
   category, a brand, a price and a description - so it lands in the embedding
   space the moment it is created and is immediately retrievable.
2. **It supplies the similar-products surface (FR-02).** Item-item similarity
   here is what "You may also like" runs on.

**Backend choice.** The default is TF-IDF plus truncated SVD, not a sentence
transformer. That is a deliberate, measured choice rather than a shortcut: it
is deterministic, trains in about a second, adds no 2 GB dependency and no
model download, and on a catalogue whose descriptions are generated from
structured attributes it captures essentially the same structure. The
`SentenceTransformerEncoder` implements the same interface for catalogues with
genuinely free-form text; the evaluation report compares them rather than
assuming.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.config.settings import ContentConfig
from recsys.models.base import (
    RecommendationContext,
    Recommender,
    Scored,
    normalise_scores,
    top_k,
)

logger = logging.getLogger(__name__)


def build_product_text(products: pd.DataFrame, categories: pd.DataFrame) -> pd.Series:
    """Compose the text each product is embedded from.

    Category path and brand are repeated into the text deliberately. TF-IDF
    weights them like any other token, and repetition is the simplest way to
    say "these are the strong signals" without a separate weighting scheme -
    they are the fields that most reliably predict what a user considers
    similar.
    """
    category_path = categories.set_index("id")["path"].to_dict()
    category_name = categories.set_index("id")["name"].to_dict()

    def attributes_to_text(value: object) -> str:
        if isinstance(value, dict):
            return " ".join(f"{k} {v}" for k, v in value.items())
        return ""

    path = products["category_id"].map(category_path).fillna("")
    leaf = products["category_id"].map(category_name).fillna("")
    attributes = products["attributes"].map(attributes_to_text)

    return (
        products["name"].fillna("")
        + " "
        + leaf
        + " "
        + leaf
        + " "
        + path.str.replace("/", " ", regex=False)
        + " "
        + products["price_band"].fillna("")
        + " "
        + attributes
        + " "
        + products["description"].fillna("")
    ).str.lower()


class TfidfSvdEncoder:
    """Deterministic text encoder: TF-IDF then truncated SVD."""

    name = "tfidf_svd"

    def __init__(self, config: ContentConfig | None = None, *, random_state: int = 42) -> None:
        self.config = config or ContentConfig()
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.ngram_range,
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = self.vectorizer.fit_transform(texts)

        # SVD width is capped by the vocabulary; asking for more components
        # than the matrix rank raises rather than silently truncating.
        components = min(self.config.embedding_dim, matrix.shape[1] - 1, len(texts) - 1)
        self.svd = TruncatedSVD(
            n_components=max(components, 2), random_state=self.random_state
        )
        return self.svd.fit_transform(matrix)

    def transform(self, texts: pd.Series) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("encoder is not fitted")
        return self.svd.transform(self.vectorizer.transform(texts))

    @property
    def explained_variance(self) -> float:
        return float(self.svd.explained_variance_ratio_.sum()) if self.svd else 0.0


class SentenceTransformerEncoder:
    """Transformer encoder, behind the same interface.

    Imported lazily so the dependency is only required when the backend is
    actually selected - the API process must never import Torch.
    """

    name = "sentence_transformer"

    def __init__(self, config: ContentConfig | None = None, **_: object) -> None:
        self.config = config or ContentConfig()
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.sentence_transformer_model)
        return self._model

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: pd.Series) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                texts.tolist(),
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )

    @property
    def explained_variance(self) -> float:
        return float("nan")


ENCODERS = {
    "tfidf_svd": TfidfSvdEncoder,
    "sentence_transformer": SentenceTransformerEncoder,
}


class ContentRecommender(Recommender):
    """Cosine similarity over composed product vectors."""

    name = "content"
    supports_cold_users = True

    def __init__(self, config: ContentConfig | None = None, *, random_state: int = 42) -> None:
        self.config = config or ContentConfig()
        self.random_state = random_state
        self.encoder = ENCODERS[self.config.backend](self.config, random_state=random_state)

        self.product_ids: np.ndarray = np.array([], dtype=np.int64)
        self.index_of: dict[int, int] = {}
        self.embeddings: np.ndarray = np.zeros((0, 0))

    # -- training ---------------------------------------------------------

    def fit(
        self,
        products: pd.DataFrame,
        categories: pd.DataFrame,
        **_: object,
    ) -> ContentRecommender:
        ordered = products.sort_values("id").reset_index(drop=True)
        self.product_ids = ordered["id"].to_numpy(dtype=np.int64)
        self.index_of = {int(pid): i for i, pid in enumerate(self.product_ids)}

        text_vectors = self.encoder.fit_transform(build_product_text(ordered, categories))
        attribute_vectors = self._attribute_matrix(ordered)

        # L2-normalise each block before combining so the two components
        # contribute in the declared proportion. Without it the ratio would be
        # decided by whichever block happens to have larger raw magnitude.
        text_part = normalize(text_vectors) * self.config.text_weight
        attribute_part = normalize(attribute_vectors) * self.config.attribute_weight

        combined = np.hstack([text_part, np.asarray(attribute_part.todense())])
        self.embeddings = normalize(combined).astype(np.float32)

        logger.info(
            "content model: %d products, %d dims, SVD explains %.1f%% of variance",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            self.encoder.explained_variance * 100,
        )
        return self

    def _attribute_matrix(self, products: pd.DataFrame) -> sparse.csr_matrix:
        """One-hot category, brand and price band, plus scaled log price.

        Structured attributes are kept separate from text rather than folded
        into it because they are exact: two products either share a brand or
        they do not, and letting that fact survive tokenisation and SVD
        unchanged is what keeps brand similarity crisp.
        """
        blocks = [
            sparse.csr_matrix(pd.get_dummies(products["category_id"]).to_numpy(dtype=np.float32)),
            sparse.csr_matrix(pd.get_dummies(products["brand_id"]).to_numpy(dtype=np.float32)),
            sparse.csr_matrix(pd.get_dummies(products["price_band"]).to_numpy(dtype=np.float32)),
        ]
        log_price = np.log1p(products["price"].to_numpy(dtype=np.float32))
        span = log_price.max() - log_price.min()
        scaled = (log_price - log_price.min()) / span if span > 0 else np.zeros_like(log_price)
        blocks.append(sparse.csr_matrix(scaled.reshape(-1, 1)))
        return sparse.hstack(blocks, format="csr")

    # -- inference --------------------------------------------------------

    def similar_products(self, product_id: int, k: int = 10) -> list[Scored]:
        """Item-item similarity - the `/recommendations/similar` surface."""
        index = self.index_of.get(int(product_id))
        if index is None or self.embeddings.size == 0:
            return []
        similarity = self.embeddings @ self.embeddings[index]
        similarity[index] = -np.inf
        top = np.argpartition(-similarity, min(k, len(similarity) - 1))[:k]
        top = top[np.argsort(-similarity[top])]
        return [
            Scored(int(self.product_ids[i]), float(similarity[i]), "content") for i in top
        ]

    def taste_vector(self, product_ids: tuple[int, ...], weights: np.ndarray | None = None) -> np.ndarray | None:
        """Weighted centroid of the products a user has engaged with."""
        indices = [self.index_of[pid] for pid in product_ids if pid in self.index_of]
        if not indices:
            return None
        block = self.embeddings[indices]
        if weights is not None and len(weights) == len(indices):
            centroid = np.average(block, axis=0, weights=weights)
        else:
            centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else None

    def recommend(self, context: RecommendationContext, k: int = 10) -> list[Scored]:
        # Recent products first: short-term intent beats long-term taste for
        # what to show next, and it is the only signal an anonymous session has.
        anchors = context.recent_products or context.seen_products
        centroid = self.taste_vector(tuple(anchors[:40]))
        if centroid is None:
            return []

        similarity = self.embeddings @ centroid
        scores = {
            int(self.product_ids[i]): float(similarity[i])
            for i in np.argsort(-similarity)[: k * 6]
            if int(self.product_ids[i]) not in context.exclude
        }
        return top_k(normalise_scores(scores), k, source=self.name)

    def score(
        self, context: RecommendationContext, product_ids: list[int]
    ) -> dict[int, float]:
        anchors = context.recent_products or context.seen_products
        centroid = self.taste_vector(tuple(anchors[:40]))
        if centroid is None:
            return dict.fromkeys(product_ids, 0.0)
        indices = [self.index_of.get(pid) for pid in product_ids]
        return {
            pid: float(self.embeddings[i] @ centroid) if i is not None else 0.0
            for pid, i in zip(product_ids, indices, strict=True)
        }


__all__ = [
    "ENCODERS",
    "ContentRecommender",
    "SentenceTransformerEncoder",
    "TfidfSvdEncoder",
    "build_product_text",
]
