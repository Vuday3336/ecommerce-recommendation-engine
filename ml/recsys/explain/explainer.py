"""Recommendation explanations (FR-11).

Every recommendation carries a reason. This is not decoration: an unexplained
recommendation is unauditable, and "why did we show this?" is the first question
asked when a rail looks wrong.

Two layers, deliberately:

1. **Rule-based reasons**, derived from *which candidate source retrieved the
   item* and *which component dominated its blended score*. These are cheap
   (microseconds), always available, and phrased for a shopper.
2. **SHAP attributions** for the learned ranker, computed offline or on demand
   for the admin dashboard. These are expensive and are phrased for an analyst.

Keeping them separate matters. A shopper wants "because you bought running
shoes"; an analyst debugging a bad recommendation wants "`category_affinity`
contributed +0.31 to this score". Rendering the second to a shopper is
unhelpful, and offering only the first makes the model unauditable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from recsys.inference.engine import CatalogueIndex


@dataclass(slots=True)
class Explanation:
    """A human-readable reason plus machine-readable evidence."""

    text: str
    reason_code: str
    #: Structured support: anchor products, categories, component scores.
    #: Stored as `recommendations.explanation_evidence` so a decision can be
    #: reconstructed long after the model that made it has been replaced.
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 4),
        }


#: Templates keyed by reason code. Ordered by how specific and convincing they
#: are: a concrete anchor product beats a category, which beats "popular".
TEMPLATES: dict[str, str] = {
    "frequently_bought": "Frequently bought with {anchor}",
    "also_viewed": "Customers who viewed {anchor} also viewed this",
    "similar_to_anchor": "Similar to {anchor}",
    "similar_to_recent": "Because you viewed {anchor}",
    "category_affinity": "You often shop {category}",
    "brand_affinity": "You like {brand}",
    "collaborative": "Popular with shoppers who like what you like",
    "trending": "Trending in {category} right now",
    "popular_in_category": "Popular in {category}",
    "recently_viewed": "You viewed this recently",
    "exploration": "New in {category}",
    "generic": "Recommended for you",
}

#: Which reason a candidate source implies, and how convincing it is. When
#: several sources retrieved the same item, the highest-priority one supplies
#: the wording - the strongest true statement, not the first one found.
SOURCE_REASONS: dict[str, tuple[str, int]] = {
    "frequently_bought_together": ("frequently_bought", 100),
    "covisitation": ("also_viewed", 90),
    "content": ("similar_to_recent", 80),
    "collaborative_als": ("collaborative", 70),
    "brand_affinity": ("brand_affinity", 60),
    "category_affinity": ("category_affinity", 50),
    "recently_viewed": ("recently_viewed", 40),
    "trending": ("trending", 30),
    "popularity": ("popular_in_category", 20),
    "exploration": ("exploration", 10),
}

#: Surfaces whose reason is fixed by the surface itself: on a product page the
#: anchor is unambiguous, so the source ordering does not apply.
SURFACE_REASONS: dict[str, str] = {
    "pdp_similar": "similar_to_anchor",
    "pdp_frequently_bought_together": "frequently_bought",
    "pdp_also_viewed": "also_viewed",
    "home_because_you_viewed": "similar_to_recent",
}


class Explainer:
    """Builds shopper-facing reasons from retrieval provenance."""

    def __init__(
        self,
        *,
        catalogue: CatalogueIndex,
        category_affinity: dict[int, dict[int, float]] | None = None,
        brand_affinity: dict[int, dict[int, float]] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.category_affinity = category_affinity or {}
        self.brand_affinity = brand_affinity or {}

    def explain(
        self,
        *,
        product_id: int,
        surface: str,
        sources: list[str],
        components: dict[str, float] | None = None,
        anchor_product_id: int | None = None,
    ) -> Explanation:
        components = components or {}
        product = self.catalogue.get(product_id)

        reason_code = SURFACE_REASONS.get(surface)
        if reason_code is None:
            reason_code = self._best_reason(sources, components)

        category = (
            self.catalogue.category_names.get(product.category_id, "this category")
            if product
            else "this category"
        )
        brand = (
            self.catalogue.brand_names.get(product.brand_id, "this brand")
            if product
            else "this brand"
        )
        anchor = (
            (self.catalogue.get(anchor_product_id).name if self.catalogue.get(anchor_product_id) else "it")
            if anchor_product_id is not None
            else "it"
        )

        text = TEMPLATES.get(reason_code, TEMPLATES["generic"]).format(
            anchor=anchor, category=category, brand=brand
        )

        return Explanation(
            text=text,
            reason_code=reason_code,
            evidence={
                "sources": sources,
                "components": components,
                "anchor_product_id": anchor_product_id,
                "category_id": product.category_id if product else None,
                "brand_id": product.brand_id if product else None,
            },
            confidence=self._confidence(sources, components),
        )

    def _best_reason(self, sources: list[str], components: dict[str, float]) -> str:
        """Pick the most convincing *true* reason.

        Preference goes to the retrieval source, because "frequently bought
        with X" is a concrete claim about this product. The blended-score
        breakdown is the fallback: it says which signal mattered, but not what
        to tell a shopper.
        """
        candidates = [
            SOURCE_REASONS[source] for source in sources if source in SOURCE_REASONS
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]

        blend = {k: v for k, v in components.items() if k != "ranker"}
        if blend:
            dominant = max(blend, key=blend.get)
            return {
                "collaborative": "collaborative",
                "content": "similar_to_recent",
                "trending": "trending",
                "affinity": "category_affinity",
                "popularity": "popular_in_category",
            }.get(dominant, "generic")
        return "generic"

    @staticmethod
    def _confidence(sources: list[str], components: dict[str, float]) -> float:
        """Rough confidence, used to decide whether to show a reason at all.

        Agreement between independent retrieval sources is the strongest cheap
        signal that a recommendation is well-founded, so it drives this. A
        low-confidence item still gets shown - it just falls back to the
        generic wording rather than making a specific claim the evidence does
        not support.
        """
        agreement = min(len(sources) / 3.0, 1.0)
        strength = max(components.values(), default=0.0)
        return round(0.6 * agreement + 0.4 * min(strength, 1.0), 4)


class ShapExplainer:
    """SHAP attributions for the learned ranker - analyst-facing.

    Separate from `Explainer` because it is expensive (a TreeExplainer pass per
    candidate set) and because its output is a feature attribution table, not a
    sentence. Used by the admin dashboard and when debugging a specific bad
    recommendation, never on the shopper hot path.
    """

    def __init__(self, ranker: Any) -> None:
        self.ranker = ranker
        self._explainer: Any | None = None

    def _load(self) -> Any | None:
        if self._explainer is None:
            try:
                import shap

                self._explainer = shap.TreeExplainer(self.ranker.model)
            except ImportError:
                logger.info("shap is not installed; falling back to gain importance")
                return None
            except Exception:
                logger.exception("could not build a SHAP explainer")
                return None
        return self._explainer

    def attribute(self, features: Any, top: int = 10) -> list[dict[str, float]]:
        """Per-feature contribution for one candidate row."""
        explainer = self._load()
        if explainer is None:
            # Global gain importance is a poor substitute for per-prediction
            # attribution, but it is honest about what it is and keeps the
            # dashboard working without an optional dependency.
            frame = self.ranker.feature_importance(top)
            return [
                {"feature": row["feature"], "contribution": float(row["gain_share"])}
                for _, row in frame.iterrows()
            ]

        values = explainer.shap_values(features)
        row = values[0] if hasattr(values, "__len__") else values
        pairs = sorted(
            zip(self.ranker.feature_columns, row, strict=True),
            key=lambda item: -abs(item[1]),
        )[:top]
        return [{"feature": name, "contribution": float(value)} for name, value in pairs]


__all__ = [
    "SOURCE_REASONS",
    "SURFACE_REASONS",
    "TEMPLATES",
    "Explainer",
    "Explanation",
    "ShapExplainer",
]
