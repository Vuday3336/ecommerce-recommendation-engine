/**
 * Product detail page.
 *
 * Three rails, and the distinction between them is the point (FR-02 to FR-04):
 *
 *   Similar products      - substitutes, from the content embedding
 *   Frequently bought     - complements, from basket co-occurrence
 *   Customers also viewed - comparison set, from session co-visitation
 *
 * Serving the same list under all three headings is the classic broken product
 * page: "customers who bought this phone also bought these four other phones".
 */

import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  fetchAlsoViewed,
  fetchExplanation,
  fetchFrequentlyBought,
  fetchSimilar,
} from '@/api/client';
import { RecommendationRail } from '@/components/RecommendationRail';
import { useTracking } from '@/hooks/useTracking';
import { useAppDispatch, useAppSelector, cartAdded } from '@/store';
import type { ExplanationResponse, RecommendationSection } from '@/types/api';

export function ProductPage() {
  const { productId } = useParams<{ productId: string }>();
  const id = Number(productId);
  const userId = useAppSelector((s) => s.session.userId);
  const diagnostics = useAppSelector((s) => s.session.diagnostics);
  const dispatch = useAppDispatch();
  const { track } = useTracking();

  const [similar, setSimilar] = useState<RecommendationSection | null>(null);
  const [bought, setBought] = useState<RecommendationSection | null>(null);
  const [viewed, setViewed] = useState<RecommendationSection | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!Number.isFinite(id) || id <= 0) return;
    track('PRODUCT_VIEW', id, {}, 'pdp');

    let cancelled = false;
    setLoading(true);

    // Fired together rather than sequentially: they are independent, and
    // awaiting them in series would make the page three round trips deep.
    Promise.allSettled([
      fetchSimilar(id),
      fetchFrequentlyBought(id),
      fetchAlsoViewed(id),
    ])
      .then(([s, f, v]) => {
        if (cancelled) return;
        if (s.status === 'fulfilled') setSimilar(s.value);
        if (f.status === 'fulfilled') setBought(f.value);
        if (v.status === 'fulfilled') setViewed(v.value);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [id, track]);

  useEffect(() => {
    if (!diagnostics || !userId || !Number.isFinite(id)) return;
    fetchExplanation(userId, id)
      .then(setExplanation)
      .catch(() => setExplanation(null));
  }, [diagnostics, userId, id]);

  const anchor = similar?.items[0]?.product;

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <div className="mb-10 grid gap-8 md:grid-cols-[320px_1fr]">
        <div className="flex h-72 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 text-6xl font-bold text-slate-300">
          {String(id).slice(0, 2)}
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">Product #{id}</p>
          <h1 className="mt-1 text-2xl font-bold text-slate-900">
            {anchor ? `Similar to ${anchor.name}` : `Product ${id}`}
          </h1>
          <p className="mt-4 max-w-prose text-sm text-slate-600">
            This page exists to exercise the product-page recommendation surfaces.
            Viewing it emits a <code className="rounded bg-slate-100 px-1">PRODUCT_VIEW</code>{' '}
            event, which updates the real-time counters and changes what the
            homepage recommends on your next visit.
          </p>

          <div className="mt-6 flex gap-3">
            <button
              type="button"
              onClick={() => {
                track('ADD_TO_CART', id, { quantity: 1 }, 'pdp');
                dispatch(cartAdded({ productId: id, name: `Product ${id}`, price: 0 }));
              }}
              className="rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-brand-700"
            >
              Add to cart
            </button>
            <button
              type="button"
              onClick={() => track('WISHLIST', id, {}, 'pdp')}
              className="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-400"
            >
              Save for later
            </button>
          </div>

          {diagnostics && explanation && (
            <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Why this product would be recommended
              </p>
              {explanation.would_recommend ? (
                <>
                  <p className="mt-2 text-sm text-slate-800">
                    {explanation.explanation?.text} (rank {explanation.position}, score{' '}
                    {explanation.score?.toFixed(3)})
                  </p>
                  <pre className="mt-2 overflow-x-auto rounded bg-white p-2 font-mono text-[11px] text-slate-600">
                    {JSON.stringify(explanation.score_components, null, 2)}
                  </pre>
                </>
              ) : (
                <p className="mt-2 text-sm text-slate-600">{explanation.reason}</p>
              )}
            </div>
          )}
        </div>
      </div>

      <RecommendationRail
        title="Similar products"
        section={similar}
        source="pdp_similar"
        loading={loading}
        showDiagnostics={diagnostics}
      />
      <RecommendationRail
        title="Frequently bought together"
        section={bought}
        source="pdp_frequently_bought_together"
        loading={loading}
        showDiagnostics={diagnostics}
        emptyMessage="No complement has enough co-purchase evidence for this product yet."
      />
      <RecommendationRail
        title="Customers who viewed this also viewed"
        section={viewed}
        source="pdp_also_viewed"
        loading={loading}
        showDiagnostics={diagnostics}
      />
    </div>
  );
}
