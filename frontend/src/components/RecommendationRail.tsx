/**
 * One recommendation rail.
 *
 * Every recommendation surface in the storefront renders through this single
 * component. That is deliberate: impression tracking, explanation display and
 * click attribution are subtle and easy to get slightly wrong, and having one
 * implementation means they are right everywhere or wrong everywhere - never
 * silently inconsistent between the homepage and the product page.
 */

import { useCallback } from 'react';
import { Link } from 'react-router-dom';
import type { RecommendationItem, RecommendationSection } from '@/types/api';
import { useImpression, useTracking } from '@/hooks/useTracking';

interface RailProps {
  title: string;
  section?: RecommendationSection | null;
  items?: RecommendationItem[];
  source: string;
  loading?: boolean;
  /** Show strategy, latency and model version. Useful in the admin view. */
  showDiagnostics?: boolean;
  emptyMessage?: string;
}

function formatPrice(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

function ProductCard({
  item,
  source,
  showScore,
}: {
  item: RecommendationItem;
  source: string;
  showScore: boolean;
}) {
  const { trackImpression, trackClick } = useTracking();

  const onVisible = useCallback(
    (visibleMs: number) => trackImpression(item.product.id, item.position, visibleMs, source),
    [item.product.id, item.position, source, trackImpression],
  );
  const ref = useImpression(onVisible);

  return (
    <Link
      ref={ref}
      to={`/product/${item.product.id}`}
      onClick={() => trackClick(item.product.id, item.position, source)}
      className="group flex w-56 shrink-0 flex-col rounded-xl border border-slate-200 bg-white p-3 transition hover:border-brand-400 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-brand-500"
    >
      <div className="mb-3 flex h-28 items-center justify-center rounded-lg bg-gradient-to-br from-slate-100 to-slate-200 text-3xl font-semibold text-slate-400">
        {item.product.name.slice(0, 1)}
      </div>

      <p className="line-clamp-2 min-h-[2.5rem] text-sm font-medium text-slate-800 group-hover:text-brand-700">
        {item.product.name}
      </p>

      <div className="mt-2 flex items-baseline justify-between">
        <span className="text-base font-semibold text-slate-900">
          {formatPrice(item.product.price)}
        </span>
        <span className="text-xs text-amber-600">
          {item.product.rating_average.toFixed(1)} &#9733;
        </span>
      </div>

      {/* The explanation is the product feature, not a debug string (FR-11). */}
      <p className="mt-2 line-clamp-2 text-xs italic text-slate-500">{item.explanation}</p>

      {showScore && (
        <p className="mt-1 font-mono text-[10px] text-slate-400">
          {item.score.toFixed(3)} · {item.recommendation_type}
        </p>
      )}
    </Link>
  );
}

export function RecommendationRail({
  title,
  section,
  items,
  source,
  loading = false,
  showDiagnostics = false,
  emptyMessage,
}: RailProps) {
  const resolved = items ?? section?.items ?? [];

  if (loading) {
    return (
      <section className="mb-10">
        <h2 className="mb-3 text-lg font-semibold text-slate-900">{title}</h2>
        <div className="flex gap-4 overflow-hidden">
          {Array.from({ length: 5 }).map((_, index) => (
            <div
              key={index}
              className="h-56 w-56 shrink-0 animate-pulse rounded-xl bg-slate-200"
            />
          ))}
        </div>
      </section>
    );
  }

  // An empty rail renders nothing rather than an empty box. A rail with no
  // items is a normal outcome - a product with no complements, a user with no
  // history - and showing an empty shell makes the page look broken.
  if (resolved.length === 0) {
    if (!emptyMessage) return null;
    return (
      <section className="mb-10">
        <h2 className="mb-3 text-lg font-semibold text-slate-900">{title}</h2>
        <p className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500">
          {emptyMessage}
        </p>
      </section>
    );
  }

  return (
    <section className="mb-10">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        {showDiagnostics && section && (
          <p className="font-mono text-xs text-slate-400">
            {section.strategy} · {section.model_version} · pool {section.candidate_pool_size} ·{' '}
            {section.latency_ms.toFixed(0)}ms{section.cache_hit ? ' · cached' : ''}
            {section.variant ? ` · ${section.variant}` : ''}
          </p>
        )}
      </div>

      <div className="flex gap-4 overflow-x-auto pb-2 [scrollbar-width:thin]">
        {resolved.map((item) => (
          <ProductCard
            key={`${item.recommendation_id ?? item.product.id}`}
            item={item}
            source={source}
            showScore={showDiagnostics}
          />
        ))}
      </div>
    </section>
  );
}
