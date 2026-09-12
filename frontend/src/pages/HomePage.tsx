/**
 * Storefront homepage.
 *
 * One API call returns every rail (FR-07). Five calls would recompute the
 * user's context and candidate pass five times, which is the expensive part of
 * the request.
 */

import { useEffect, useState } from 'react';
import { fetchHome } from '@/api/client';
import { RecommendationRail } from '@/components/RecommendationRail';
import { useTracking } from '@/hooks/useTracking';
import { useAppSelector } from '@/store';
import type { HomeRecommendations } from '@/types/api';

export function HomePage() {
  const userId = useAppSelector((s) => s.session.userId);
  const diagnostics = useAppSelector((s) => s.session.diagnostics);
  const { track } = useTracking();

  const [data, setData] = useState<HomeRecommendations | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    track('SESSION_START', null, { entry: 'direct' });
  }, [track]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchHome(userId)
      .then((response) => {
        if (!cancelled) setData(response);
      })
      .catch(() => {
        // The storefront stays usable when recommendations fail; the rails are
        // simply absent rather than the page erroring.
        if (!cancelled) setError('Recommendations are unavailable right now.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [userId]);

  const sections = data?.sections ?? {};

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">
          {userId ? 'Your storefront' : 'Discover products'}
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          {userId
            ? `Personalised for user ${userId}`
            : 'Browsing anonymously — recommendations improve as you explore'}
          {data ? ` · model ${data.model_version}` : ''}
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          {error}
        </div>
      )}

      <RecommendationRail
        title="Recommended for you"
        section={sections.for_you}
        source="home_for_you"
        loading={loading}
        showDiagnostics={diagnostics}
        emptyMessage="Browse a few products and personalised recommendations will appear here."
      />

      <RecommendationRail
        title="Because you viewed"
        section={sections.because_you_viewed}
        source="home_because_you_viewed"
        loading={loading}
        showDiagnostics={diagnostics}
      />

      <RecommendationRail
        title="Trending now"
        section={sections.trending}
        source="home_trending"
        loading={loading}
        showDiagnostics={diagnostics}
      />

      <RecommendationRail
        title="Frequently bought together"
        section={sections.frequently_bought_together}
        source="home_frequently_bought_together"
        loading={loading}
        showDiagnostics={diagnostics}
      />

      <RecommendationRail
        title="Continue shopping"
        section={sections.continue_shopping}
        source="home_continue_shopping"
        loading={loading}
        showDiagnostics={diagnostics}
      />
    </div>
  );
}
