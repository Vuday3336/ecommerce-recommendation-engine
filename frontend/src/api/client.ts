/**
 * Axios client and API surface.
 *
 * One session id is generated per browser and persisted, because co-visitation
 * and session-based cold start both depend on the session being stable across
 * requests. A session id regenerated per page load would make every visitor
 * look like a fresh one-page session, and the strongest signal an anonymous
 * user produces would be destroyed.
 */

import axios, { type AxiosInstance } from 'axios';
import type {
  ExplanationResponse,
  HomeRecommendations,
  RecommendationHealth,
  RecommendationSection,
  TrackedEvent,
} from '@/types/api';

const SESSION_STORAGE_KEY = 'recsys.session_id';
const TOKEN_STORAGE_KEY = 'recsys.access_token';

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    // Private browsing and blocked site data both throw here. The app must
    // still work; it just loses session continuity.
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

export function getSessionId(): string {
  const existing = readStorage(SESSION_STORAGE_KEY);
  if (existing && existing.length >= 8) return existing;
  const generated =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().replace(/-/g, '')
      : `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  writeStorage(SESSION_STORAGE_KEY, generated);
  return generated;
}

export function getAccessToken(): string | null {
  return readStorage(TOKEN_STORAGE_KEY);
}

export function setAccessToken(token: string): void {
  writeStorage(TOKEN_STORAGE_KEY, token);
}

export const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api/v1',
  timeout: 8000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.request.use((config) => {
  config.headers.set('X-Session-Id', getSessionId());
  const token = getAccessToken();
  if (token) config.headers.set('Authorization', `Bearer ${token}`);
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Recommendation failures must never break a page render. Callers handle a
    // rejected promise by rendering nothing for that rail; the storefront
    // continues to work.
    if (import.meta.env.DEV) {
      // eslint-disable-next-line no-console
      console.warn('API error', error?.config?.url, error?.response?.status);
    }
    return Promise.reject(error);
  },
);

// ---------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------

export async function fetchHome(
  userId: number | null,
  limit = 12,
): Promise<HomeRecommendations> {
  const path = userId ? `/recommendations/home/${userId}` : '/recommendations/home';
  const { data } = await api.get<HomeRecommendations>(path, { params: { limit } });
  return data;
}

export async function fetchSimilar(
  productId: number,
  limit = 10,
): Promise<RecommendationSection> {
  const { data } = await api.get<RecommendationSection>(
    `/recommendations/similar/${productId}`,
    { params: { limit } },
  );
  return data;
}

export async function fetchFrequentlyBought(
  productId: number,
  limit = 5,
): Promise<RecommendationSection> {
  const { data } = await api.get<RecommendationSection>(
    `/recommendations/frequently-bought/${productId}`,
    { params: { limit } },
  );
  return data;
}

export async function fetchAlsoViewed(
  productId: number,
  limit = 10,
): Promise<RecommendationSection> {
  const { data } = await api.get<RecommendationSection>(
    `/recommendations/also-viewed/${productId}`,
    { params: { limit } },
  );
  return data;
}

export async function fetchTrending(limit = 12): Promise<RecommendationSection> {
  const { data } = await api.get<RecommendationSection>('/recommendations/trending', {
    params: { limit },
  });
  return data;
}

export async function fetchRecentlyViewed(
  userId: number,
  limit = 20,
): Promise<RecommendationSection> {
  const { data } = await api.get<RecommendationSection>(
    `/recommendations/recent/${userId}`,
    { params: { limit } },
  );
  return data;
}

export async function fetchExplanation(
  userId: number,
  productId: number,
): Promise<ExplanationResponse> {
  const { data } = await api.get<ExplanationResponse>(
    `/recommendations/explain/${userId}/${productId}`,
  );
  return data;
}

export async function fetchHealth(): Promise<RecommendationHealth> {
  const { data } = await api.get<RecommendationHealth>('/recommendations/health');
  return data;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export async function sendEvents(events: TrackedEvent[]): Promise<void> {
  if (events.length === 0) return;
  await api.post('/events/batch', { events });
}

export function sendEventsBeacon(events: TrackedEvent[]): boolean {
  /**
   * Flush on page unload.
   *
   * `sendBeacon` is the only reliable way to send during unload: a normal XHR
   * is cancelled when the document goes away, which would silently drop the
   * last events of every session - exactly the ones nearest a conversion.
   */
  if (events.length === 0 || typeof navigator === 'undefined' || !navigator.sendBeacon) {
    return false;
  }
  const base = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';
  const blob = new Blob([JSON.stringify({ events })], { type: 'application/json' });
  return navigator.sendBeacon(`${base}/events/batch`, blob);
}
