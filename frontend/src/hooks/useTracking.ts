/**
 * Behavioural event tracking.
 *
 * Three properties this has to guarantee (FR-23):
 *
 * 1. **Never blocks a render.** Events go into a queue; the queue is flushed on
 *    a timer, at a size threshold, and on page hide.
 * 2. **Never loses the tail of a session.** The unload flush uses `sendBeacon`,
 *    because a normal request is cancelled when the document goes away - and
 *    the events nearest a conversion are exactly the ones at the end.
 * 3. **Never breaks the page.** Every failure path is swallowed. Analytics
 *    going down must not take the storefront with it.
 */

import { useCallback, useEffect, useRef } from 'react';
import { getSessionId, sendEvents, sendEventsBeacon } from '@/api/client';
import type { EventType, TrackedEvent } from '@/types/api';

const FLUSH_INTERVAL_MS = 4000;
const FLUSH_AT_SIZE = 20;
const MAX_QUEUE = 200;

/** Module-level so every component shares one queue and one flush timer. */
const queue: TrackedEvent[] = [];

function detectDevice(): TrackedEvent['device_type'] {
  if (typeof window === 'undefined') return 'unknown';
  const width = window.innerWidth;
  if (width < 768) return 'mobile';
  if (width < 1024) return 'tablet';
  return 'desktop';
}

function enqueue(event: TrackedEvent): void {
  // Bounded: if the API is down, the queue must not grow without limit and
  // exhaust the tab's memory. Oldest events are dropped first, since the most
  // recent behaviour is the most valuable.
  if (queue.length >= MAX_QUEUE) queue.shift();
  queue.push(event);
}

async function flush(): Promise<void> {
  if (queue.length === 0) return;
  const batch = queue.splice(0, queue.length);
  try {
    await sendEvents(batch);
  } catch {
    /* dropped deliberately: analytics must never surface an error to the user */
  }
}

export interface Tracker {
  track: (
    eventType: EventType,
    productId?: number | null,
    metadata?: Record<string, unknown>,
    source?: string,
  ) => void;
  trackImpression: (productId: number, position: number, visibleMs: number, source: string) => void;
  trackClick: (productId: number, position: number, source: string) => void;
  flushNow: () => Promise<void>;
}

export function useTracking(): Tracker {
  const timer = useRef<number | null>(null);

  useEffect(() => {
    timer.current = window.setInterval(() => {
      void flush();
    }, FLUSH_INTERVAL_MS);

    const onHide = () => {
      if (queue.length === 0) return;
      const batch = queue.splice(0, queue.length);
      if (!sendEventsBeacon(batch)) {
        // Beacon unavailable: put them back and let the timer try.
        queue.unshift(...batch);
      }
    };

    // `visibilitychange` rather than `beforeunload`: mobile browsers routinely
    // never fire `beforeunload` when an app is backgrounded or the tab is
    // discarded, so relying on it loses most mobile session tails.
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') onHide();
    });
    window.addEventListener('pagehide', onHide);

    return () => {
      if (timer.current) window.clearInterval(timer.current);
      window.removeEventListener('pagehide', onHide);
      void flush();
    };
  }, []);

  const track = useCallback(
    (
      eventType: EventType,
      productId?: number | null,
      metadata?: Record<string, unknown>,
      source = 'app',
    ) => {
      enqueue({
        event_type: eventType,
        session_id: getSessionId(),
        product_id: productId ?? null,
        occurred_at: new Date().toISOString(),
        source,
        device_type: detectDevice(),
        metadata: metadata ?? {},
      });
      if (queue.length >= FLUSH_AT_SIZE) void flush();
    },
    [],
  );

  const trackImpression = useCallback(
    (productId: number, position: number, visibleMs: number, source: string) => {
      track('PRODUCT_VIEW', productId, { dwell_ms: visibleMs, position }, source);
    },
    [track],
  );

  const trackClick = useCallback(
    (productId: number, position: number, source: string) => {
      track('PRODUCT_CLICK', productId, { position }, source);
    },
    [track],
  );

  return { track, trackImpression, trackClick, flushNow: flush };
}

/**
 * Fire a callback once when an element has been genuinely visible.
 *
 * This is what makes impressions honest. Counting an API response as an
 * impression would include rails the user never scrolled to, inflating the CTR
 * denominator for exactly the surfaces that are working. The dwell threshold
 * additionally excludes items that flicker past during a fast scroll.
 */
export function useImpression(
  onVisible: (visibleMs: number) => void,
  { threshold = 0.5, minVisibleMs = 800 }: { threshold?: number; minVisibleMs?: number } = {},
): (node: HTMLElement | null) => void {
  const reported = useRef(false);
  const shownAt = useRef<number | null>(null);
  const observer = useRef<IntersectionObserver | null>(null);

  return useCallback(
    (node: HTMLElement | null) => {
      if (observer.current) {
        observer.current.disconnect();
        observer.current = null;
      }
      if (!node || reported.current || typeof IntersectionObserver === 'undefined') return;

      observer.current = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) {
              shownAt.current = performance.now();
            } else if (shownAt.current !== null) {
              const visible = performance.now() - shownAt.current;
              shownAt.current = null;
              if (visible >= minVisibleMs && !reported.current) {
                reported.current = true;
                onVisible(Math.round(visible));
                observer.current?.disconnect();
              }
            }
          }
        },
        { threshold },
      );
      observer.current.observe(node);
    },
    [onVisible, threshold, minVisibleMs],
  );
}
