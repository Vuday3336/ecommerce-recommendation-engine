/**
 * API response types.
 *
 * These mirror the Pydantic schemas in `backend/app/schemas/`. They are written
 * by hand rather than generated so the frontend can be read on its own, but the
 * shapes must stay in step - the integration test in `tests/` calls the real API
 * and asserts these fields exist, which is what catches a drift between them.
 */

export interface ProductSummary {
  id: number;
  name: string;
  price: number;
  category_id: number;
  brand_id: number;
  rating_average: number;
  image_url: string | null;
}

export interface RecommendationItem {
  product: ProductSummary;
  score: number;
  /** Candidate source that produced this item, e.g. "collaborative_als". */
  recommendation_type: string;
  explanation: string;
  explanation_evidence: Record<string, unknown>;
  score_components: Record<string, number>;
  position: number;
  /** Echoed back on impression and click so outcomes join to this decision. */
  recommendation_id: string | null;
}

export interface RecommendationSection {
  section: string;
  surface: string;
  items: RecommendationItem[];
  /** Which rung of the degradation ladder served this section. */
  strategy: string;
  model_version: string;
  request_id: string;
  variant: string | null;
  experiment_key: string | null;
  candidate_pool_size: number;
  cache_hit: boolean;
  latency_ms: number;
}

export interface HomeRecommendations {
  user_id: number | null;
  for_you: RecommendationItem[];
  because_you_viewed: RecommendationItem[];
  trending: RecommendationItem[];
  frequently_bought_together: RecommendationItem[];
  continue_shopping: RecommendationItem[];
  sections: Record<string, RecommendationSection>;
  model_version: string;
  request_id: string;
}

export interface ExplanationResponse {
  product_id: number;
  would_recommend: boolean;
  position: number | null;
  score: number | null;
  explanation: {
    text: string;
    reason_code: string;
    evidence: Record<string, unknown>;
    confidence: number;
  } | null;
  score_components: Record<string, number>;
  strategy: string | null;
  model_version: string;
  reason: string | null;
}

export type EventType =
  | 'PRODUCT_VIEW'
  | 'PRODUCT_CLICK'
  | 'SEARCH'
  | 'ADD_TO_CART'
  | 'REMOVE_FROM_CART'
  | 'WISHLIST'
  | 'PURCHASE'
  | 'PRODUCT_SHARE'
  | 'PRODUCT_RATING'
  | 'PRODUCT_REVIEW'
  | 'SESSION_START'
  | 'SESSION_END';

export interface TrackedEvent {
  event_type: EventType;
  session_id: string;
  product_id?: number | null;
  occurred_at?: string;
  source?: string;
  device_type?: 'desktop' | 'mobile' | 'tablet' | 'unknown';
  recommendation_id?: number | null;
  metadata?: Record<string, unknown>;
}

export interface EngineHealth {
  loaded: boolean;
  model_version: string;
  models: string[];
  catalogue_size: number;
  has_ranker: boolean;
  has_pipeline: boolean;
  users_with_features: number;
}

export interface RecommendationHealth {
  status: string;
  engine: EngineHealth;
  event_sink: Record<string, number>;
}
