/**
 * Admin ML dashboard (Phase 23).
 *
 * Four panels, each answering a question an operator actually asks:
 *
 *   Model performance      - is the model good?
 *   Model monitoring       - which model is serving, and is it healthy?
 *   Recommendation mix     - what is it actually recommending?
 *   Drift                  - has the world moved since it was trained?
 *
 * All numbers come from the artefact directory the engine is serving from, so
 * the dashboard cannot show metrics for a model that is not answering requests.
 */

import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '@/api/client';

const PALETTE = ['#3364ff', '#1d43f5', '#598bff', '#8eb4ff', '#16a34a', '#f59e0b', '#ef4444', '#8b5cf6'];

interface ComparisonRow {
  model: string;
  'ndcg@10': number;
  'recall@10': number;
  'precision@10': number;
  'hit_rate@10': number;
  coverage: number;
  diversity: number;
  novelty: number;
  personalisation: number;
  users: number;
}

interface SignificanceRow {
  candidate: string;
  difference: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  significant: boolean;
}

interface ModelsPayload {
  comparison: ComparisonRow[];
  by_segment: Record<string, unknown>[];
  stage1_recall: number | null;
  source_contribution: Record<string, number>;
  feature_importance: { feature: string; gain_share: number }[];
  significance: SignificanceRow[];
  split: Record<string, unknown>;
  ranker_best_iteration: number | null;
}

interface MonitoringPayload {
  model: {
    name: string;
    version: string;
    loaded: boolean;
    models: string[];
    has_ranker: boolean;
    catalogue_size: number;
    users_with_features: number;
  };
  training: { split: Record<string, unknown>; timings_seconds: Record<string, number> };
  event_sink: Record<string, number>;
  cache_available: boolean;
  database_available: boolean;
}

interface DistributionPayload {
  sampled_users: number;
  distinct_products: number;
  catalogue_size: number;
  coverage: number;
  products: { product_id: number; name: string; count: number; share: number }[];
  sources: Record<string, number>;
  categories: { category_id: number; name: string; count: number }[];
}

interface DriftPayload {
  should_retrain: boolean;
  features: {
    feature: string;
    psi: number;
    ks_p_value: number | null;
    severity: string;
    reference_mean: number | null;
    current_mean: number | null;
  }[];
}

function Metric({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'default' | 'good' | 'warn' | 'bad';
}) {
  const toneClass = {
    default: 'text-slate-900',
    good: 'text-emerald-600',
    warn: 'text-amber-600',
    bad: 'text-red-600',
  }[tone];
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${toneClass}`}>{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-400">{hint}</p>}
    </div>
  );
}

function Panel({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="mb-8 rounded-2xl border border-slate-200 bg-white p-6">
      <h2 className="text-base font-semibold text-slate-900">{title}</h2>
      {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

export function AdminDashboard() {
  const [models, setModels] = useState<ModelsPayload | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringPayload | null>(null);
  const [distribution, setDistribution] = useState<DistributionPayload | null>(null);
  const [drift, setDrift] = useState<DriftPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([
      api.get<ModelsPayload>('/admin/models'),
      api.get<MonitoringPayload>('/admin/monitoring'),
      api.get<DistributionPayload>('/admin/distribution', { params: { sample_users: 200 } }),
      api.get<DriftPayload>('/admin/drift'),
    ]).then(([m, mo, d, dr]) => {
      if (m.status === 'fulfilled') setModels(m.value.data);
      if (mo.status === 'fulfilled') setMonitoring(mo.value.data);
      if (d.status === 'fulfilled') setDistribution(d.value.data);
      if (dr.status === 'fulfilled') setDrift(dr.value.data);
      if (m.status === 'rejected') {
        setError('Admin data unavailable. Sign in as an analyst, or train a model first.');
      }
    });
  }, []);

  const best = models?.comparison?.[0];
  const alerting = drift?.features.filter((f) => f.severity === 'alert').length ?? 0;

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">ML operations</h1>
        <p className="mt-1 text-sm text-slate-500">
          Offline evaluation, serving health, recommendation mix and feature drift.
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          {error}
        </div>
      )}

      <div className="mb-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Metric
          label="Best model NDCG@10"
          value={best ? best['ndcg@10'].toFixed(4) : '—'}
          hint={best?.model}
        />
        <Metric
          label="Stage-1 recall@300"
          value={models?.stage1_recall != null ? models.stage1_recall.toFixed(3) : '—'}
          hint="ceiling on the whole system"
        />
        <Metric
          label="Catalogue coverage"
          value={best ? `${(best.coverage * 100).toFixed(1)}%` : '—'}
          tone={best && best.coverage < 0.1 ? 'warn' : 'default'}
          hint="share of products ever recommended"
        />
        <Metric
          label="Drift alerts"
          value={String(alerting)}
          tone={alerting >= 2 ? 'bad' : alerting > 0 ? 'warn' : 'good'}
          hint={drift?.should_retrain ? 'retrain recommended' : 'within thresholds'}
        />
      </div>

      <Panel
        title="Model comparison"
        subtitle="Offline evaluation on the held-out future window. Higher is better."
      >
        {models && (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart
              data={models.comparison.slice(0, 10)}
              layout="vertical"
              margin={{ left: 140, right: 24 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="model" type="category" width={140} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: number) => v.toFixed(5)} />
              <Legend />
              <Bar dataKey="ndcg@10" fill="#3364ff" name="NDCG@10" radius={[0, 4, 4, 0]} />
              <Bar dataKey="recall@10" fill="#8eb4ff" name="Recall@10" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}

        {models && models.significance.length > 0 && (
          <div className="mt-6 overflow-x-auto">
            <p className="mb-2 text-sm font-medium text-slate-700">
              Paired bootstrap vs the strongest baseline — a point estimate is not a result
            </p>
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="py-2">Model</th>
                  <th>Δ NDCG@10</th>
                  <th>95% CI</th>
                  <th>p</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {models.significance.slice(0, 8).map((row) => (
                  <tr key={row.candidate} className="border-b border-slate-100">
                    <td className="py-1.5 font-sans">{row.candidate}</td>
                    <td className={row.difference > 0 ? 'text-emerald-600' : 'text-red-600'}>
                      {row.difference > 0 ? '+' : ''}
                      {row.difference.toFixed(5)}
                    </td>
                    <td className="text-slate-500">
                      [{row.ci_low.toFixed(4)}, {row.ci_high.toFixed(4)}]
                    </td>
                    <td>{row.p_value.toFixed(4)}</td>
                    <td className="font-sans">
                      {row.significant ? (
                        <span className={row.difference > 0 ? 'text-emerald-600' : 'text-red-600'}>
                          {row.difference > 0 ? 'better' : 'worse'}
                        </span>
                      ) : (
                        <span className="text-slate-400">not significant</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Retrieval source mix" subtitle="Recall contribution per candidate source">
          {models && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={Object.entries(models.source_contribution).map(([source, value]) => ({
                  source,
                  value,
                }))}
                layout="vertical"
                margin={{ left: 150 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="source" type="category" width={150} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => v.toFixed(4)} />
                <Bar dataKey="value" fill="#1d43f5" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel title="What gets served" subtitle="Source of served items across sampled users">
          {distribution && (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={Object.entries(distribution.sources).map(([name, value]) => ({
                    name,
                    value,
                  }))}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={(entry) => entry.name}
                >
                  {Object.keys(distribution.sources).map((key, index) => (
                    <Cell key={key} fill={PALETTE[index % PALETTE.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <Panel
        title="Most recommended products"
        subtitle={
          distribution
            ? `${distribution.distinct_products} distinct products across ${distribution.sampled_users} users — ${(distribution.coverage * 100).toFixed(1)}% of the catalogue`
            : undefined
        }
      >
        {distribution && (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={distribution.products.slice(0, 15)} margin={{ bottom: 90 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="name"
                angle={-40}
                textAnchor="end"
                height={110}
                interval={0}
                tick={{ fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#598bff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Model monitoring" subtitle="What is serving right now">
          {monitoring && (
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <dt className="text-slate-500">Model</dt>
              <dd className="font-mono">{monitoring.model.name}</dd>
              <dt className="text-slate-500">Version</dt>
              <dd className="font-mono">{monitoring.model.version}</dd>
              <dt className="text-slate-500">Ranker loaded</dt>
              <dd className={monitoring.model.has_ranker ? 'text-emerald-600' : 'text-amber-600'}>
                {monitoring.model.has_ranker ? 'yes' : 'no — serving hybrid'}
              </dd>
              <dt className="text-slate-500">Catalogue</dt>
              <dd className="font-mono">{monitoring.model.catalogue_size.toLocaleString()}</dd>
              <dt className="text-slate-500">Users with features</dt>
              <dd className="font-mono">{monitoring.model.users_with_features.toLocaleString()}</dd>
              <dt className="text-slate-500">Cache</dt>
              <dd className={monitoring.cache_available ? 'text-emerald-600' : 'text-amber-600'}>
                {monitoring.cache_available ? 'connected' : 'unavailable (degraded)'}
              </dd>
              <dt className="text-slate-500">Database</dt>
              <dd className={monitoring.database_available ? 'text-emerald-600' : 'text-amber-600'}>
                {monitoring.database_available ? 'connected' : 'unavailable (degraded)'}
              </dd>
              <dt className="text-slate-500">Events written</dt>
              <dd className="font-mono">{monitoring.event_sink.written ?? 0}</dd>
              <dt className="text-slate-500">Events dropped</dt>
              <dd
                className={
                  (monitoring.event_sink.dropped ?? 0) > 0 ? 'font-mono text-red-600' : 'font-mono'
                }
              >
                {monitoring.event_sink.dropped ?? 0}
              </dd>
            </dl>
          )}
        </Panel>

        <Panel
          title="Feature drift"
          subtitle="PSI against the distribution the serving model was trained on"
        >
          {drift && (
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="py-2">Feature</th>
                  <th>PSI</th>
                  <th>Train mean</th>
                  <th>Now</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {drift.features.map((row) => (
                  <tr key={row.feature} className="border-b border-slate-100">
                    <td className="py-1.5 font-sans">{row.feature}</td>
                    <td>{row.psi.toFixed(4)}</td>
                    <td className="text-slate-500">{row.reference_mean?.toFixed(3) ?? '—'}</td>
                    <td className="text-slate-500">{row.current_mean?.toFixed(3) ?? '—'}</td>
                    <td
                      className={
                        row.severity === 'alert'
                          ? 'text-red-600'
                          : row.severity === 'warning'
                            ? 'text-amber-600'
                            : 'text-emerald-600'
                      }
                    >
                      {row.severity}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      </div>

      <Panel title="Ranking feature importance" subtitle="Gain share, LightGBM LambdaRank">
        {models && (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart
              data={models.feature_importance.slice(0, 12)}
              layout="vertical"
              margin={{ left: 200 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
              <YAxis dataKey="feature" type="category" width={200} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
              <Bar dataKey="gain_share" fill="#16a34a" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Panel>
    </div>
  );
}
