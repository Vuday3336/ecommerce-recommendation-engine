/**
 * Application shell and routing.
 *
 * Two applications share one build: the storefront, which generates behaviour
 * and displays recommendations, and the admin dashboard, which reports on the
 * models. They share the API client and the tracking hook, so the events the
 * dashboard analyses are the same ones the storefront emits.
 */

import { NavLink, Route, Routes } from 'react-router-dom';
import { AdminDashboard } from '@/pages/AdminDashboard';
import { HomePage } from '@/pages/HomePage';
import { ProductPage } from '@/pages/ProductPage';
import {
  selectCartCount,
  toggledDiagnostics,
  switchedUser,
  useAppDispatch,
  useAppSelector,
} from '@/store';

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        `rounded-lg px-3 py-1.5 text-sm font-medium transition ${
          isActive ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-100'
        }`
      }
    >
      {label}
    </NavLink>
  );
}

export function App() {
  const dispatch = useAppDispatch();
  const userId = useAppSelector((s) => s.session.userId);
  const diagnostics = useAppSelector((s) => s.session.diagnostics);
  const cartCount = useAppSelector(selectCartCount);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-6 py-3">
          <span className="text-sm font-bold tracking-tight text-brand-700">recsys</span>

          <nav className="flex gap-1">
            <NavItem to="/" label="Storefront" />
            <NavItem to="/admin" label="ML operations" />
          </nav>

          <div className="ml-auto flex items-center gap-3 text-sm">
            <label className="flex items-center gap-2 text-slate-500">
              user
              <input
                type="number"
                min={1}
                value={userId ?? ''}
                onChange={(e) => dispatch(switchedUser(Number(e.target.value)))}
                className="w-20 rounded border border-slate-300 px-2 py-1 font-mono text-xs"
              />
            </label>

            {/* Switching users is what demonstrates personalisation: the same
                page, a different history, a visibly different set of rails. */}
            <button
              type="button"
              onClick={() => dispatch(toggledDiagnostics())}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                diagnostics
                  ? 'border-brand-500 bg-brand-50 text-brand-700'
                  : 'border-slate-300 text-slate-600 hover:border-slate-400'
              }`}
            >
              {diagnostics ? 'diagnostics on' : 'diagnostics off'}
            </button>

            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
              cart {cartCount}
            </span>
          </div>
        </div>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/product/:productId" element={<ProductPage />} />
          <Route path="/admin" element={<AdminDashboard />} />
          <Route
            path="*"
            element={
              <div className="mx-auto max-w-7xl px-6 py-16 text-center text-slate-500">
                Page not found.
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
