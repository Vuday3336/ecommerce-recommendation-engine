/**
 * Redux store.
 *
 * Only genuinely global, cross-route state lives here: who is signed in, what
 * is in the cart, and which diagnostics the developer has toggled on.
 *
 * Recommendation payloads are deliberately **not** in Redux. They are
 * per-surface, short-lived, already cached server-side, and stale within
 * minutes. Putting them in a global store would add a second cache with
 * different invalidation rules to the one Redis already provides - two caches
 * that disagree is worse than one that is occasionally stale.
 */

import { configureStore, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { useDispatch, useSelector, type TypedUseSelectorHook } from 'react-redux';

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

interface SessionState {
  userId: number | null;
  displayName: string | null;
  role: 'customer' | 'analyst' | 'admin';
  /** Renders strategy, latency and score on every rail. */
  diagnostics: boolean;
}

const initialSession: SessionState = {
  // The demo signs in as a seeded user so the personalised surfaces have
  // history to work with. A real deployment resolves this from the token.
  userId: 42,
  displayName: 'Demo shopper',
  role: 'admin',
  diagnostics: false,
};

const sessionSlice = createSlice({
  name: 'session',
  initialState: initialSession,
  reducers: {
    signedIn(state, action: PayloadAction<{ userId: number; displayName: string }>) {
      state.userId = action.payload.userId;
      state.displayName = action.payload.displayName;
    },
    signedOut(state) {
      state.userId = null;
      state.displayName = null;
      state.role = 'customer';
    },
    switchedUser(state, action: PayloadAction<number>) {
      state.userId = action.payload;
    },
    toggledDiagnostics(state) {
      state.diagnostics = !state.diagnostics;
    },
  },
});

// ---------------------------------------------------------------------------
// Cart
// ---------------------------------------------------------------------------

export interface CartLine {
  productId: number;
  name: string;
  price: number;
  quantity: number;
}

interface CartState {
  lines: CartLine[];
}

const cartSlice = createSlice({
  name: 'cart',
  initialState: { lines: [] } as CartState,
  reducers: {
    added(state, action: PayloadAction<Omit<CartLine, 'quantity'>>) {
      const existing = state.lines.find((l) => l.productId === action.payload.productId);
      if (existing) {
        existing.quantity += 1;
      } else {
        state.lines.push({ ...action.payload, quantity: 1 });
      }
    },
    removed(state, action: PayloadAction<number>) {
      state.lines = state.lines.filter((l) => l.productId !== action.payload);
    },
    cleared(state) {
      state.lines = [];
    },
  },
});

export const { signedIn, signedOut, switchedUser, toggledDiagnostics } = sessionSlice.actions;
export const { added: cartAdded, removed: cartRemoved, cleared: cartCleared } = cartSlice.actions;

export const store = configureStore({
  reducer: {
    session: sessionSlice.reducer,
    cart: cartSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export const selectCartTotal = (state: RootState): number =>
  state.cart.lines.reduce((sum, line) => sum + line.price * line.quantity, 0);

export const selectCartCount = (state: RootState): number =>
  state.cart.lines.reduce((sum, line) => sum + line.quantity, 0);
