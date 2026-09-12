import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    // Proxy in development so the browser sees one origin. Without it every
    // request is cross-origin and needs CORS preflight, which adds a round
    // trip to each call and makes cookie-based auth awkward.
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Recharts is ~350 kB and is only needed by the admin dashboard, which
        // shoppers never open. Splitting it out keeps the storefront bundle
        // small; without this the charting library is on the critical path of
        // every product page.
        manualChunks: {
          charts: ['recharts'],
          vendor: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
});
