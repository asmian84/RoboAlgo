import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  // ── Dev server (local) ───────────────────────────────────────────────────────
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },

  // ── Production build ─────────────────────────────────────────────────────────
  build: {
    outDir: 'dist',
    // Emit a single chunk per route for smaller initial payload
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react':   ['react', 'react-dom', 'react-router-dom'],
          'vendor-charts':  ['lightweight-charts', 'recharts'],
          'vendor-supabase': ['@supabase/supabase-js'],
          'vendor-query':   ['@tanstack/react-query'],
        },
      },
    },
  },

  // Ensure environment variables are available in the build
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
})
