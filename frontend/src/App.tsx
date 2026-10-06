import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import TerminalLayout      from './components/layout/TerminalLayout'
import FinderPage          from './pages/FinderPage'
import ChartPage           from './pages/ChartPage'
import PaperTradingPage    from './pages/PaperTradingPage'
import SettingsPage        from './pages/SettingsPage'
import LoginPage           from './pages/LoginPage'
import { AlertProvider }     from './components/alerts/AlertManager'
import { AuthProvider, useAuth } from './context/AuthContext'

// ── Auth guard — redirects to /login if no session ────────────────────────────
function ProtectedLayout() {
  const { session, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-emerald-400 text-sm font-mono animate-pulse">Authenticating…</div>
      </div>
    )
  }

  if (!session) {
    return <Navigate to="/login" replace />
  }

  return (
    <AlertProvider>
      <TerminalLayout>
        <Routes>
          {/* ── The three destinations ──────────────────────────────────
             FINDER  the one scanner — penny, small-cap, ETF long/short,
                     breakout and crypto are PRESETS inside it
             CHART   the deep view, opened by clicking a finder result
             PAPER   the mock account, and the auto-trader that trades into it

             This replaced twelve page routes plus thirteen redirect aliases
             left over from an earlier consolidation. Everything removed is
             recoverable from tag pre-consolidation-2026-09-29. */}
          <Route path="/"        element={<Navigate to="/finder" replace />} />
          <Route path="/finder"  element={<FinderPage />} />
          <Route path="/chart"   element={<ChartPage />} />
          <Route path="/paper"   element={<PaperTradingPage />} />

          <Route path="/settings" element={<SettingsPage />} />

          {/* Old links and bookmarks land on the surface that absorbed them
             rather than on a dead end. */}
          <Route path="/trades"  element={<Navigate to="/finder" replace />} />
          <Route path="/crypto"  element={<Navigate to="/finder" replace />} />
          <Route path="/signals" element={<Navigate to="/finder" replace />} />
          <Route path="/scan"    element={<Navigate to="/finder" replace />} />
          <Route path="/auto"    element={<Navigate to="/paper" replace />} />

          {/* Catch-all — already authed, send to the scanner */}
          <Route path="*" element={<Navigate to="/finder" replace />} />
        </Routes>
      </TerminalLayout>
    </AlertProvider>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public route */}
          <Route path="/login" element={<LoginRoute />} />
          {/* All other routes are protected */}
          <Route path="/*" element={<ProtectedLayout />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

// Redirect already-authed users away from login page
function LoginRoute() {
  const { session, loading } = useAuth()
  if (loading) return null
  if (session) return <Navigate to="/" replace />
  return <LoginPage />
}
