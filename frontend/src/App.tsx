import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import TerminalLayout      from './components/layout/TerminalLayout'
import ChartPage           from './pages/ChartPage'
import SignalsPage         from './pages/SignalsPage'
import CommandCenterPage   from './pages/CommandCenterPage'
import RegimeTimelinePage  from './pages/RegimeTimelinePage'
import PaperTradingPage    from './pages/PaperTradingPage'
import OpportunityMatrixPage from './pages/OpportunityMatrixPage'
import AnalyticsPage         from './pages/AnalyticsPage'
import TickerIntelPage       from './pages/TickerIntelPage'
import LoginPage             from './pages/LoginPage'
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
          {/* ── Primary nav (6 sidebar items) ── */}
          <Route path="/"          element={<TickerIntelPage />} />
          <Route path="/intel"     element={<TickerIntelPage />} />
          <Route path="/chart"     element={<ChartPage />} />
          <Route path="/signals"   element={<SignalsPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/paper"     element={<PaperTradingPage />} />
          <Route path="/command"   element={<CommandCenterPage />} />

          {/* ── Consolidated redirects — Signals hub ── */}
          <Route path="/scan"           element={<Navigate to="/signals" replace />} />
          <Route path="/watchlist"      element={<Navigate to="/signals" replace />} />
          <Route path="/probabilities"  element={<Navigate to="/signals" replace />} />
          <Route path="/recommendation" element={<Navigate to="/signals" replace />} />
          <Route path="/comparison"     element={<Navigate to="/signals" replace />} />

          {/* ── Consolidated redirects — Analytics hub ── */}
          <Route path="/backtest"  element={<Navigate to="/analytics" replace />} />
          <Route path="/evolution" element={<Navigate to="/analytics" replace />} />
          <Route path="/research"  element={<Navigate to="/analytics" replace />} />
          <Route path="/features"  element={<Navigate to="/analytics" replace />} />

          {/* ── Consolidated redirects — Intel hub ── */}
          <Route path="/bull-bear" element={<Navigate to="/intel" replace />} />
          <Route path="/cycles"    element={<Navigate to="/intel" replace />} />
          <Route path="/dashboard" element={<Navigate to="/intel" replace />} />

          {/* ── Deep-link aliases (unique pages, kept accessible) ── */}
          <Route path="/matrix"          element={<OpportunityMatrixPage />} />
          <Route path="/regime-timeline" element={<RegimeTimelinePage />} />

          {/* Catch-all — already authed, send to root */}
          <Route path="*" element={<Navigate to="/" replace />} />
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
