/**
 * frontend/src/pages/SettingsPage.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Data-provider API key management.
 *
 * Providers:
 *   Alpha Vantage — intraday MTF + daily OHLCV (free 5 req/min OR premium 75–600)
 *   Finnhub        — market data fallback
 *   Alpaca Markets — brokerage data feed (requires paid subscription)
 *   yfinance       — always free, no key needed
 *
 * Keys are stored server-side in settings.json — never in localStorage.
 * The "Test" button makes a live API call and shows latency + sample data.
 */

import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API = '/api/settings'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ProviderStatus {
  ok:         boolean
  provider?:  string
  tier?:      string
  rpm?:       number
  sample?:    string
  error?:     string
  elapsed_ms?: number
}

interface SettingsData {
  alphavantage: {
    api_key_masked: string
    api_key_set:    boolean
    rpm:            number
    tier:           string    // "free" | "premium" | "professional"
    enabled:        boolean
  }
  finnhub: {
    api_key_masked: string
    api_key_set:    boolean
    enabled:        boolean
  }
  alpaca: {
    api_key_masked:    string
    api_secret_masked: string
    api_key_set:       boolean
    api_secret_set:    boolean
    base_url:          string
    enabled:           boolean
  }
  data_sources: {
    intraday_mtf: string
    daily_ohlcv:  string
    options:      string
  }
}

// ── Helper components ─────────────────────────────────────────────────────────

function TierBadge({ tier }: { tier: string }) {
  const cfg: Record<string, { cls: string; label: string }> = {
    free:         { cls: 'text-gray-400 bg-gray-800 border-gray-700',           label: 'FREE' },
    premium:      { cls: 'text-amber-400 bg-amber-950 border-amber-700',        label: 'PREMIUM' },
    professional: { cls: 'text-emerald-400 bg-emerald-950 border-emerald-700',  label: 'PRO' },
  }
  const c = cfg[tier] ?? cfg.free
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[8px] font-black uppercase border ${c.cls}`}>
      {c.label}
    </span>
  )
}

function StatusDot({ ok, testing }: { ok?: boolean; testing?: boolean }) {
  if (testing) return <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
  if (ok === true)  return <div className="w-2 h-2 rounded-full bg-emerald-400" />
  if (ok === false) return <div className="w-2 h-2 rounded-full bg-red-400" />
  return <div className="w-2 h-2 rounded-full bg-gray-600" />
}

function TestResult({ result }: { result: ProviderStatus | null }) {
  if (!result) return null
  return (
    <div className={`mt-2 rounded p-2 text-[10px] border ${
      result.ok
        ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-300'
        : 'bg-red-950/40 border-red-800/50 text-red-400'
    }`}>
      {result.ok ? (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-bold">✓ Connected</span>
          {result.sample && <span className="text-emerald-400/80">{result.sample}</span>}
          {result.tier   && <TierBadge tier={result.tier} />}
          {result.rpm    && <span className="text-gray-500">{result.rpm} req/min</span>}
          {result.elapsed_ms != null && <span className="ml-auto text-gray-600">{result.elapsed_ms}ms</span>}
        </div>
      ) : (
        <span>✗ {result.error}</span>
      )}
    </div>
  )
}

// ── Provider Card ─────────────────────────────────────────────────────────────

interface ProviderCardProps {
  title:       string
  icon:        string
  description: string
  isSet:       boolean
  enabled:     boolean
  tags?:       string[]
  children:    React.ReactNode
}

function ProviderCard({ title, icon, description, isSet, enabled, tags, children }: ProviderCardProps) {
  return (
    <div className={`bg-gray-900 rounded-xl border p-5 space-y-4 transition-colors ${
      enabled && isSet ? 'border-gray-700' : 'border-gray-800/60'
    }`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{icon}</span>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-gray-100">{title}</span>
              {isSet  && <span className="text-[8px] font-bold text-emerald-400 bg-emerald-950 border border-emerald-800 px-1.5 py-0.5 rounded">KEY SET</span>}
              {!isSet && <span className="text-[8px] font-bold text-gray-500 bg-gray-800 border border-gray-700 px-1.5 py-0.5 rounded">NOT SET</span>}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5">{description}</div>
          </div>
        </div>
        <div className="flex gap-1 flex-wrap justify-end">
          {tags?.map(t => (
            <span key={t} className="text-[8px] text-violet-400 bg-violet-950/50 border border-violet-800/40 px-1.5 py-0.5 rounded">
              {t}
            </span>
          ))}
        </div>
      </div>
      {children}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const [settings,    setSettings]    = useState<SettingsData | null>(null)
  const [loading,     setLoading]     = useState(true)
  const [saving,      setSaving]      = useState(false)
  const [saveMsg,     setSaveMsg]     = useState('')

  // Form state — keys (blank = don't change stored key)
  const [avKey,       setAvKey]       = useState('')
  const [avRpm,       setAvRpm]       = useState(5)
  const [avEnabled,   setAvEnabled]   = useState(true)
  const [fhKey,       setFhKey]       = useState('')
  const [fhEnabled,   setFhEnabled]   = useState(true)
  const [alpKey,      setAlpKey]      = useState('')
  const [alpSecret,   setAlpSecret]   = useState('')
  const [alpEnabled,  setAlpEnabled]  = useState(false)
  const [mtfSource,   setMtfSource]   = useState('alphavantage')
  const [dailySource, setDailySource] = useState('alphavantage')

  // Test results per provider
  const [avTest,  setAvTest]  = useState<ProviderStatus | null>(null)
  const [fhTest,  setFhTest]  = useState<ProviderStatus | null>(null)
  const [alpTest, setAlpTest] = useState<ProviderStatus | null>(null)
  const [yfTest,  setYfTest]  = useState<ProviderStatus | null>(null)
  const [testing, setTesting] = useState<string | null>(null)

  // Visibility toggles for secret fields
  const [showAv,  setShowAv]  = useState(false)
  const [showFh,  setShowFh]  = useState(false)
  const [showAlp, setShowAlp] = useState(false)
  const [showAlpS,setShowAlpS]= useState(false)

  const fetchSettings = useCallback(async () => {
    try {
      const { data } = await axios.get<SettingsData>(API)
      setSettings(data)
      setAvRpm(data.alphavantage.rpm)
      setAvEnabled(data.alphavantage.enabled)
      setFhEnabled(data.finnhub.enabled)
      setAlpEnabled(data.alpaca.enabled)
      setMtfSource(data.data_sources.intraday_mtf)
      setDailySource(data.data_sources.daily_ohlcv)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchSettings() }, [fetchSettings])

  async function handleSave() {
    setSaving(true); setSaveMsg('')
    try {
      await axios.post(API, {
        alphavantage:  { api_key: avKey,    rpm: avRpm,    enabled: avEnabled },
        finnhub:       { api_key: fhKey,    enabled: fhEnabled },
        alpaca:        { api_key: alpKey,   api_secret: alpSecret, enabled: alpEnabled,
                         base_url: 'https://data.alpaca.markets' },
        data_sources:  { intraday_mtf: mtfSource, daily_ohlcv: dailySource, options: 'yahoo' },
      })
      setSaveMsg('✓ Saved and applied — no restart needed')
      setAvKey(''); setFhKey(''); setAlpKey(''); setAlpSecret('')
      await fetchSettings()
    } catch (e: unknown) {
      setSaveMsg('✗ Save failed')
    } finally {
      setSaving(false)
      setTimeout(() => setSaveMsg(''), 4000)
    }
  }

  async function handleTest(provider: string, extra: Record<string, string> = {}) {
    setTesting(provider)
    const clear = () => {
      if (provider === 'alphavantage') setAvTest(null)
      else if (provider === 'finnhub') setFhTest(null)
      else if (provider === 'alpaca')  setAlpTest(null)
      else if (provider === 'yfinance')setYfTest(null)
    }
    clear()
    try {
      const { data } = await axios.post<ProviderStatus>(
        `${API}/test/${provider}`,
        extra,
      )
      if (provider === 'alphavantage') setAvTest(data)
      else if (provider === 'finnhub') setFhTest(data)
      else if (provider === 'alpaca')  setAlpTest(data)
      else if (provider === 'yfinance')setYfTest(data)
    } catch {
      const err = { ok: false, error: 'Request failed' }
      if (provider === 'alphavantage') setAvTest(err)
      else if (provider === 'finnhub') setFhTest(err)
      else if (provider === 'alpaca')  setAlpTest(err)
      else if (provider === 'yfinance')setYfTest(err)
    } finally {
      setTesting(null)
    }
  }

  function inputCls(base = '') {
    return `w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-200
            focus:outline-none focus:border-emerald-600 placeholder-gray-600 transition-colors ${base}`
  }

  function toggleBtn(show: boolean, toggle: () => void) {
    return (
      <button type="button" onClick={toggle}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 text-[10px]">
        {show ? 'hide' : 'show'}
      </button>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const avTier  = settings ? settings.alphavantage.tier : 'free'
  const avIsSet = settings?.alphavantage.api_key_set ?? false
  const fhIsSet = settings?.finnhub.api_key_set     ?? false
  const alpIsSet= settings?.alpaca.api_key_set      ?? false

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">

      {/* ── Page header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-black text-gray-100 tracking-tight">Data Provider Settings</h1>
          <p className="text-[11px] text-gray-500 mt-0.5">
            Configure API keys for market data sources. Keys stored securely on the server.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {saveMsg && (
            <span className={`text-[10px] font-bold ${saveMsg.startsWith('✓') ? 'text-emerald-400' : 'text-red-400'}`}>
              {saveMsg}
            </span>
          )}
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-2 text-xs font-bold bg-emerald-700 hover:bg-emerald-600 disabled:bg-gray-700 text-white rounded-lg transition-colors"
          >
            {saving ? 'Saving…' : '💾 Save & Apply'}
          </button>
        </div>
      </div>

      {/* ── Alpha Vantage ── */}
      <ProviderCard
        title="Alpha Vantage"
        icon="📊"
        description="Intraday (15min, 60min) + daily OHLCV. Premium unlocks MTF alignment and higher call rates."
        isSet={avIsSet}
        enabled={avEnabled}
        tags={['MTF Alignment', 'Daily OHLCV', 'Pounce Scanner']}
      >
        {/* Current tier display */}
        {avIsSet && (
          <div className="flex items-center gap-2 bg-gray-800/50 rounded-lg px-3 py-2 border border-gray-700/50">
            <StatusDot ok={avIsSet} />
            <span className="text-[10px] text-gray-400">Current key:</span>
            <span className="text-[10px] font-mono text-gray-300">{settings?.alphavantage.api_key_masked}</span>
            <TierBadge tier={avTier} />
            <span className="text-[10px] text-gray-600 ml-auto">{avRpm} req/min</span>
          </div>
        )}

        {/* New key input */}
        <div className="space-y-3">
          <div>
            <label className="text-[9px] text-gray-600 uppercase tracking-wide block mb-1">
              API Key {avIsSet ? '(leave blank to keep current)' : '(required)'}
            </label>
            <div className="relative">
              <input
                type={showAv ? 'text' : 'password'}
                value={avKey}
                onChange={e => setAvKey(e.target.value)}
                placeholder={avIsSet ? '••••••••••••' : 'Enter your Alpha Vantage API key'}
                className={inputCls('pr-12')}
              />
              {toggleBtn(showAv, () => setShowAv(v => !v))}
            </div>
          </div>

          {/* RPM selector */}
          <div>
            <label className="text-[9px] text-gray-600 uppercase tracking-wide block mb-1">
              Calls Per Minute (from your AV dashboard)
            </label>
            <div className="flex gap-2">
              {[
                { rpm: 5,   label: 'Free',         sub: '5 req/min' },
                { rpm: 75,  label: 'Premium',      sub: '75 req/min' },
                { rpm: 600, label: 'Professional', sub: '600 req/min' },
              ].map(opt => (
                <button
                  key={opt.rpm}
                  type="button"
                  onClick={() => setAvRpm(opt.rpm)}
                  className={`flex-1 rounded-lg py-2 px-2 border text-center transition-all ${
                    avRpm === opt.rpm
                      ? 'bg-emerald-700/30 border-emerald-600 text-emerald-200'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600 hover:text-gray-300'
                  }`}
                >
                  <div className="text-[10px] font-bold">{opt.label}</div>
                  <div className="text-[9px] opacity-70">{opt.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Enabled toggle + test */}
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                onClick={() => setAvEnabled(v => !v)}
                className={`w-8 h-4 rounded-full border transition-colors cursor-pointer ${
                  avEnabled ? 'bg-emerald-700 border-emerald-600' : 'bg-gray-800 border-gray-700'
                }`}
              >
                <div className={`w-3 h-3 rounded-full bg-white m-0.5 transition-transform ${avEnabled ? 'translate-x-4' : ''}`} />
              </div>
              <span className="text-[10px] text-gray-500">Enabled</span>
            </label>
            <button
              type="button"
              onClick={() => handleTest('alphavantage', avKey ? { api_key: avKey } : {})}
              disabled={testing === 'alphavantage'}
              className="ml-auto px-3 py-1.5 text-[10px] font-bold bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded transition-colors disabled:opacity-50"
            >
              {testing === 'alphavantage' ? (
                <span className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 border border-amber-400 border-t-transparent rounded-full animate-spin" />
                  Testing…
                </span>
              ) : '⚡ Test Connection'}
            </button>
          </div>
          <TestResult result={avTest} />
        </div>

        <div className="text-[9px] text-gray-600 border-t border-gray-800 pt-3">
          Get your key at{' '}
          <a href="https://www.alphavantage.co/support/#api-key" target="_blank" rel="noreferrer"
             className="text-violet-400 hover:underline">alphavantage.co</a>
          {' '} · Premium unlocks 75+ req/min + 2yr intraday history
        </div>
      </ProviderCard>

      {/* ── yfinance (always free) ── */}
      <ProviderCard
        title="yfinance (Yahoo Finance)"
        icon="🆓"
        description="Free fallback — no API key needed. Used when Alpha Vantage is unavailable or rate-limited."
        isSet={true}
        enabled={true}
        tags={['Fallback', 'No Key Required']}
      >
        <div className="flex items-center gap-3 bg-gray-800/40 rounded-lg px-3 py-2 border border-gray-700/40">
          <StatusDot ok={true} />
          <span className="text-[10px] text-gray-400">Always available — no configuration needed</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-[10px] text-gray-500 flex-1">
            Limitations: 4H interval not native (resampled from 15m), occasional throttling under heavy load
          </div>
          <button
            type="button"
            onClick={() => handleTest('yfinance')}
            disabled={testing === 'yfinance'}
            className="px-3 py-1.5 text-[10px] font-bold bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded transition-colors disabled:opacity-50 shrink-0"
          >
            {testing === 'yfinance' ? '…' : '⚡ Test'}
          </button>
        </div>
        <TestResult result={yfTest} />
      </ProviderCard>

      {/* ── Finnhub ── */}
      <ProviderCard
        title="Finnhub"
        icon="📈"
        description="Market data, quotes, and fundamentals. Free tier gives 60 calls/min."
        isSet={fhIsSet}
        enabled={fhEnabled}
        tags={['Quotes', 'Fundamentals']}
      >
        {fhIsSet && (
          <div className="flex items-center gap-2 bg-gray-800/50 rounded-lg px-3 py-2 border border-gray-700/50">
            <StatusDot ok={fhIsSet} />
            <span className="text-[10px] text-gray-400">Current key:</span>
            <span className="text-[10px] font-mono text-gray-300">{settings?.finnhub.api_key_masked}</span>
          </div>
        )}
        <div>
          <label className="text-[9px] text-gray-600 uppercase tracking-wide block mb-1">
            API Key {fhIsSet ? '(leave blank to keep current)' : ''}
          </label>
          <div className="relative">
            <input
              type={showFh ? 'text' : 'password'}
              value={fhKey}
              onChange={e => setFhKey(e.target.value)}
              placeholder={fhIsSet ? '••••••••••••' : 'Enter your Finnhub API key'}
              className={inputCls('pr-12')}
            />
            {toggleBtn(showFh, () => setShowFh(v => !v))}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <div
              onClick={() => setFhEnabled(v => !v)}
              className={`w-8 h-4 rounded-full border transition-colors cursor-pointer ${
                fhEnabled ? 'bg-emerald-700 border-emerald-600' : 'bg-gray-800 border-gray-700'
              }`}
            >
              <div className={`w-3 h-3 rounded-full bg-white m-0.5 transition-transform ${fhEnabled ? 'translate-x-4' : ''}`} />
            </div>
            <span className="text-[10px] text-gray-500">Enabled</span>
          </label>
          <button
            type="button"
            onClick={() => handleTest('finnhub', fhKey ? { api_key: fhKey } : {})}
            disabled={testing === 'finnhub'}
            className="ml-auto px-3 py-1.5 text-[10px] font-bold bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded transition-colors disabled:opacity-50"
          >
            {testing === 'finnhub' ? '…' : '⚡ Test'}
          </button>
        </div>
        <TestResult result={fhTest} />
        <div className="text-[9px] text-gray-600 border-t border-gray-800 pt-3">
          Free key at{' '}
          <a href="https://finnhub.io" target="_blank" rel="noreferrer" className="text-violet-400 hover:underline">
            finnhub.io
          </a>
        </div>
      </ProviderCard>

      {/* ── Alpaca ── */}
      <ProviderCard
        title="Alpaca Markets"
        icon="🦙"
        description="Brokerage data feed — real-time SIP tick data. Requires an Alpaca account + Unlimited Data Plan ($99/mo)."
        isSet={alpIsSet}
        enabled={alpEnabled}
        tags={['Real-time SIP', 'Paid Subscription']}
      >
        <div className="bg-amber-950/20 border border-amber-800/30 rounded p-2 text-[9px] text-amber-400/70">
          ⚠ Requires Alpaca brokerage account + Unlimited Data subscription.
          Free tier returns 403 on historical bar endpoints.
        </div>

        {alpIsSet && (
          <div className="flex items-center gap-2 bg-gray-800/50 rounded-lg px-3 py-2 border border-gray-700/50">
            <StatusDot ok={alpIsSet} />
            <span className="text-[10px] text-gray-400">Key:</span>
            <span className="text-[10px] font-mono text-gray-300">{settings?.alpaca.api_key_masked}</span>
            <span className="text-[10px] text-gray-400 ml-3">Secret:</span>
            <span className="text-[10px] font-mono text-gray-300">{settings?.alpaca.api_secret_masked}</span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-3">
          <div>
            <label className="text-[9px] text-gray-600 uppercase tracking-wide block mb-1">API Key ID</label>
            <div className="relative">
              <input
                type={showAlp ? 'text' : 'password'}
                value={alpKey}
                onChange={e => setAlpKey(e.target.value)}
                placeholder={alpIsSet ? '••••••••' : 'APCA-API-KEY-ID'}
                className={inputCls('pr-12')}
              />
              {toggleBtn(showAlp, () => setShowAlp(v => !v))}
            </div>
          </div>
          <div>
            <label className="text-[9px] text-gray-600 uppercase tracking-wide block mb-1">Secret Key</label>
            <div className="relative">
              <input
                type={showAlpS ? 'text' : 'password'}
                value={alpSecret}
                onChange={e => setAlpSecret(e.target.value)}
                placeholder={settings?.alpaca.api_secret_set ? '••••••••' : 'APCA-API-SECRET-KEY'}
                className={inputCls('pr-12')}
              />
              {toggleBtn(showAlpS, () => setShowAlpS(v => !v))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <div
              onClick={() => setAlpEnabled(v => !v)}
              className={`w-8 h-4 rounded-full border transition-colors cursor-pointer ${
                alpEnabled ? 'bg-emerald-700 border-emerald-600' : 'bg-gray-800 border-gray-700'
              }`}
            >
              <div className={`w-3 h-3 rounded-full bg-white m-0.5 transition-transform ${alpEnabled ? 'translate-x-4' : ''}`} />
            </div>
            <span className="text-[10px] text-gray-500">Enabled</span>
          </label>
          <button
            type="button"
            onClick={() => handleTest('alpaca', {
              ...(alpKey    ? { api_key:    alpKey   } : {}),
              ...(alpSecret ? { api_secret: alpSecret } : {}),
            })}
            disabled={testing === 'alpaca' || (!alpIsSet && !alpKey)}
            className="ml-auto px-3 py-1.5 text-[10px] font-bold bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded transition-colors disabled:opacity-50"
          >
            {testing === 'alpaca' ? '…' : '⚡ Test'}
          </button>
        </div>
        <TestResult result={alpTest} />
        <div className="text-[9px] text-gray-600 border-t border-gray-800 pt-3">
          Sign up at{' '}
          <a href="https://alpaca.markets" target="_blank" rel="noreferrer" className="text-violet-400 hover:underline">
            alpaca.markets
          </a>
          {' '}→ Dashboard → API Keys
        </div>
      </ProviderCard>

      {/* ── Data Source Routing ── */}
      <div className="bg-gray-900 rounded-xl border border-gray-700 p-5 space-y-4">
        <div>
          <div className="text-sm font-bold text-gray-100">Data Source Routing</div>
          <div className="text-[10px] text-gray-500 mt-0.5">
            Which provider handles each data type. Changes apply immediately on save.
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Intraday MTF */}
          <div>
            <label className="text-[9px] text-gray-500 uppercase tracking-wide block mb-2">
              Intraday MTF (15min / 1H / 4H)
            </label>
            <div className="text-[9px] text-gray-600 mb-2">
              Used in pounce scanner for timeframe alignment (15m ▲ 1H ▲ 4H ▲ 1D scoring)
            </div>
            <div className="flex gap-2">
              {[
                { val: 'alphavantage', label: 'Alpha Vantage', req: 'Premium' },
                { val: 'yfinance',     label: 'yfinance',      req: 'Free' },
              ].map(opt => (
                <button
                  key={opt.val}
                  type="button"
                  onClick={() => setMtfSource(opt.val)}
                  className={`flex-1 rounded-lg py-2 px-2 border text-center transition-all ${
                    mtfSource === opt.val
                      ? 'bg-violet-700/30 border-violet-600 text-violet-200'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'
                  }`}
                >
                  <div className="text-[10px] font-bold">{opt.label}</div>
                  <div className="text-[9px] opacity-60">{opt.req}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Daily OHLCV */}
          <div>
            <label className="text-[9px] text-gray-500 uppercase tracking-wide block mb-2">
              Daily OHLCV
            </label>
            <div className="text-[9px] text-gray-600 mb-2">
              Used for daily price bars, MA calculations, backtest data
            </div>
            <div className="flex gap-2">
              {[
                { val: 'alphavantage', label: 'Alpha Vantage', req: 'Free or Premium' },
                { val: 'yfinance',     label: 'yfinance',      req: 'Free' },
              ].map(opt => (
                <button
                  key={opt.val}
                  type="button"
                  onClick={() => setDailySource(opt.val)}
                  className={`flex-1 rounded-lg py-2 px-2 border text-center transition-all ${
                    dailySource === opt.val
                      ? 'bg-violet-700/30 border-violet-600 text-violet-200'
                      : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'
                  }`}
                >
                  <div className="text-[10px] font-bold">{opt.label}</div>
                  <div className="text-[9px] opacity-60">{opt.req}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50 space-y-1.5">
          <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Active Routing</div>
          {[
            { label: 'MTF alignment (15m/1H/4H/1D)', source: mtfSource },
            { label: 'Daily bars (OHLCV)', source: dailySource },
            { label: 'Options chain', source: 'yahoo' },
          ].map(row => (
            <div key={row.label} className="flex items-center gap-2">
              <span className="text-[9px] text-gray-500 flex-1">{row.label}</span>
              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                row.source === 'alphavantage' ? 'text-amber-400 bg-amber-950/50' :
                row.source === 'yfinance'     ? 'text-blue-400 bg-blue-950/50'   :
                'text-gray-400 bg-gray-800'
              }`}>
                {row.source}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Save button (bottom) ── */}
      <div className="flex justify-end gap-3 pt-2">
        {saveMsg && (
          <span className={`text-[10px] font-bold self-center ${saveMsg.startsWith('✓') ? 'text-emerald-400' : 'text-red-400'}`}>
            {saveMsg}
          </span>
        )}
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-2.5 text-xs font-bold bg-emerald-700 hover:bg-emerald-600 disabled:bg-gray-700 text-white rounded-lg transition-colors"
        >
          {saving ? 'Saving…' : '💾 Save & Apply All Settings'}
        </button>
      </div>

    </div>
  )
}
