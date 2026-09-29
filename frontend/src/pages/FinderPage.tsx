// frontend/src/pages/FinderPage.tsx
//
// FINDER — the single scanner. Penny stocks, small caps, ETF long/short pairs,
// breakout/momentum signals, and crypto used to be four separate pages (EDGE,
// SIGNALS, TRADES, CRYPTO) plus this one. They are now presets on one screen:
// pick a preset, hit scan, see symbol/price/direction/trade-levels, click a
// row to open the chart. The user never has to know "fast DB scan" vs "live
// full-market scan" is two different backend paths — this page decides that
// per preset.
import { useEffect, useMemo, useRef, useState, lazy, Suspense } from 'react'

const KLineChartOverlay = lazy(() => import('../components/chart/KLineChartOverlay'))

// ── Backend shapes ───────────────────────────────────────────────────────────
// Verified by reading api/routers/finder.py, scanner_engine/finder.py, and
// api/routers/crypto_signals.py directly (no live backend was reachable from
// this environment — see worker report). Two shape bugs in the previous
// version of this page were caught this way and are fixed here:
//   1. /api/finder/presets returns a bare dict keyed by preset id
//      ({signal: {...}, pennyking: {...}, maker: {...}}), not {presets: [...]}.
//   2. /api/finder/health returns a bare dict keyed by table name
//      ({price_data: {...}, indicators: {...}, ...}), not {tables: [...]}.
//      There is also no `stale` field — staleness has to be derived from
//      latest_date / instruments_current / instruments_total.

interface FinderChecks {
  // Each key is present only if the backend actually evaluated it — e.g.
  // `breakout_ready` and `confluence` are omitted entirely (not `null`)
  // unless the preset's criteria asked for them. Missing key = not evaluated.
  price_band?: boolean
  liquidity?: boolean
  golden_cross?: boolean
  consolidating?: boolean
  no_gaps?: boolean
  breakout_ready?: boolean
  confluence?: boolean
}

interface FinderScanResult {
  symbol: string
  price: number
  volume: number
  checks: FinderChecks
  passed: number
  failed: number
  ma50: number | null
  ma200: number | null
  compression_score: number | null
  compression_duration: number | null
  gap_max_pct: number
  breakout_strength: number | null
  confluence: number | null
  // Trade levels derived from the matched consolidation. Null when there was
  // no consolidation to measure from — the backend does not invent them.
  entry: number | null
  stop: number | null
  target: number | null
  risk_reward: number | null
  range_high: number | null
  range_low: number | null
  range_mid: number | null
  distance_to_entry_pct: number | null
}

interface FinderScanResponse {
  count: number
  results: FinderScanResult[]
  stats?: {
    universe_size: number
    stage1_survivors: number
    deep_analyzed: number
    matched: number
    elapsed_seconds: number
    failed_downloads: number
  }
}

interface FinderHealthTable {
  latest_date: string | null
  instruments_current: number
  instruments_total: number
}
type FinderHealth = Record<string, FinderHealthTable>

interface CryptoTradeRec {
  symbol: string
  verdict: string        // BUY | SELL | HOLD | NO_TRADE
  direction: string       // LONG | SHORT | NO_TRADE
  score: number
  entry: number
  stop: number
  target: number
  risk_reward: number
  position_size: number
  rejection_reason?: string
}

interface CryptoScanResponse {
  error?: string
  symbols_scanned?: number
  recommendations?: CryptoTradeRec[]
}

// ── Leveraged ETF pairs ──────────────────────────────────────────────────────
// There is no backend endpoint for "ETFs long/short on an underlying" — per
// the task, this is transcribed from config/settings.py LEVERAGED_ETF_PAIRS
// (INDEX_LEVERAGED + SECTOR_LEVERAGED + COMMODITY_LEVERAGED +
// SINGLE_STOCK_LEVERAGED). Keep in sync manually if settings.py changes.

interface EtfPair {
  bull: string
  bear: string | null
  description: string
  underlying: string | null
}

const ETF_PAIRS: EtfPair[] = [
  // ── Index ──
  { bull: 'TQQQ', bear: 'SQQQ', description: 'NASDAQ-100 3x', underlying: 'QQQ' },
  { bull: 'UPRO', bear: 'SPXU', description: 'S&P 500 3x', underlying: 'SPY' },
  { bull: 'SSO', bear: 'SDS', description: 'S&P 500 2x', underlying: 'SPY' },
  { bull: 'UDOW', bear: 'SDOW', description: 'Dow Jones 3x', underlying: 'SPY' },
  { bull: 'TNA', bear: 'TZA', description: 'Russell 2000 3x', underlying: 'IWM' },
  { bull: 'WANT', bear: null, description: 'Consumer Disc 3x', underlying: 'XLY' },
  // ── Sector ──
  { bull: 'SOXL', bear: 'SOXS', description: 'Semiconductor 3x', underlying: 'SOXX' },
  { bull: 'FAS', bear: 'FAZ', description: 'Financial 3x', underlying: 'JPM' },
  { bull: 'TECL', bear: 'TECS', description: 'Technology 3x', underlying: 'QQQ' },
  { bull: 'CURE', bear: null, description: 'Healthcare 3x', underlying: 'XLV' },
  { bull: 'DFEN', bear: null, description: 'Aerospace & Defense 3x', underlying: 'ITA' },
  { bull: 'MIDU', bear: null, description: 'Mid-Cap 3x', underlying: 'MDY' },
  { bull: 'LABU', bear: 'LABD', description: 'Biotech 3x', underlying: 'XBI' },
  { bull: 'NAIL', bear: null, description: 'Homebuilders 3x', underlying: 'ITB' },
  { bull: 'WEBL', bear: 'WEBS', description: 'Online Retail 3x', underlying: 'FDN' },
  { bull: 'RETL', bear: null, description: 'Retail 3x', underlying: 'XRT' },
  // ── Commodity ──
  { bull: 'GUSH', bear: 'DRIP', description: 'Oil & Gas E&P 2x', underlying: null },
  { bull: 'BOIL', bear: 'KOLD', description: 'Natural Gas 2x', underlying: null },
  { bull: 'UGL', bear: 'GLL', description: 'Gold 2x', underlying: null },
  { bull: 'UCO', bear: 'SCO', description: 'Crude Oil 2x', underlying: null },
  { bull: 'AGQ', bear: 'ZSL', description: 'Silver 2x', underlying: null },
  // ── Single-stock ──
  { bull: 'MSTU', bear: 'MSTZ', description: 'MicroStrategy 2x', underlying: 'MSTR' },
  { bull: 'NVDL', bear: 'NVDS', description: 'NVIDIA 2x', underlying: 'NVDA' },
  { bull: 'NVDU', bear: 'NVD', description: 'NVIDIA 2x', underlying: 'NVDA' },
  { bull: 'TSLL', bear: 'TSLQ', description: 'Tesla 2x', underlying: 'TSLA' },
  { bull: 'AAPU', bear: 'AAPD', description: 'Apple 2x', underlying: 'AAPL' },
  { bull: 'AMZU', bear: 'AMZD', description: 'Amazon 2x', underlying: 'AMZN' },
  { bull: 'METU', bear: 'METD', description: 'Meta 2x', underlying: 'META' },
  { bull: 'GGLL', bear: 'GGLS', description: 'Google 2x', underlying: 'GOOGL' },
  { bull: 'MSFU', bear: 'MSFD', description: 'Microsoft 2x', underlying: 'MSFT' },
  { bull: 'AMDL', bear: 'AMDD', description: 'AMD 2x', underlying: 'AMD' },
  { bull: 'PLTU', bear: 'PLTZ', description: 'Palantir 2x', underlying: 'PLTR' },
  { bull: 'MUU', bear: 'MULL', description: 'Micron 2x', underlying: 'MU' },
  { bull: 'SMCX', bear: 'SMST', description: 'Super Micro 2x', underlying: 'SMCI' },
  { bull: 'IONX', bear: 'IONL', description: 'IonQ 2x', underlying: 'IONQ' },
  { bull: 'AVGX', bear: 'AVL', description: 'Broadcom 2x', underlying: 'AVGO' },
  { bull: 'ORCX', bear: null, description: 'Oracle 2x', underlying: 'ORCL' },
  { bull: 'MSFL', bear: null, description: 'Microsoft 2x', underlying: 'MSFT' },
  { bull: 'AMDG', bear: null, description: 'AMD 2x', underlying: 'AMD' },
  { bull: 'NFXL', bear: null, description: 'Netflix 2x', underlying: 'NFLX' },
  { bull: 'CONL', bear: null, description: 'Coinbase 2x', underlying: 'COIN' },
  { bull: 'MRAL', bear: null, description: 'Marathon Digital 2x', underlying: 'MARA' },
  { bull: 'BRKU', bear: null, description: 'Berkshire B 2x', underlying: 'BRK-B' },
  { bull: 'BABX', bear: null, description: 'Alibaba 2x', underlying: 'BABA' },
  { bull: 'RDTL', bear: null, description: 'Reddit 2x', underlying: 'RDDT' },
  { bull: 'TSMU', bear: 'TSMG', description: 'TSMC 2x', underlying: 'TSM' },
  { bull: 'CRWL', bear: null, description: 'CrowdStrike 2x', underlying: 'CRWD' },
  { bull: 'SOFX', bear: null, description: 'SoFi 2x', underlying: 'SOFI' },
  { bull: 'RKLX', bear: null, description: 'Rocket Lab 2x', underlying: 'RKLB' },
  { bull: 'UBRL', bear: null, description: 'Uber 2x', underlying: 'UBER' },
  { bull: 'DLLL', bear: null, description: 'Dell 2x', underlying: 'DELL' },
  { bull: 'NOWL', bear: null, description: 'ServiceNow 2x', underlying: 'NOW' },
  { bull: 'CRMG', bear: null, description: 'Salesforce 2x', underlying: 'CRM' },
  { bull: 'ARMG', bear: null, description: 'ARM Holdings 2x', underlying: 'ARM' },
  { bull: 'ADBG', bear: null, description: 'Adobe 2x', underlying: 'ADBE' },
  { bull: 'LLYX', bear: null, description: 'Eli Lilly 2x', underlying: 'LLY' },
  { bull: 'PYPG', bear: null, description: 'PayPal 2x', underlying: 'PYPL' },
  { bull: 'INTW', bear: null, description: 'Intel 2x', underlying: 'INTC' },
  { bull: 'ROBN', bear: 'HOOG', description: 'Hood 2x', underlying: 'HOOD' },
  { bull: 'SMCY', bear: null, description: 'YieldMax SMCI Strategy', underlying: 'SMCI' },
  { bull: 'SQQU', bear: null, description: 'Block (SQ) 2x', underlying: 'SQ' },
  { bull: 'AFRX', bear: null, description: 'Affirm 2x', underlying: 'AFRM' },
  { bull: 'QQQX', bear: null, description: 'Qualcomm 2x', underlying: 'QCOM' },
]

const ETF_SYMBOL_INFO = new Map<string, { pair: EtfPair; side: 'bull' | 'bear' }>()
for (const pair of ETF_PAIRS) {
  ETF_SYMBOL_INFO.set(pair.bull, { pair, side: 'bull' })
  if (pair.bear) ETF_SYMBOL_INFO.set(pair.bear, { pair, side: 'bear' })
}

// ── Unified row — one shape for both finder-scan and crypto-scan results, so
// there is a single row renderer regardless of which preset produced it. ────

interface Row {
  symbol: string
  price: number | null
  trendUp: boolean | null           // true = bullish trend, false = bearish, null = unknown
  entry: number | null
  stop: number | null
  target: number | null
  riskReward: number | null
  /** % price must move to reach entry. Negative means it is already through. */
  distanceToEntryPct: number | null
  positionSize: number | null
  compressionScore: number | null
  compressionDuration: number | null
  ma50: number | null
  ma200: number | null
  gapMaxPct: number | null
  volume: number | null
  checks: FinderChecks | null
  passed: number | null
  failed: number | null
  note: string | null               // pair relationship, rejection reason, etc — muted supporting text
}

function rowFromFinderResult(r: FinderScanResult, note: string | null = null): Row {
  return {
    symbol: r.symbol,
    price: r.price,
    trendUp: r.ma50 != null && r.ma200 != null ? r.ma50 >= r.ma200 : null,
    // Derived from the consolidation the screener matched on: entry at the
    // range high, stop at the low, target a range-height above. Absent only
    // when there is no consolidation to measure from — never invented.
    entry: r.entry ?? null,
    stop: r.stop ?? null,
    target: r.target ?? null,
    riskReward: r.risk_reward ?? null,
    distanceToEntryPct: r.distance_to_entry_pct ?? null,
    positionSize: null,
    compressionScore: r.compression_score,
    compressionDuration: r.compression_duration,
    ma50: r.ma50,
    ma200: r.ma200,
    gapMaxPct: r.gap_max_pct,
    volume: r.volume,
    checks: r.checks,
    passed: r.passed,
    failed: r.failed,
    note,
  }
}

function rowFromCryptoRec(rec: CryptoTradeRec): Row {
  const actionable = rec.verdict === 'BUY' || rec.verdict === 'SELL'
  return {
    symbol: rec.symbol,
    price: actionable ? rec.entry : null,
    trendUp: rec.direction === 'LONG' ? true : rec.direction === 'SHORT' ? false : null,
    entry: actionable ? rec.entry : null,
    stop: actionable ? rec.stop : null,
    target: actionable ? rec.target : null,
    riskReward: actionable ? rec.risk_reward : null,
    // Crypto recommendations quote entry at the current price, so there is no
    // gap to the trigger to report.
    distanceToEntryPct: null,
    positionSize: actionable ? rec.position_size : null,
    compressionScore: null,
    compressionDuration: null,
    ma50: null,
    ma200: null,
    gapMaxPct: null,
    volume: null,
    checks: null,
    passed: null,
    failed: null,
    note: !actionable ? (rec.rejection_reason || rec.verdict) : null,
  }
}

// ── Presets — plain language, not backend jargon ────────────────────────────
// Each preset owns which scan path it needs; the user only ever sees "scan".

type ScanMode = 'fast' | 'wide' | 'etf' | 'crypto'

interface UiPreset {
  id: string
  label: string
  hint: string
  mode: ScanMode
  backendPreset?: string   // signal | pennyking | maker
}

const UI_PRESETS: UiPreset[] = [
  {
    id: 'penny',
    label: 'Penny stocks',
    hint: 'Under $5, with real liquidity',
    mode: 'wide',
    backendPreset: 'pennyking',
  },
  {
    id: 'small',
    label: 'Small caps',
    hint: '$5–20 band',
    mode: 'wide',
    backendPreset: 'maker',
  },
  {
    id: 'etf',
    label: 'ETFs — long / short',
    hint: 'Leveraged bull/bear pairs on an underlying',
    mode: 'etf',
  },
  {
    id: 'breakout',
    label: 'Breakout & momentum',
    hint: 'Golden cross, consolidation, clean price action',
    mode: 'fast',
    backendPreset: 'signal',
  },
  {
    id: 'crypto',
    label: 'Crypto',
    hint: 'Top crypto by market cap',
    mode: 'crypto',
  },
]

// ── Formatting helpers ──────────────────────────────────────────────────────

function fmtPrice(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  const digits = n >= 1 ? 2 : n >= 0.01 ? 4 : 6
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function fmtNum(n: number | null | undefined, digits = 1): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function fmtVolume(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function daysSince(dateStr: string): number {
  return (Date.now() - new Date(dateStr + 'T00:00:00').getTime()) / 86_400_000
}

function isStaleTable(t: FinderHealthTable): boolean {
  if (t.latest_date == null) return true
  if (daysSince(t.latest_date) > 5) return true
  if (t.instruments_total > 0 && t.instruments_current / t.instruments_total < 0.5) return true
  return false
}

// ── Small presentational pieces ─────────────────────────────────────────────

// Direction is PRICE/TREND direction only, per the design system — never used
// for pass/fail. Bullish trend (ma50 >= ma200, or crypto LONG) = --up,
// bearish = --down, unknown = dimmed dash (never colored).
function TrendMark({ up }: { up: boolean | null }) {
  if (up === null) {
    return <span className="text-[color:var(--text-muted)]">–</span>
  }
  return (
    <span className={up ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>
      {up ? '▲' : '▼'}
    </span>
  )
}

// Checks: passed and failed are both subtle (same weight) — colour never
// encodes pass/fail, only the glyph does. Not-evaluated is dimmed further and
// uses a dash, never red, so an unfilled criterion cannot read as "failed."
function CheckMark({ v }: { v: boolean | undefined }) {
  if (v === undefined) {
    return <span className="text-[color:var(--text-muted)] opacity-50">–</span>
  }
  return <span className="text-[color:var(--text-secondary)]">{v ? '✓' : '✕'}</span>
}

const CHECK_DEFS: { key: keyof FinderChecks; label: string }[] = [
  { key: 'price_band', label: 'Price band' },
  { key: 'liquidity', label: 'Liquidity' },
  { key: 'golden_cross', label: 'Golden cross' },
  { key: 'consolidating', label: 'Consolidating' },
  { key: 'no_gaps', label: 'No gaps' },
  { key: 'breakout_ready', label: 'Breakout ready' },
  { key: 'confluence', label: 'Confluence' },
]

function TradeStat({ label, value, tone, suffix }: { label: string; value: number | null; tone?: 'up' | 'down'; suffix?: string }) {
  const color = tone === 'up' ? 'text-[color:var(--up)]' : tone === 'down' ? 'text-[color:var(--down)]' : 'text-[color:var(--text-primary)]'
  return (
    <div>
      <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{label}</div>
      <div className={`num text-[length:var(--text-base)] font-[var(--weight-medium)] ${color}`}>
        {value == null ? '—' : suffix ? `${value.toFixed(1)}${suffix}` : `$${fmtPrice(value)}`}
      </div>
    </div>
  )
}

// ── Result row ───────────────────────────────────────────────────────────────

function ResultRow({ row, onOpenChart }: { row: Row; onOpenChart: (symbol: string) => void }) {
  const [expanded, setExpanded] = useState(false)
  const hasTrade = row.entry != null || row.stop != null || row.target != null
  const hasChecks = row.checks != null && row.passed != null

  return (
    <div
      className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] hover:bg-[color:var(--bg-raised)] transition-colors cursor-pointer px-4 py-3"
      onClick={() => onOpenChart(row.symbol)}
      role="button"
      tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpenChart(row.symbol) } }}
      aria-label={`Open ${row.symbol} chart`}
    >
      {/* Header: symbol (large) + price + trend */}
      <div className="flex items-baseline justify-between gap-4">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="num text-[length:var(--text-lg)] font-[var(--weight-bold)] text-[color:var(--text-primary)]">
            {row.symbol}
          </span>
        </div>
        <div className="flex items-baseline gap-2 shrink-0">
          <span className="num text-[length:var(--text-xl)] text-[color:var(--text-primary)]">
            {row.price == null ? '—' : `$${fmtPrice(row.price)}`}
          </span>
          <TrendMark up={row.trendUp} />
        </div>
      </div>

      {row.note && (
        <div className="mt-0.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{row.note}</div>
      )}

      {/* THE TRADE — entry / stop / target / R:R, sized to be readable at a glance */}
      {hasTrade ? (
        <div className="mt-3 grid grid-cols-4 gap-3 max-w-md">
          <TradeStat label="Entry" value={row.entry} />
          <TradeStat label="Stop" value={row.stop} tone="down" />
          <TradeStat label="Target" value={row.target} tone="up" />
          <TradeStat label="R:R" value={row.riskReward} suffix=":1" />
        </div>
      ) : (
        <div className="mt-2 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          No trade levels — screener match only
        </div>
      )}

      {/* Supporting analysis — small and muted, never the focal point */}
      {(row.compressionScore != null || row.ma50 != null || row.gapMaxPct != null || row.volume != null || hasChecks) && (
        <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          {row.compressionScore != null && (
            <span>compression {fmtNum(row.compressionScore)}{row.compressionDuration != null ? ` / ${row.compressionDuration}d` : ''}</span>
          )}
          {row.ma50 != null && row.ma200 != null && (
            <span className="num">MA50 {fmtNum(row.ma50, 2)} · MA200 {fmtNum(row.ma200, 2)}</span>
          )}
          {row.gapMaxPct != null && <span>max gap {fmtNum(row.gapMaxPct)}%</span>}
          {row.volume != null && <span>vol {fmtVolume(row.volume)}</span>}
          {hasChecks && (
            <button
              onClick={e => { e.stopPropagation(); setExpanded(x => !x) }}
              className="text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)] transition-colors"
            >
              {expanded ? '▾' : '▸'} {row.passed}/{(row.passed ?? 0) + (row.failed ?? 0)} checks
            </button>
          )}
        </div>
      )}

      {expanded && row.checks && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[length:var(--text-xs)] text-[color:var(--text-secondary)]">
          {CHECK_DEFS.map(c => (
            <span key={c.key} className="inline-flex items-center gap-1">
              <CheckMark v={row.checks?.[c.key]} /> {c.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Empty state ──────────────────────────────────────────────────────────────

function EmptyState({ preset, staleTables, hasHealthInfo }: { preset: UiPreset; staleTables: [string, FinderHealthTable][]; hasHealthInfo: boolean }) {
  let headline: string
  let sub: string

  if (preset.mode === 'crypto') {
    headline = 'Scan ran, no crypto setups matched.'
    sub = 'Crypto conditions move fast — try again shortly or with a different preset.'
  } else if (staleTables.length > 0) {
    headline = 'No current data, not "no matches."'
    sub = `${staleTables.map(([name]) => name).join(', ')} ${staleTables.length === 1 ? 'is' : 'are'} stale — the scan ran against whatever data exists. Check the freshness banner above.`
  } else if (hasHealthInfo) {
    headline = 'Scan ran, nothing matched.'
    sub = 'Underlying data is current — try a different preset.'
  } else {
    headline = 'Scan ran, nothing matched.'
    sub = 'Freshness could not be confirmed — an empty result may also reflect stale data.'
  }

  return (
    <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-8 text-center">
      <p className="text-[length:var(--text-base)] text-[color:var(--text-secondary)]">{headline}</p>
      <p className="mt-1 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{sub}</p>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function FinderPage() {
  const [activePresetId, setActivePresetId] = useState<string>(UI_PRESETS[0].id)
  const activePreset = UI_PRESETS.find(p => p.id === activePresetId) ?? UI_PRESETS[0]

  const [results, setResults] = useState<Row[] | null>(null)
  const [scanMeta, setScanMeta] = useState<string | null>(null)
  const [scanningMode, setScanningMode] = useState<ScanMode | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [health, setHealth] = useState<FinderHealth>({})
  const [healthError, setHealthError] = useState<string | null>(null)

  const [chartSymbol, setChartSymbol] = useState<string | null>(null)

  // Guards against a stale response landing after a newer scan started.
  const requestId = useRef(0)

  useEffect(() => {
    fetch('/api/finder/health')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: FinderHealth) => setHealth(d ?? {}))
      .catch(err => setHealthError(err instanceof Error ? err.message : String(err)))
  }, [])

  const staleTables = useMemo(
    () => Object.entries(health).filter(([, t]) => isStaleTable(t)),
    [health],
  )
  const hasHealthInfo = Object.keys(health).length > 0

  async function runScan(preset: UiPreset) {
    setActivePresetId(preset.id)
    const myRequest = ++requestId.current
    setScanningMode(preset.mode)
    setError(null)
    setResults(null)
    setScanMeta(null)

    try {
      if (preset.mode === 'crypto') {
        const res = await fetch('/api/crypto-signals/scan?profile=SWING&account=10000&max_symbols=30')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const d: CryptoScanResponse = await res.json()
        if (d.error) throw new Error(d.error)
        if (myRequest !== requestId.current) return
        const recs = d.recommendations ?? []
        setResults(recs.map(rowFromCryptoRec))
        setScanMeta(`${d.symbols_scanned ?? recs.length} scanned`)
        return
      }

      if (preset.mode === 'etf') {
        // No dedicated endpoint — run the fast screener with the filters
        // loosened (we want every tracked leveraged ETF, not just the ones
        // currently golden-crossed/consolidating), then keep only symbols
        // that appear in the leveraged-ETF pair table.
        const params = new URLSearchParams({
          limit: '500',
          require_golden_cross: 'false',
          require_consolidation: 'false',
          min_volume: '0',
          max_gap_pct: '100',
        })
        const res = await fetch(`/api/finder/scan?${params}`)
        if (!res.ok) {
          const body = await res.json().catch(() => null)
          throw new Error(body?.detail ?? `HTTP ${res.status}`)
        }
        const d: FinderScanResponse = await res.json()
        if (myRequest !== requestId.current) return
        const rows = (d.results ?? [])
          .filter(r => ETF_SYMBOL_INFO.has(r.symbol))
          .map(r => {
            const info = ETF_SYMBOL_INFO.get(r.symbol)!
            const opposite = info.side === 'bull' ? info.pair.bear : info.pair.bull
            const bias = info.side === 'bull' ? 'Long side' : 'Short side'
            const underlyingNote = info.pair.underlying ? ` on ${info.pair.underlying}` : ''
            const oppositeNote = opposite ? ` · opposite: ${opposite}` : ''
            return rowFromFinderResult(r, `${bias} · ${info.pair.description}${underlyingNote}${oppositeNote}`)
          })
        setResults(rows)
        setScanMeta(`${rows.length} of ${ETF_PAIRS.length} tracked pairs matched`)
        return
      }

      // fast | wide — same shape, different endpoint + timing.
      const endpoint = preset.mode === 'wide' ? '/api/finder/scan/wide' : '/api/finder/scan'
      const params = new URLSearchParams({ preset: preset.backendPreset ?? '', limit: '150' })
      if (preset.mode === 'wide') params.set('max_deep', '400')
      const res = await fetch(`${endpoint}?${params}`)
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? `HTTP ${res.status}`)
      }
      const d: FinderScanResponse = await res.json()
      if (myRequest !== requestId.current) return
      setResults((d.results ?? []).map(r => rowFromFinderResult(r)))
      setScanMeta(
        d.stats
          ? `${d.count} matches · ${d.stats.deep_analyzed}/${d.stats.universe_size} analyzed · ${d.stats.elapsed_seconds}s`
          : `${d.count} matches`,
      )
    } catch (err) {
      if (myRequest !== requestId.current) return
      setError(err instanceof Error ? err.message : String(err))
      setResults(null)
      setScanMeta(null)
    } finally {
      if (myRequest === requestId.current) setScanningMode(null)
    }
  }

  const isScanning = scanningMode !== null
  const showHealthBanner = activePreset.mode !== 'crypto'

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-[length:var(--text-xl)] font-[var(--weight-bold)] text-[color:var(--text-primary)] tracking-tight">FINDER</h1>
        <p className="mt-0.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          Pick what you're looking for. Symbol, price, and the trade — entry, stop, target — when there is one.
        </p>
      </div>

      {/* Freshness banner — only meaningful for presets backed by the DB tables */}
      {showHealthBanner && staleTables.length > 0 && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-raised)] border border-[color:var(--line-strong)] p-3">
          <div className="flex items-start gap-2">
            <span className="text-[color:var(--accent)] text-[length:var(--text-sm)]">⚠</span>
            <div className="flex-1 min-w-0">
              <div className="text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--accent)] tracking-wide">
                DATA MAY BE STALE
              </div>
              <div className="mt-1 space-y-0.5">
                {staleTables.map(([name, t]) => (
                  <div key={name} className="num text-[length:var(--text-xs)] text-[color:var(--text-secondary)]">
                    {name}: latest {t.latest_date ?? 'unknown'} · {t.instruments_current}/{t.instruments_total} instruments current
                  </div>
                ))}
              </div>
              <p className="mt-1.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
                An empty result below may mean the underlying table has no current data — not that nothing matched.
              </p>
            </div>
          </div>
        </div>
      )}
      {showHealthBanner && healthError && (
        <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          Freshness check unavailable ({healthError}) — results below cannot be cross-checked against data currency.
        </div>
      )}

      {/* Preset picker — plain language */}
      <div className="flex flex-wrap gap-2">
        {UI_PRESETS.map(p => {
          const active = activePresetId === p.id
          return (
            <button
              key={p.id}
              onClick={() => runScan(p)}
              disabled={isScanning}
              className={`text-left px-4 py-2.5 rounded-[var(--radius-lg)] transition-colors disabled:opacity-50 ${
                active ? 'bg-[color:var(--bg-raised)]' : 'bg-[color:var(--bg-panel)] hover:bg-[color:var(--bg-raised)]'
              }`}
            >
              <div className={`text-[length:var(--text-sm)] font-[var(--weight-bold)] ${active ? 'text-[color:var(--accent)]' : 'text-[color:var(--text-primary)]'}`}>
                {p.label}
              </div>
              <div className="mt-0.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{p.hint}</div>
            </button>
          )
        })}
      </div>

      {/* Scan status */}
      {isScanning && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] px-4 py-3 flex items-center gap-3">
          <div className="w-4 h-4 border-2 border-[color:var(--line-strong)] border-t-[color:var(--accent)] rounded-full animate-spin shrink-0" />
          <span className="text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">
            {scanningMode === 'wide'
              ? 'Scanning the full market — live data over the whole US ticker universe. Usually 30–60 seconds.'
              : scanningMode === 'crypto'
                ? 'Scanning crypto markets…'
                : 'Scanning…'}
          </span>
        </div>
      )}
      {!isScanning && scanMeta && !error && (
        <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{scanMeta}</div>
      )}

      {error && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] border border-[color:var(--line-strong)] p-3 text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">
          {error}
        </div>
      )}

      {/* Results */}
      {results !== null && !error && (
        results.length === 0 ? (
          <EmptyState preset={activePreset} staleTables={showHealthBanner ? staleTables : []} hasHealthInfo={showHealthBanner && hasHealthInfo} />
        ) : (
          <div className="space-y-2">
            {results.map(row => (
              <ResultRow key={row.symbol} row={row} onOpenChart={setChartSymbol} />
            ))}
          </div>
        )
      )}

      {results === null && !error && !isScanning && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-8 text-center">
          <p className="text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">Pick a preset above to scan.</p>
        </div>
      )}

      {/* Deep-analysis chart — same lazy-load pattern as TradesPage.tsx */}
      {chartSymbol && (
        <Suspense
          fallback={
            <div className="fixed inset-0 z-50 bg-[color:var(--bg-base)] flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="w-6 h-6 border-2 border-[color:var(--line-strong)] border-t-[color:var(--accent)] rounded-full animate-spin" />
                <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">Loading {chartSymbol} chart…</div>
              </div>
            </div>
          }
        >
          <KLineChartOverlay symbol={chartSymbol} onClose={() => setChartSymbol(null)} />
        </Suspense>
      )}
    </div>
  )
}
