import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSignals, useCommandCenter } from '../api/hooks'
import type { Signal } from '../types'
import MarketStatePanel      from '../components/command/MarketStatePanel'
import OpportunityRadarPanel from '../components/command/OpportunityRadarPanel'
import MarketOverviewBanner  from '../components/dashboard/MarketOverviewBanner'

// ── Trade classification ──────────────────────────────────────────────────────
const SWING_PHASES  = new Set(['Accumulation', 'Early Bull', 'Momentum Bull'])
const AVOID_PHASES  = new Set(['Distribution', 'Early Bear', 'Capitulation'])

// Phase quality for composite conviction (0–100)
const PHASE_QUALITY: Record<string, number> = {
  'Accumulation': 100, 'Early Bull': 95, 'Recovery': 80, 'Momentum Bull': 70,
  'Late Bull': 55, 'Distribution': 20, 'Early Bear': 12,
  'Late Bear': 8, 'Markdown': 5, 'Capitulation': 10,
}

function convictionScore(s: Signal): number {
  const phaseQ = PHASE_QUALITY[s.market_phase] ?? 50
  return 0.60 * (s.probability * 100) + 0.40 * phaseQ
}

function convictionLabel(score: number): { label: string; color: string } {
  if (score >= 78) return { label: 'HIGH', color: '#22c55e' }
  if (score >= 60) return { label: 'MEDIUM', color: '#eab308' }
  return { label: 'LOW', color: '#ef4444' }
}

// ── Asset universe classification ─────────────────────────────────────────────
// BTC / ETH / XRP spot ETFs + leveraged crypto ETFs (no crypto stocks/miners)
const CRYPTO_ETF_SYMBOLS = new Set([
  'IBIT','FBTC','ARKB','GBTC',          // Bitcoin spot ETFs
  'ETHA','FETH',                         // Ethereum spot ETFs
  'XRPI',                                // XRP ETF
  'BITU','BITI',                         // Leveraged/inverse Bitcoin ETFs
])

// All leveraged sector/index ETFs
const LEVERAGED_ETF_SYMBOLS = new Set([
  // Index leveraged
  'TQQQ','SQQQ','UPRO','SPXU','SSO','SDS','UDOW','SDOW','TNA','TZA','WANT',
  // Sector leveraged
  'SOXL','SOXS','FAS','FAZ','TECL','TECS','CURE','DFEN','MIDU','LABU','LABD','NAIL','WEBL','WEBS','RETL',
  // Commodity leveraged
  'GUSH','DRIP','BOIL','KOLD','UGL','GLL','UCO','SCO','AGQ','ZSL',
  // Index drivers
  'QQQ','SPY','IWM','SOXX','DIA','MDY','XLK','XLF','XLE','XLV',
])

// Single-stock leveraged ETFs
const SS_LEVERAGED_SYMBOLS = new Set([
  'MSTU','MSTZ','NVDL','NVDS','NVDU','NVD','TSLL','TSLQ','AAPU','AAPD',
  'AMZU','AMZD','METU','METD','GGLL','GGLS','MSFU','MSFD','AMDL','AMDD',
  'PLTU','PLTZ','MUU','MULL','SMCX','SMST','IONX','IONL','AVGX','AVL',
  'ORCX','MSFL','AMDG','NFXL','CONL','MRAL','BRKU','BABX','RDTL',
  'TSMU','TSMG','CRWL','SOFX','RKLX','UBRL','DLLL','NOWL','CRMG','ARMG',
  'ADBG','LLYX','PYPG','INTW','ROBN','HOOG','SMCY','SQQU','AFRX','QQQX',
])

// S&P 500 / large-cap stocks
const SP500_SYMBOLS = new Set([
  'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO',
  'JPM','V','MA','UNH','HD','PG','COST','MRK',
  'AMD','ORCL','CRM','NFLX','PANW','SNOW','PLTR','SHOP',
  'LLY','PYPL','ADBE','ARM','INTC','NOW','DELL','BABA',
  'XOM','CVX','COP','WMT','KO','PEP','NKE','MCD','DIS','SBUX',
  'BAC','GS','MS','BLK','AXP','WFC','C',
  'JNJ','ABBV','AMGN','BMY','REGN','GILD','CVS','MDT',
  'GE','CAT','HON','RTX','BA','UPS','FDX','DE',
  'QCOM','TXN','IBM','AMAT','LRCX','KLAC','ADI','MRVL',
  'SPGI','MCO','ICE','CME','NEE','T','VZ','AMT','PLD',
  'TSM','MU','SMCI',
])

// Russell 2000 / small-cap stocks
const RUSSELL2000_SYMBOLS = new Set([
  'ACHR','JOBY','LUNR','RDW','RKLB',
  'RIOT','MARA','HUT','IREN','CIFR','CLBT',
  'AI','BBAI','RXRX','DOCN','IONQ',
  'SQ','AFRM','UPST','HOOD','SOFI','RDDT',
  'SMAR','BEAM','EDIT','NTLA',
  'CELH','AXON','CFLT','DDOG','MDB',
  'COIN','MSTR','UBER',
])

type AssetCategory = 'crypto_etf' | 'leveraged_etf' | 'ss_leveraged' | 'sp500' | 'russell2000' | 'other'

function getCategory(symbol: string): AssetCategory {
  if (CRYPTO_ETF_SYMBOLS.has(symbol))    return 'crypto_etf'
  if (LEVERAGED_ETF_SYMBOLS.has(symbol)) return 'leveraged_etf'
  if (SS_LEVERAGED_SYMBOLS.has(symbol))  return 'ss_leveraged'
  if (SP500_SYMBOLS.has(symbol))         return 'sp500'
  if (RUSSELL2000_SYMBOLS.has(symbol))   return 'russell2000'
  return 'other'
}

function isETF(symbol: string): boolean {
  return CRYPTO_ETF_SYMBOLS.has(symbol) || LEVERAGED_ETF_SYMBOLS.has(symbol) || SS_LEVERAGED_SYMBOLS.has(symbol)
}

// Keep SECTOR_SYMBOLS for swing trade logic (broad market ETFs only)
const SECTOR_SYMBOLS = new Set([...LEVERAGED_ETF_SYMBOLS])

function isSwingTrade(s: Signal): boolean {
  return (SECTOR_SYMBOLS.has(s.symbol) || s.probability >= 0.82) && SWING_PHASES.has(s.market_phase)
}

function getAction(s: Signal): { label: string; color: string; bg: string } {
  const p = s.probability
  if (AVOID_PHASES.has(s.market_phase)) return { label: 'AVOID',      color: '#ef4444', bg: '#450a0a' }
  if (p >= 0.97)                         return { label: 'STRONG BUY', color: '#4ade80', bg: '#052e16' }
  if (p >= 0.90)                         return { label: 'BUY',        color: '#22c55e', bg: '#14532d' }
  if (p >= 0.80)                         return { label: 'SCALE IN',   color: '#86efac', bg: '#052e16' }
  if (p >= 0.70)                         return { label: 'SETUP',      color: '#fbbf24', bg: '#422006' }
  return                                        { label: 'WATCH',      color: '#6b7280', bg: '#1f2937' }
}

function getTradeType(s: Signal): { label: string; sublabel: string; color: string } {
  const cat = getCategory(s.symbol)
  if (AVOID_PHASES.has(s.market_phase))
    return { label: 'AVOID', sublabel: 'do not enter', color: '#ef4444' }
  if (cat === 'crypto_etf')
    return { label: 'CRYPTO ETF', sublabel: 'Bitcoin/Ethereum ETF · volatile', color: '#f59e0b' }
  if (cat === 'leveraged_etf' && SWING_PHASES.has(s.market_phase) && s.probability >= 0.75)
    return { label: 'ETF SWING', sublabel: 'sector/index rotation · days–weeks', color: '#a78bfa' }
  if (cat === 'ss_leveraged')
    return { label: 'SS LEVERAGED', sublabel: '2x single-stock · tight stop', color: '#60a5fa' }
  if (cat === 'sp500' && SWING_PHASES.has(s.market_phase) && s.probability >= 0.80)
    return { label: 'S&P 500 SWING', sublabel: 'large-cap swing · scale in', color: '#34d399' }
  if (cat === 'russell2000')
    return { label: 'SMALL-CAP', sublabel: 'high-momentum · small size', color: '#fb923c' }
  if (SWING_PHASES.has(s.market_phase) && s.probability >= 0.80)
    return { label: 'SWING', sublabel: 'hold days–weeks · trail stop', color: '#34d399' }
  if (s.probability >= 0.65)
    return { label: 'SETUP', sublabel: 'building conviction · watch', color: '#fbbf24' }
  return { label: 'WATCH', sublabel: 'not yet actionable', color: '#4b5563' }
}

// ── Sub-components ────────────────────────────────────────────────────────────
function ProbBar({ value }: { value: number }) {
  const pct = value * 100
  const color = pct >= 90 ? '#4ade80' : pct >= 80 ? '#22c55e' : pct >= 70 ? '#fbbf24' : '#9ca3af'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono font-bold" style={{ color }}>{pct.toFixed(1)}%</span>
    </div>
  )
}

function ActionBadge({ action }: { action: ReturnType<typeof getAction> }) {
  return (
    <span className="px-2 py-0.5 rounded text-xs font-black tracking-wider whitespace-nowrap"
      style={{ color: action.color, backgroundColor: action.bg, border: `1px solid ${action.color}40` }}>
      {action.label}
    </span>
  )
}

// ── Trade card ────────────────────────────────────────────────────────────────
function TradeCard({ s, onClick }: { s: Signal; onClick: () => void }) {
  const action    = getAction(s)
  const tradeType = getTradeType(s)
  const pct       = (s.probability * 100).toFixed(1)
  const upside    = s.sell_price > s.buy_price
    ? (((s.sell_price - s.buy_price) / s.buy_price) * 100).toFixed(1) : null
  const conv      = convictionLabel(convictionScore(s))

  return (
    <div
      onClick={onClick}
      className="bg-gray-900 border rounded-xl p-4 cursor-pointer transition-all hover:scale-[1.01]"
      style={{ borderColor: action.color + '40' }}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xl font-black text-white">{s.symbol}</span>
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded"
              style={{ color: tradeType.color, backgroundColor: tradeType.color + '18' }}>
              {tradeType.label}
            </span>
          </div>
          <span className="text-[11px] text-gray-500">{tradeType.sublabel}</span>
        </div>
        <div className="text-right">
          <div className="text-2xl font-black" style={{ color: action.color }}>{pct}%</div>
          <div className="flex gap-1 justify-end mt-1">
            <span className="text-[9px] font-black px-1.5 py-0.5 rounded border"
              style={{ color: conv.color, borderColor: conv.color + '60', backgroundColor: conv.color + '15' }}>
              {conv.label} CONVICTION
            </span>
          </div>
          <div className="mt-1"><ActionBadge action={action} /></div>
        </div>
      </div>

      <div className="text-[10px] text-gray-600 mb-2">{s.market_phase}</div>

      <div className="grid grid-cols-4 gap-1 text-center">
        {[
          { label: 'ENTRY',  value: s.buy_price,        color: '#f3f4f6' },
          { label: 'ADD',    value: s.accumulate_price,  color: '#60a5fa' },
          { label: 'T1',     value: s.scale_price,       color: '#34d399' },
          { label: 'TARGET', value: s.sell_price,        color: '#fbbf24' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-gray-800/60 rounded p-2">
            <div className="text-[9px] text-gray-500 mb-0.5">{label}</div>
            <div className="text-xs font-bold font-mono" style={{ color }}>${value.toFixed(2)}</div>
          </div>
        ))}
      </div>

      {upside && (
        <div className="mt-2 text-[10px] text-gray-600 text-right">
          +{upside}% to target
        </div>
      )}
    </div>
  )
}

// ── Section header ────────────────────────────────────────────────────────────
function SectionHeader({ label, sublabel, color, count }: { label: string; sublabel: string; color: string; count: number }) {
  return (
    <div className="flex items-baseline gap-3 mb-3">
      <h3 className="text-base font-black tracking-tight" style={{ color }}>{label}</h3>
      <span className="text-xs text-gray-600">{sublabel}</span>
      <span className="ml-auto text-xs text-gray-600">{count} setups</span>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
type SortKey = 'probability' | 'symbol'

export default function DashboardPage() {
  const { data: signals, isLoading } = useSignals()
  const { data: cmdData } = useCommandCenter()
  const navigate = useNavigate()
  const [tierFilter, setTierFilter] = useState<string>('ALL')
  const [sort, setSort] = useState<SortKey>('probability')

  const all = signals || []
  const latestDate = all.length
    ? all.reduce((max, s) => s.date > max ? s.date : max, all[0].date) : null

  // Composite conviction buckets (not DB tier — DB tier is stale from 2020)
  const high   = all.filter(s => convictionScore(s) >= 78).length
  const medium = all.filter(s => { const c = convictionScore(s); return c >= 60 && c < 78 }).length
  const low    = all.filter(s => convictionScore(s) < 60).length

  const actionable = all.filter(s => !AVOID_PHASES.has(s.market_phase) && s.probability >= 0.70)

  // Sector / Index ETF swings (leveraged index + sector, not single-stock)
  const sectorEtfTrades = actionable
    .filter(s => LEVERAGED_ETF_SYMBOLS.has(s.symbol) && SWING_PHASES.has(s.market_phase))
    .sort((a, b) => b.probability - a.probability).slice(0, 6)

  // Crypto ETF setups
  const cryptoTrades = actionable
    .filter(s => CRYPTO_ETF_SYMBOLS.has(s.symbol))
    .sort((a, b) => b.probability - a.probability).slice(0, 4)

  // S&P 500 stock setups
  const sp500Trades = actionable
    .filter(s => SP500_SYMBOLS.has(s.symbol))
    .sort((a, b) => b.probability - a.probability).slice(0, 6)

  // Russell 2000 / small-cap setups
  const russell2000Trades = actionable
    .filter(s => RUSSELL2000_SYMBOLS.has(s.symbol))
    .sort((a, b) => b.probability - a.probability).slice(0, 4)

  // Single-stock leveraged ETF setups (day trade / short swing)
  const ssLeveragedTrades = actionable
    .filter(s => SS_LEVERAGED_SYMBOLS.has(s.symbol))
    .sort((a, b) => b.probability - a.probability).slice(0, 6)

  // Table — filter by composite conviction
  const filtered = tierFilter === 'ALL'
    ? all
    : tierFilter === 'HIGH'
      ? all.filter(s => convictionScore(s) >= 78)
      : tierFilter === 'MEDIUM'
        ? all.filter(s => { const c = convictionScore(s); return c >= 60 && c < 78 })
        : all.filter(s => convictionScore(s) < 60)

  const sorted = [...filtered].sort((a, b) =>
    sort === 'probability' ? b.probability - a.probability : a.symbol.localeCompare(b.symbol)
  )

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight">TRADES NOW</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            {latestDate ? `Latest: ${latestDate}` : 'Loading...'} · Click any trade to open chart
          </p>
        </div>
        <div className="text-right text-xs flex items-center gap-1 flex-wrap justify-end">
          {([
            { tier: 'HIGH',   count: high,   color: '#22c55e' },
            { tier: 'MEDIUM', count: medium, color: '#eab308' },
            { tier: 'LOW',    count: low,    color: '#9ca3af' },
          ] as const).map(({ tier, count, color }) => (
            <button
              key={tier}
              onClick={() => {
                setTierFilter(tier)
                document.getElementById('signals-table')?.scrollIntoView({ behavior: 'smooth' })
              }}
              className="px-2 py-0.5 rounded hover:bg-gray-800 transition-colors"
            >
              <span className="font-bold" style={{ color }}>{count}</span>
              <span className="text-gray-600 ml-1">{tier === 'MEDIUM' ? 'MED' : tier}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Market Overview ── */}
      <MarketOverviewBanner />

      {/* ── Market State ── */}
      {cmdData?.market_state_summary && (
        <MarketStatePanel data={cmdData.market_state_summary} />
      )}

      {/* ── Opportunity Radar ── */}
      {cmdData?.opportunity_radar && (
        <OpportunityRadarPanel
          instruments={cmdData.opportunity_radar.instruments ?? []}
          earlyStageCount={cmdData.opportunity_radar.early_stage_count ?? 0}
          error={cmdData.opportunity_radar.error}
        />
      )}

      {/* ── Sector & Index ETF Swings ── */}
      {!isLoading && sectorEtfTrades.length > 0 && (
        <div>
          <SectionHeader label="SECTOR & INDEX ETFs" sublabel="leveraged ETF rotation · hold days–weeks · 1D/1W TF" color="#a78bfa" count={sectorEtfTrades.length} />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {sectorEtfTrades.map(s => <TradeCard key={s.symbol+'-etf'} s={s} onClick={() => navigate(`/chart?symbol=${s.symbol}`)} />)}
          </div>
        </div>
      )}

      {/* ── Crypto ETFs ── */}
      {!isLoading && cryptoTrades.length > 0 && (
        <div>
          <SectionHeader label="CRYPTO ETFs" sublabel="spot BTC/ETH ETFs · volatile · size smaller" color="#f59e0b" count={cryptoTrades.length} />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            {cryptoTrades.map(s => <TradeCard key={s.symbol+'-crypto'} s={s} onClick={() => navigate(`/chart?symbol=${s.symbol}`)} />)}
          </div>
        </div>
      )}

      {/* ── S&P 500 Stocks ── */}
      {!isLoading && sp500Trades.length > 0 && (
        <div>
          <SectionHeader label="S&P 500 STOCKS" sublabel="large-cap · swing & position trades · scale in on dips" color="#34d399" count={sp500Trades.length} />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {sp500Trades.map(s => <TradeCard key={s.symbol+'-sp500'} s={s} onClick={() => navigate(`/chart?symbol=${s.symbol}`)} />)}
          </div>
        </div>
      )}

      {/* ── Russell 2000 / Small-cap ── */}
      {!isLoading && russell2000Trades.length > 0 && (
        <div>
          <SectionHeader label="SMALL-CAP / RUSSELL 2000" sublabel="high-momentum · higher risk · smaller position size" color="#fb923c" count={russell2000Trades.length} />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            {russell2000Trades.map(s => <TradeCard key={s.symbol+'-r2k'} s={s} onClick={() => navigate(`/chart?symbol=${s.symbol}`)} />)}
          </div>
        </div>
      )}

      {/* ── Single-stock Leveraged ETFs ── */}
      {!isLoading && ssLeveragedTrades.length > 0 && (
        <div>
          <SectionHeader label="SINGLE-STOCK LEVERAGED" sublabel="2x ETF on individual stocks · day/short-swing · tight stop" color="#60a5fa" count={ssLeveragedTrades.length} />
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {ssLeveragedTrades.map(s => <TradeCard key={s.symbol+'-ssl'} s={s} onClick={() => navigate(`/chart?symbol=${s.symbol}`)} />)}
          </div>
        </div>
      )}

      {/* ── Full signal table ── */}
      <div id="signals-table" className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">All Signals</h3>
          <div className="flex gap-2 items-center flex-wrap">
            <div className="flex gap-1">
              {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as const).map(t => (
                <button key={t} onClick={() => setTierFilter(t)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    tierFilter === t ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-500 hover:bg-gray-700'
                  }`}>
                  {t === 'ALL' ? 'All' : t === 'HIGH' ? `HIGH(${high})` : t === 'MEDIUM' ? `MED(${medium})` : `LOW(${low})`}
                </button>
              ))}
            </div>
            <div className="flex gap-1 border-l border-gray-700 pl-2">
              {(['probability', 'symbol'] as SortKey[]).map(s => (
                <button key={s} onClick={() => setSort(s)}
                  className={`px-2 py-0.5 rounded text-xs transition-colors ${
                    sort === s ? 'bg-emerald-700 text-white' : 'bg-gray-800 text-gray-500 hover:bg-gray-700'
                  }`}>
                  {s === 'probability' ? '% Sort' : 'A-Z'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-16 text-gray-500">Loading signals...</div>
        ) : sorted.length === 0 ? (
          <div className="text-center py-16 text-gray-500">No signals. Run pipeline first.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500 text-[11px] uppercase tracking-wider">
                  <th className="text-left py-2.5 px-3">Action</th>
                  <th className="text-left py-2.5 px-3">Symbol</th>
                  <th className="text-left py-2.5 px-3">Conviction</th>
                  <th className="text-left py-2.5 px-3">Probability</th>
                  <th className="text-left py-2.5 px-3">Phase</th>
                  <th className="text-right py-2.5 px-3">ENTRY</th>
                  <th className="text-right py-2.5 px-3">ADD MORE</th>
                  <th className="text-right py-2.5 px-3 text-emerald-600">T1</th>
                  <th className="text-right py-2.5 px-3 text-yellow-600">TARGET</th>
                  <th className="text-right py-2.5 px-3">+%</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((s: Signal) => {
                  const action    = getAction(s)
                  const tradeType = getTradeType(s)
                  const upside    = s.sell_price > s.buy_price
                    ? (((s.sell_price - s.buy_price) / s.buy_price) * 100).toFixed(1) : null
                  return (
                    <tr key={s.symbol + s.date}
                      onClick={() => navigate(`/chart?symbol=${s.symbol}`)}
                      className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors">
                      <td className="py-2 px-3"><ActionBadge action={action} /></td>
                      <td className="py-2 px-3 font-black text-white">{s.symbol}</td>
                      <td className="py-2 px-3">
                        {(() => { const c = convictionLabel(convictionScore(s)); return (
                          <span className="text-[10px] font-black px-2 py-0.5 rounded border"
                            style={{ color: c.color, borderColor: c.color + '50', backgroundColor: c.color + '12' }}>
                            {c.label}
                          </span>
                        )})()}
                      </td>
                      <td className="py-2 px-3"><ProbBar value={s.probability} /></td>
                      <td className="py-2 px-3 text-xs text-gray-400">{s.market_phase}</td>
                      <td className="py-2 px-3 text-right font-mono text-sm font-bold text-white">${s.buy_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right font-mono text-sm text-blue-400">${s.accumulate_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right font-mono text-sm text-emerald-400">${s.scale_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right font-mono text-sm text-yellow-400">${s.sell_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right font-mono text-xs">
                        {upside ? <span className="text-emerald-500">+{upside}%</span> : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Execution guide ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        {[
          { label: 'SECTOR SWING', color: '#a78bfa', desc: 'Sector ETF rotation. Enter on red days, 10% at a time. Target: days–weeks.' },
          { label: 'SWING', color: '#34d399', desc: 'High conviction setup. Enter at ENTRY, add at ADD level. Trail stop after T1.' },
          { label: 'DAY TRADE', color: '#60a5fa', desc: 'Use 15m/1h chart for entry. Tight stop below ADD level. Exit same day.' },
          { label: '3-TIER EXIT', color: '#fbbf24', desc: 'Sell ⅓ at T1 (guarantee profit), ⅓ at TARGET, keep ⅓ as house money.' },
        ].map(({ label, color, desc }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="font-bold mb-1" style={{ color }}>{label}</div>
            <div className="text-gray-500 leading-relaxed">{desc}</div>
          </div>
        ))}
      </div>

    </div>
  )
}
