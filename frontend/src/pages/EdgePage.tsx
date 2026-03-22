import { useState } from 'react'
import DashboardPage from './DashboardPage'
import BacktestPage  from './BacktestPage'
import EvolutionPage from './EvolutionPage'
import ResearchPage  from './ResearchPage'

type Tab = 'live' | 'backtest' | 'evolution' | 'research'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'live',      label: 'Live',      icon: '◉' },
  { id: 'backtest',  label: 'Backtest',  icon: '◑' },
  { id: 'evolution', label: 'Evolution', icon: '⟳' },
  { id: 'research',  label: 'Research',  icon: '◎' },
]

export default function EdgePage() {
  const [tab, setTab] = useState<Tab>('live')

  return (
    <div className="space-y-4">

      {/* ── Header + tabs ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight">EDGE</h2>
          <p className="text-xs text-gray-500 mt-0.5">Live Setups · Backtest · Evolution · Research</p>
        </div>

        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                tab === t.id
                  ? 'bg-emerald-700/30 text-emerald-300 border border-emerald-700/40'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <span className="text-xs">{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Tab content ── */}
      {tab === 'live'      && <DashboardPage />}
      {tab === 'backtest'  && <BacktestPage />}
      {tab === 'evolution' && <EvolutionPage />}
      {tab === 'research'  && <ResearchPage />}

    </div>
  )
}
