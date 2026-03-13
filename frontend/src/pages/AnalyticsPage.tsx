import { useState } from 'react'
import BacktestPage  from './BacktestPage'
import EvolutionPage from './EvolutionPage'
import ResearchPage  from './ResearchPage'

type Tab = 'backtest' | 'evolution' | 'research'

const TABS: { id: Tab; label: string; icon: string; desc: string }[] = [
  { id: 'backtest',  label: 'Backtest',  icon: '◑', desc: 'Historical signal win-rates, expectancy and phase analysis' },
  { id: 'evolution', label: 'Evolution', icon: '⟳', desc: 'Strategy fitness scores, suggestions and trade replay' },
  { id: 'research',  label: 'Research',  icon: '◎', desc: 'Market research tools and data exploration' },
]

export default function AnalyticsPage() {
  const [tab, setTab] = useState<Tab>('backtest')

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-black text-white tracking-tight">ANALYTICS</h2>
        <p className="text-xs text-gray-500 mt-0.5">Backtest · Evolution · Research</p>
      </div>

      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.id
                ? 'bg-violet-700 text-white'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {tab === 'backtest'  && <BacktestPage />}
      {tab === 'evolution' && <EvolutionPage />}
      {tab === 'research'  && <ResearchPage />}
    </div>
  )
}
