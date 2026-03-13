import { useState } from 'react'
import BullBearPage from './BullBearPage'
import CyclePage from './CyclePage'
import FeaturePage from './FeaturePage'
import ComparisonPage from './ComparisonPage'

type Tab = 'bull-bear' | 'cycles' | 'features' | 'compare'

const TABS: { id: Tab; label: string; icon: string; desc: string }[] = [
  { id: 'bull-bear', label: 'Bull / Bear',  icon: '⇅', desc: 'Leveraged pair analysis' },
  { id: 'cycles',    label: 'Cycles',        icon: '◎', desc: 'Market cycle heatmap' },
  { id: 'features',  label: 'Features',      icon: '⊡', desc: 'ML feature matrix' },
  { id: 'compare',   label: 'Compare',       icon: '⊞', desc: 'Multi-symbol charts' },
]

export default function ResearchPage() {
  const [tab, setTab] = useState<Tab>('bull-bear')

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-black text-white tracking-tight">RESEARCH</h2>
        <p className="text-xs text-gray-500 mt-0.5">Market analysis tools — cycles, features, and pair comparisons</p>
      </div>

      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.id
                ? 'bg-gray-700 text-white'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      <div>
        {tab === 'bull-bear' && <BullBearPage />}
        {tab === 'cycles'    && <CyclePage />}
        {tab === 'features'  && <FeaturePage />}
        {tab === 'compare'   && <ComparisonPage />}
      </div>
    </div>
  )
}
