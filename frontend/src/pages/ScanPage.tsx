import { useState } from 'react'
import WatchlistPage   from './WatchlistPage'
import PatternScanPage from './PatternScanPage'

type Tab = 'patterns' | 'watchlist'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'patterns',  label: 'Pattern Scan', icon: '◈' },
  { id: 'watchlist', label: 'Watchlist',    icon: '★' },
]

export default function ScanPage() {
  const [tab, setTab] = useState<Tab>('patterns')

  return (
    <div className="space-y-4">
      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.id
                ? 'bg-emerald-700 text-white'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {tab === 'patterns'  && <PatternScanPage />}
      {tab === 'watchlist' && <WatchlistPage />}
    </div>
  )
}
