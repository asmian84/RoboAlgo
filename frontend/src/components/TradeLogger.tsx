import { useState, useRef } from 'react'
import axios from 'axios'

interface TradeLoggerProps {
  symbol?: string
  entryPrice?: number
  targetPrice?: number
  stopPrice?: number
  signals?: Record<string, unknown>
  onSuccess?: (tradeId: string) => void
}

export function TradeLogger({
  symbol = '',
  entryPrice = 0,
  targetPrice = 0,
  stopPrice = 0,
  signals = {},
  onSuccess,
}: TradeLoggerProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const [formData, setFormData] = useState({
    symbol: symbol.toUpperCase(),
    timeframe: 'daily',
    entry_price: entryPrice,
    target_price: targetPrice,
    stop_price: stopPrice,
    notes: '',
  })

  const screenshotInputRef = useRef<HTMLInputElement>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      // Get screenshot if user selected one
      let screenshotB64: string | null = null
      const fileInput = screenshotInputRef.current
      if (fileInput?.files?.[0]) {
        const file = fileInput.files[0]
        const reader = new FileReader()
        screenshotB64 = await new Promise((resolve) => {
          reader.onload = () => {
            const b64 = (reader.result as string).split(',')[1]
            resolve(b64)
          }
          reader.readAsDataURL(file)
        })
      }

      // Prepare request
      const payload = {
        symbol: formData.symbol,
        timeframe: formData.timeframe,
        entry_price: formData.entry_price,
        target_price: formData.target_price,
        stop_price: formData.stop_price,
        signals: signals || {},
        screenshot_base64: screenshotB64,
        notes: formData.notes,
      }

      // Send to API
      const res = await axios.post('/api/trades/log-entry', payload)
      const { trade_id } = res.data

      // Reset form
      setFormData({
        symbol: symbol.toUpperCase(),
        timeframe: 'daily',
        entry_price: entryPrice,
        target_price: targetPrice,
        stop_price: stopPrice,
        notes: '',
      })
      if (fileInput) fileInput.value = ''
      setIsOpen(false)

      // Callback
      onSuccess?.(trade_id)

      // Show success notification
      alert(`✓ Trade logged: ${trade_id}`)
    } catch (err: unknown) {
      const msg = (err as any)?.response?.data?.detail || 'Failed to log trade'
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 z-40 px-4 py-2 rounded-lg font-bold text-white bg-emerald-600 hover:bg-emerald-500 transition-all shadow-lg flex items-center gap-2"
      >
        <span className="text-lg">📝</span>
        Log Trade
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-96 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-gray-100">Log Trade Entry</h2>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-500 hover:text-gray-300 text-xl"
              >
                ✕
              </button>
            </div>

            {error && (
              <div className="mb-4 p-2 bg-red-950/60 border border-red-800 rounded text-xs text-red-400">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              {/* Symbol */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Symbol
                </label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => setFormData({ ...formData, symbol: e.target.value.toUpperCase() })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                  required
                />
              </div>

              {/* Timeframe */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Timeframe
                </label>
                <select
                  value={formData.timeframe}
                  onChange={(e) => setFormData({ ...formData, timeframe: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                >
                  <option>1m</option>
                  <option>5m</option>
                  <option>15m</option>
                  <option>1h</option>
                  <option>4h</option>
                  <option selected>daily</option>
                </select>
              </div>

              {/* Entry Price */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Entry Price
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.entry_price || ''}
                  onChange={(e) => setFormData({ ...formData, entry_price: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                  required
                />
              </div>

              {/* Target Price */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Target Price
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.target_price || ''}
                  onChange={(e) => setFormData({ ...formData, target_price: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                  required
                />
              </div>

              {/* Stop Price */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Stop Price
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={formData.stop_price || ''}
                  onChange={(e) => setFormData({ ...formData, stop_price: parseFloat(e.target.value) || 0 })}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                  required
                />
              </div>

              {/* Screenshot */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  📸 Chart Screenshot (optional)
                </label>
                <input
                  ref={screenshotInputRef}
                  type="file"
                  accept="image/*"
                  className="w-full text-xs text-gray-400 file:px-2 file:py-1 file:rounded file:border-0 file:text-xs file:font-bold file:bg-gray-800 file:text-gray-300 hover:file:bg-gray-700"
                />
                <p className="text-[9px] text-gray-600 mt-1">
                  Use Cmd+Shift+4 (Mac) or Win+Shift+S (Windows) to capture
                </p>
              </div>

              {/* Notes */}
              <div>
                <label className="text-xs text-gray-400 uppercase font-bold block mb-1">
                  Notes (optional)
                </label>
                <textarea
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                  rows={2}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-100 focus:border-emerald-600 outline-none"
                  placeholder="e.g., reasons for this trade, confluence factors, market conditions..."
                />
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full mt-4 px-3 py-2 rounded font-bold text-sm text-white bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 transition-all"
              >
                {isLoading ? '⏳ Logging…' : '✓ Log Trade'}
              </button>
            </form>

            <p className="text-[10px] text-gray-600 mt-3">
              All trades logged to: <code className="bg-gray-800 px-1 rounded">data/trades/</code>
              {' '} · Signal metadata: {Object.keys(signals || {}).length > 0 ? '✓ attached' : '— will add from context'}
            </p>
          </div>
        </div>
      )}
    </>
  )
}
