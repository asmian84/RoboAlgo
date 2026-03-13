/**
 * AlertManager — persistent alert system for RoboAlgo.
 *
 * Alert types:
 *   quality_threshold  — signal quality score crosses a threshold
 *   breakout           — breakout gate passes for a symbol
 *   sweep              — liquidity sweep detected
 *   regime_change      — market state changes for a symbol
 *
 * Alerts are evaluated against CommandCenter data on every refresh.
 * Toast notifications appear in the bottom-right corner.
 * Browser notifications are requested on first alert fire.
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { useCommandCenter } from '../../api/hooks'

// ── Types ─────────────────────────────────────────────────────────────────────

export type AlertType = 'quality_threshold' | 'breakout' | 'sweep' | 'regime_change'

export interface AlertRule {
  id:         string
  type:       AlertType
  symbol:     string           // specific symbol, or '*' for all
  threshold?: number           // used by quality_threshold
  state?:     string           // used by regime_change (target state)
  enabled:    boolean
  createdAt:  string
}

export interface AlertFired {
  id:       string
  ruleId:   string
  type:     AlertType
  symbol:   string
  message:  string
  firedAt:  string
  dismissed: boolean
}

// ── Context ───────────────────────────────────────────────────────────────────

interface AlertCtx {
  rules:    AlertRule[]
  fired:    AlertFired[]
  addRule:  (rule: Omit<AlertRule, 'id' | 'createdAt'>) => void
  removeRule: (id: string) => void
  toggleRule: (id: string) => void
  dismiss:  (id: string) => void
  clearAll: () => void
}

const AlertContext = createContext<AlertCtx | null>(null)

export function useAlerts() {
  const ctx = useContext(AlertContext)
  if (!ctx) throw new Error('useAlerts must be used inside AlertProvider')
  return ctx
}

// ── Color map ─────────────────────────────────────────────────────────────────

const TYPE_COLOR: Record<AlertType, string> = {
  quality_threshold: '#eab308',
  breakout:          '#22c55e',
  sweep:             '#f97316',
  regime_change:     '#60a5fa',
}

const TYPE_ICON: Record<AlertType, string> = {
  quality_threshold: '★',
  breakout:          '↗',
  sweep:             '⚡',
  regime_change:     '⊙',
}

const TYPE_LABEL: Record<AlertType, string> = {
  quality_threshold: 'Quality Alert',
  breakout:          'Breakout Alert',
  sweep:             'Sweep Alert',
  regime_change:     'Regime Change',
}

// ── Toast component ───────────────────────────────────────────────────────────

function AlertToast({ alert, onDismiss }: { alert: AlertFired; onDismiss: () => void }) {
  const color = TYPE_COLOR[alert.type]

  useEffect(() => {
    const t = setTimeout(onDismiss, 8000)
    return () => clearTimeout(t)
  }, [onDismiss])

  return (
    <div
      className="flex items-start gap-3 px-4 py-3 rounded-xl border shadow-2xl bg-gray-900 max-w-xs w-full cursor-pointer"
      style={{ borderColor: color + '60', boxShadow: `0 0 20px ${color}20` }}
      onClick={onDismiss}
    >
      <span className="text-lg flex-shrink-0 mt-0.5" style={{ color }}>{TYPE_ICON[alert.type]}</span>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-bold" style={{ color }}>{TYPE_LABEL[alert.type]}</p>
        <p className="text-xs text-gray-300 mt-0.5 leading-snug">{alert.message}</p>
        <p className="text-[10px] text-gray-600 mt-1">
          {new Date(alert.firedAt).toLocaleTimeString()} · click to dismiss
        </p>
      </div>
    </div>
  )
}

// ── Toast container (bottom-right) ───────────────────────────────────────────

function ToastContainer({ alerts, onDismiss }: { alerts: AlertFired[]; onDismiss: (id: string) => void }) {
  const visible = alerts.filter(a => !a.dismissed).slice(-4)
  if (!visible.length) return null
  return (
    <div className="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 items-end pointer-events-none">
      {visible.map(a => (
        <div key={a.id} className="pointer-events-auto animate-fadeIn">
          <AlertToast alert={a} onDismiss={() => onDismiss(a.id)} />
        </div>
      ))}
    </div>
  )
}

// ── Provider ─────────────────────────────────────────────────────────────────

export function AlertProvider({ children }: { children: React.ReactNode }) {
  const [rules, setRules] = useState<AlertRule[]>([])
  const [fired, setFired] = useState<AlertFired[]>([])
  const prevStatesRef = useRef<Map<string, string>>(new Map())

  const { data: cc } = useCommandCenter()

  // Request browser notifications
  const requestNotifications = useCallback(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  const fire = useCallback((ruleId: string, type: AlertType, symbol: string, message: string) => {
    const alert: AlertFired = {
      id:       crypto.randomUUID(),
      ruleId,
      type,
      symbol,
      message,
      firedAt:  new Date().toISOString(),
      dismissed: false,
    }
    setFired(prev => [...prev.slice(-99), alert])

    // Browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(`RoboAlgo · ${TYPE_LABEL[type]}`, {
          body: message,
          icon: '/favicon.ico',
        })
      } catch { /* silently ignore in environments that block this */ }
    }
  }, [])

  // Evaluate rules against fresh CC data
  useEffect(() => {
    if (!cc || !rules.length) return

    const signals  = cc.opportunity_map?.signals ?? []
    const instruments = cc.market_state_summary?.instruments ?? []

    rules.filter(r => r.enabled).forEach(rule => {
      const { id, type, symbol, threshold, state } = rule

      if (type === 'quality_threshold') {
        const sigs = symbol === '*' ? signals : signals.filter(s => s.symbol === symbol)
        sigs.forEach(sig => {
          const q = sig.setup_quality_score ?? 0
          if (threshold != null && q >= threshold) {
            fire(id, type, sig.symbol, `${sig.symbol} quality score ${q.toFixed(0)} ≥ threshold ${threshold}`)
          }
        })
      }

      else if (type === 'breakout') {
        const sigs = symbol === '*' ? signals : signals.filter(s => s.symbol === symbol)
        sigs.forEach(sig => {
          if (sig.breakout_gate_passed) {
            fire(id, type, sig.symbol, `${sig.symbol} breakout gate passed · score ${(sig.breakout_quality_score ?? 0).toFixed(0)}`)
          }
        })
      }

      else if (type === 'sweep') {
        const sigs = symbol === '*' ? signals : signals.filter(s => s.symbol === symbol)
        sigs.forEach(sig => {
          if (sig.sweep_gate_passed && sig.sweep_type && sig.sweep_type !== 'none') {
            fire(id, type, sig.symbol, `${sig.symbol} liquidity ${sig.sweep_type.replace('_', ' ')} detected`)
          }
        })
      }

      else if (type === 'regime_change') {
        const insts = symbol === '*' ? instruments : instruments.filter(i => i.symbol === symbol)
        insts.forEach(inst => {
          const prev = prevStatesRef.current.get(inst.symbol)
          if (prev && prev !== inst.state && (!state || inst.state === state)) {
            fire(id, type, inst.symbol, `${inst.symbol} regime changed: ${prev} → ${inst.state}`)
          }
          prevStatesRef.current.set(inst.symbol, inst.state)
        })
      }
    })
  }, [cc, rules, fire])

  const addRule = useCallback((rule: Omit<AlertRule, 'id' | 'createdAt'>) => {
    requestNotifications()
    setRules(prev => [...prev, { ...rule, id: crypto.randomUUID(), createdAt: new Date().toISOString() }])
  }, [requestNotifications])

  const removeRule  = (id: string)  => setRules(prev => prev.filter(r => r.id !== id))
  const toggleRule  = (id: string)  => setRules(prev => prev.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r))
  const dismiss     = (id: string)  => setFired(prev => prev.map(a => a.id === id ? { ...a, dismissed: true } : a))
  const clearAll    = ()            => setFired([])

  return (
    <AlertContext.Provider value={{ rules, fired, addRule, removeRule, toggleRule, dismiss, clearAll }}>
      {children}
      <ToastContainer alerts={fired} onDismiss={dismiss} />
    </AlertContext.Provider>
  )
}

// ── AlertPanel — display + manage rules ──────────────────────────────────────

function RuleRow({ rule, onToggle, onRemove }: { rule: AlertRule; onToggle: () => void; onRemove: () => void }) {
  const color = TYPE_COLOR[rule.type]
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-opacity ${rule.enabled ? 'opacity-100' : 'opacity-40'}`}
      style={{ borderColor: color + '30', backgroundColor: color + '08' }}>
      <span style={{ color }} className="text-sm flex-shrink-0">{TYPE_ICON[rule.type]}</span>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-gray-200">
          {rule.symbol === '*' ? 'All symbols' : rule.symbol} · {TYPE_LABEL[rule.type]}
        </p>
        {rule.threshold != null && (
          <p className="text-[10px] text-gray-500">Quality ≥ {rule.threshold}</p>
        )}
        {rule.state && (
          <p className="text-[10px] text-gray-500">When state = {rule.state}</p>
        )}
      </div>
      <button onClick={onToggle} className="text-[10px] text-gray-500 hover:text-gray-200 px-1.5 py-0.5 rounded bg-gray-800 transition-colors">
        {rule.enabled ? 'ON' : 'OFF'}
      </button>
      <button onClick={onRemove} className="text-[10px] text-red-600 hover:text-red-400 transition-colors">✕</button>
    </div>
  )
}

// ── AlertModal ────────────────────────────────────────────────────────────────

export function AlertModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { rules, fired, addRule, removeRule, toggleRule, dismiss, clearAll } = useAlerts()

  const [tab,       setTab]       = useState<'rules' | 'history'>('rules')
  const [newType,   setNewType]   = useState<AlertType>('quality_threshold')
  const [newSymbol, setNewSymbol] = useState('*')
  const [newThresh, setNewThresh] = useState(70)
  const [newState,  setNewState]  = useState('EXPANSION')

  if (!isOpen) return null

  const history = fired.slice().reverse()

  const handleAdd = () => {
    addRule({
      type:      newType,
      symbol:    newSymbol.trim().toUpperCase() || '*',
      threshold: newType === 'quality_threshold' ? newThresh : undefined,
      state:     newType === 'regime_change'     ? newState  : undefined,
      enabled:   true,
    })
  }

  return (
    <div className="fixed inset-0 z-[9990] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 bg-gray-950 border border-gray-800 rounded-2xl shadow-2xl w-[520px] max-h-[80vh] flex flex-col">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div>
            <h2 className="text-sm font-bold text-gray-100">🔔 Alert Manager</h2>
            <p className="text-xs text-gray-500 mt-0.5">{rules.filter(r => r.enabled).length} active · {fired.filter(f => !f.dismissed).length} unread</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-200 text-lg leading-none">✕</button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-800">
          {(['rules', 'history'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`flex-1 py-2 text-xs font-medium transition-colors capitalize ${tab === t ? 'text-white border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-300'}`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">

          {tab === 'rules' && (
            <>
              {/* Add new rule */}
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-3 space-y-2">
                <p className="text-xs font-semibold text-gray-300">Add Alert Rule</p>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-gray-500 block mb-1">Type</label>
                    <select
                      value={newType}
                      onChange={e => setNewType(e.target.value as AlertType)}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200"
                    >
                      <option value="quality_threshold">Quality Threshold</option>
                      <option value="breakout">Breakout Gate</option>
                      <option value="sweep">Liquidity Sweep</option>
                      <option value="regime_change">Regime Change</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-500 block mb-1">Symbol (* = all)</label>
                    <input
                      value={newSymbol}
                      onChange={e => setNewSymbol(e.target.value)}
                      placeholder="TQQQ or *"
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 font-mono uppercase"
                    />
                  </div>
                </div>
                {newType === 'quality_threshold' && (
                  <div className="flex items-center gap-3">
                    <label className="text-[10px] text-gray-500">Min quality:</label>
                    <input
                      type="range" min={30} max={95} value={newThresh}
                      onChange={e => setNewThresh(parseInt(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-xs font-mono text-amber-400 w-8 text-right">{newThresh}</span>
                  </div>
                )}
                {newType === 'regime_change' && (
                  <div>
                    <label className="text-[10px] text-gray-500 block mb-1">Target state (blank = any)</label>
                    <select
                      value={newState}
                      onChange={e => setNewState(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200"
                    >
                      <option value="">Any state</option>
                      <option value="EXPANSION">EXPANSION</option>
                      <option value="TREND">TREND</option>
                      <option value="COMPRESSION">COMPRESSION</option>
                      <option value="CHAOS">CHAOS</option>
                    </select>
                  </div>
                )}
                <button
                  onClick={handleAdd}
                  className="w-full py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors"
                >
                  + Add Rule
                </button>
              </div>

              {/* Rule list */}
              {rules.length === 0 ? (
                <p className="text-xs text-gray-600 text-center py-4">No rules yet — add one above</p>
              ) : (
                <div className="space-y-2">
                  {rules.map(r => (
                    <RuleRow
                      key={r.id}
                      rule={r}
                      onToggle={() => toggleRule(r.id)}
                      onRemove={() => removeRule(r.id)}
                    />
                  ))}
                </div>
              )}
            </>
          )}

          {tab === 'history' && (
            <>
              {history.length > 0 && (
                <div className="flex justify-end">
                  <button onClick={clearAll} className="text-[10px] text-gray-500 hover:text-red-400 transition-colors">
                    Clear all
                  </button>
                </div>
              )}
              {history.length === 0 ? (
                <p className="text-xs text-gray-600 text-center py-4">No alerts fired yet</p>
              ) : (
                <div className="space-y-2">
                  {history.map(a => {
                    const color = TYPE_COLOR[a.type]
                    return (
                      <div
                        key={a.id}
                        className={`flex items-start gap-2 px-3 py-2 rounded-lg border transition-opacity cursor-pointer ${a.dismissed ? 'opacity-40' : 'opacity-100'}`}
                        style={{ borderColor: color + '30', backgroundColor: color + '08' }}
                        onClick={() => dismiss(a.id)}
                      >
                        <span style={{ color }} className="text-sm flex-shrink-0 mt-0.5">{TYPE_ICON[a.type]}</span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-gray-200">{a.message}</p>
                          <p className="text-[10px] text-gray-600 mt-0.5">
                            {new Date(a.firedAt).toLocaleString()}
                            {a.dismissed && ' · dismissed'}
                          </p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
