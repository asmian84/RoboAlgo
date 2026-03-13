/**
 * Vertical toolbar for manual chart drawing tools.
 * Positioned absolutely at the left edge of the chart container.
 * When a drawing is selected, shows a properties panel to its right.
 */
import { useState } from 'react'
import type { DrawingData, DrawingToolType, LineStyle } from '../../types/drawings'
import { DRAWING_COLORS } from '../../types/drawings'

interface Props {
  activeTool: DrawingToolType | null
  activeColor: string
  selectedDrawing: DrawingData | null
  drawingCount: number
  onSelectTool: (tool: DrawingToolType | null) => void
  onSetColor: (color: string) => void
  onDeleteSelected: () => void
  onClearAll: () => void
  onUpdateDrawing: (id: string, updates: Partial<DrawingData>) => void
}

const TOOLS: { type: DrawingToolType; icon: string; label: string }[] = [
  { type: 'trendline',  icon: '╲', label: 'Trendline' },
  { type: 'horizontal', icon: '─', label: 'Horizontal Line' },
  { type: 'fibonacci',  icon: 'Fib', label: 'Fibonacci Retracement' },
  { type: 'rectangle',  icon: '□', label: 'Rectangle' },
]

const LINE_STYLES: { value: LineStyle; icon: string; label: string }[] = [
  { value: 'solid',  icon: '─',  label: 'Solid' },
  { value: 'dashed', icon: '╌',  label: 'Dashed' },
  { value: 'dotted', icon: '···', label: 'Dotted' },
]

// ── Reusable active/inactive button ──────────────────────────────────────────

function ActiveBtn({
  active, onClick, title, children,
}: { active: boolean; onClick: () => void; title?: string; children: React.ReactNode }) {
  return (
    <button
      title={title}
      onClick={onClick}
      className="h-6 px-2 flex items-center justify-center rounded text-[10px] font-bold transition-all"
      style={{
        backgroundColor: active ? 'rgba(124,58,237,0.35)' : 'rgba(255,255,255,0.03)',
        color:           active ? '#c4b5fd' : '#6b7280',
        border:          active ? '1px solid rgba(124,58,237,0.6)' : '1px solid rgba(75,85,99,0.4)',
        minWidth: 22,
      }}
    >
      {children}
    </button>
  )
}

// ── Properties panel (shown when a drawing is selected) ──────────────────────

function PropertiesPanel({
  drawing,
  onUpdate,
}: { drawing: DrawingData; onUpdate: (updates: Partial<DrawingData>) => void }) {
  const lw       = drawing.lineWidth  ?? 2
  const ls       = drawing.lineStyle  ?? 'solid'
  const fo       = drawing.fillOpacity ?? 0.1
  const ext      = drawing.extended   ?? false

  const showStyle = true                                                 // all types support style
  const showFill  = drawing.type === 'rectangle' || drawing.type === 'fibonacci'
  const showExt   = drawing.type === 'trendline'
  const typeLabel = TOOLS.find(t => t.type === drawing.type)?.label ?? drawing.type

  return (
    <div
      className="absolute left-10 top-0 z-30 flex flex-col gap-2 p-2.5 bg-gray-900/95 border border-gray-700 rounded-lg shadow-2xl backdrop-blur-sm"
      style={{ minWidth: 168 }}
      onMouseDown={e => e.stopPropagation()}
    >
      {/* Type label */}
      <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-800 pb-1.5">
        {typeLabel}
      </div>

      {/* Line width */}
      <div className="flex flex-col gap-1">
        <span className="text-[9px] text-gray-600 uppercase tracking-wider">Width</span>
        <div className="flex gap-1">
          {([1, 2, 3] as const).map(w => (
            <ActiveBtn key={w} active={lw === w} onClick={() => onUpdate({ lineWidth: w })} title={`${w}px`}>
              <div style={{ width: 16, height: w + 1, borderRadius: 1, backgroundColor: 'currentColor' }} />
            </ActiveBtn>
          ))}
        </div>
      </div>

      {/* Line style */}
      {showStyle && (
        <div className="flex flex-col gap-1">
          <span className="text-[9px] text-gray-600 uppercase tracking-wider">Style</span>
          <div className="flex gap-1">
            {LINE_STYLES.map(({ value, icon, label }) => (
              <ActiveBtn key={value} active={ls === value} onClick={() => onUpdate({ lineStyle: value })} title={label}>
                {icon}
              </ActiveBtn>
            ))}
          </div>
        </div>
      )}

      {/* Fill opacity (rectangle / fib) */}
      {showFill && (
        <div className="flex flex-col gap-1">
          <span className="text-[9px] text-gray-600 uppercase tracking-wider">
            Fill {Math.round(fo * 100)}%
          </span>
          <input
            type="range"
            min={5} max={40} step={5}
            value={Math.round(fo * 100)}
            onChange={e => onUpdate({ fillOpacity: parseInt(e.target.value) / 100 })}
            className="w-full h-1 cursor-pointer accent-violet-500"
          />
        </div>
      )}

      {/* Extend as ray (trendline only) */}
      {showExt && (
        <div className="flex flex-col gap-1">
          <span className="text-[9px] text-gray-600 uppercase tracking-wider">Mode</span>
          <ActiveBtn active={ext} onClick={() => onUpdate({ extended: !ext })} title="Extend as infinite ray">
            ⇢ Extend
          </ActiveBtn>
        </div>
      )}

      {/* Divider */}
      <div className="h-px bg-gray-800" />

      {/* Color swatches — applies directly to the selected drawing */}
      <div className="flex flex-col gap-1">
        <span className="text-[9px] text-gray-600 uppercase tracking-wider">Color</span>
        <div className="grid grid-cols-4 gap-1">
          {DRAWING_COLORS.map(c => (
            <button
              key={c}
              title={c}
              onClick={() => onUpdate({ color: c })}
              className="w-5 h-5 rounded-full border-2 transition-transform hover:scale-110"
              style={{
                backgroundColor: c,
                borderColor: c === drawing.color ? '#fff' : 'transparent',
              }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main toolbar ──────────────────────────────────────────────────────────────

export default function DrawingToolbar({
  activeTool, activeColor, selectedDrawing, drawingCount,
  onSelectTool, onSetColor, onDeleteSelected, onClearAll, onUpdateDrawing,
}: Props) {
  const [showColors, setShowColors] = useState(false)

  return (
    <div
      className="absolute left-1 top-1 z-20 flex flex-col gap-0.5 p-1 bg-gray-900/90 border border-gray-800 rounded-lg backdrop-blur-sm"
      onMouseDown={e => e.stopPropagation()}
    >
      {/* Properties panel floats to the right when a drawing is selected */}
      {selectedDrawing && (
        <PropertiesPanel
          drawing={selectedDrawing}
          onUpdate={updates => onUpdateDrawing(selectedDrawing.id, updates)}
        />
      )}

      {/* Drawing tool buttons */}
      {TOOLS.map(({ type, icon, label }) => {
        const isActive = activeTool === type
        return (
          <button
            key={type}
            title={label}
            onClick={() => onSelectTool(isActive ? null : type)}
            className="w-7 h-7 flex items-center justify-center rounded text-[11px] font-bold transition-all"
            style={{
              backgroundColor: isActive ? 'rgba(124,58,237,0.35)' : 'rgba(255,255,255,0.03)',
              color:           isActive ? '#c4b5fd' : '#6b7280',
              border:          isActive ? '1px solid rgba(124,58,237,0.6)' : '1px solid transparent',
            }}
          >
            {icon}
          </button>
        )
      })}

      {/* Divider */}
      <div className="h-px bg-gray-800 mx-0.5 my-0.5" />

      {/* Global color picker — only visible when no drawing is selected */}
      {!selectedDrawing && (
        <div className="relative">
          <button
            title="Drawing color"
            onClick={() => setShowColors(!showColors)}
            className="w-7 h-7 flex items-center justify-center rounded transition-all"
            style={{ border: '1px solid transparent' }}
          >
            <div
              className="w-4 h-4 rounded-full border border-gray-600"
              style={{ backgroundColor: activeColor }}
            />
          </button>
          {showColors && (
            <div
              className="absolute left-8 top-0 z-30 p-1.5 bg-gray-900 border border-gray-700 rounded-lg shadow-xl grid grid-cols-4 gap-1"
              onMouseLeave={() => setShowColors(false)}
            >
              {DRAWING_COLORS.map(c => (
                <button
                  key={c}
                  onClick={() => { onSetColor(c); setShowColors(false) }}
                  className="w-5 h-5 rounded-full border-2 transition-transform hover:scale-110"
                  style={{
                    backgroundColor: c,
                    borderColor: c === activeColor ? '#fff' : 'transparent',
                  }}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Divider */}
      <div className="h-px bg-gray-800 mx-0.5 my-0.5" />

      {/* Delete selected drawing */}
      <button
        title="Delete selected drawing (Del)"
        onClick={onDeleteSelected}
        disabled={!selectedDrawing}
        className="w-7 h-7 flex items-center justify-center rounded text-[11px] transition-all disabled:opacity-20"
        style={{
          color: selectedDrawing ? '#f87171' : '#6b7280',
          border: '1px solid transparent',
        }}
      >
        🗑
      </button>

      {/* Clear all drawings */}
      {drawingCount > 0 && (
        <button
          title="Clear all drawings"
          onClick={onClearAll}
          className="w-7 h-7 flex items-center justify-center rounded text-[9px] text-gray-500 hover:text-red-400 transition-all"
          style={{ border: '1px solid transparent' }}
        >
          CLR
        </button>
      )}
    </div>
  )
}
