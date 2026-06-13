'use client'
import {
  Trash2, RotateCcw, Download, History, ChevronDown, ChevronUp,
  BarChart2, Key, Database, Zap,
} from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import type { ArchInfo, DocItem, HistoryItem } from '@/lib/types'

function archMainName(key: string): string {
  const full = key.split(' ').slice(1).join(' ')
  const i = full.indexOf('(')
  return i === -1 ? full : full.slice(0, i).trim()
}

interface ExportOption { label: string; fn: () => void }

interface Props {
  architectures:  ArchInfo[]
  selectedArch:   string
  compareMode:    boolean
  enableEval:     boolean
  ingestedArchs:  Set<string>
  messageCounts:  Record<string, number>
  docLibrary:     DocItem[]
  history:        HistoryItem[]
  bgeM3Loaded?:   boolean
  onSelectArch:   (k: string) => void
  onCompareToggle:() => void
  onEvalToggle:   () => void
  onClearChat:    () => void
  onReset:        () => void
  exportOptions:  ExportOption[]
  onClearCache:   () => void
  onAnalytics:    () => void
  onSettings:     () => void
  children:       React.ReactNode
}

export default function Sidebar({
  architectures, selectedArch, compareMode, enableEval, ingestedArchs,
  messageCounts, docLibrary, history, bgeM3Loaded,
  onSelectArch, onCompareToggle, onEvalToggle, onClearChat, onReset,
  exportOptions, onClearCache, onAnalytics, onSettings, children,
}: Props) {
  const [histOpen, setHistOpen]       = useState(false)
  const [exportOpen, setExportOpen]   = useState(false)
  const [cacheFeedback, setCacheFeedback] = useState(false)
  const exportRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!exportOpen) return
    const h = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node))
        setExportOpen(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [exportOpen])

  const handleClearCache = () => {
    onClearCache()
    setCacheFeedback(true)
    setTimeout(() => setCacheFeedback(false), 1500)
  }

  const actions = [
    { icon: <Trash2 size={11} />,    label: 'Clear',   fn: onClearChat,                  state: '' },
    { icon: <RotateCcw size={11} />, label: 'Reset',   fn: onReset,                      state: '' },
    { icon: <Download size={11} />,  label: 'Export',  fn: () => setExportOpen(o => !o), state: exportOpen ? 'active' : '' },
    { icon: <Database size={11} />,  label: cacheFeedback ? 'Cleared!' : 'Cache',
                                     fn: handleClearCache,                                state: cacheFeedback ? 'success' : '' },
    { icon: <BarChart2 size={11} />, label: 'Stats',   fn: onAnalytics,                  state: '' },
    { icon: <Key size={11} />,       label: 'API Key', fn: onSettings,                   state: '' },
  ]

  return (
    <aside className="w-[260px] flex flex-col overflow-hidden flex-shrink-0"
      style={{ background: '#090912', borderRight: '1px solid rgba(255,255,255,0.05)' }}>

      {/* ── Logo ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{
            background: 'linear-gradient(135deg, #7c3aed, #5b21b6)',
            boxShadow: '0 0 14px rgba(124,58,237,0.45)',
          }}>
          <Zap size={13} className="text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-white tracking-tight leading-none">RAG Studio</p>
          <p className="text-[10px] leading-none mt-0.5" style={{ color: '#475569' }}>10 Architectures</p>
        </div>
        <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded flex-shrink-0"
          style={{ color: '#a78bfa', background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.2)' }}>
          v4
        </span>
      </div>

      {/* ── Scrollable body ───────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto min-h-0 scrollbar-thin flex flex-col">

        {/* Architecture list — no inner scroll, all 10 fit at 30px each */}
        <div className="px-3 pt-3 pb-2 flex-shrink-0">
          <p className="text-[9px] font-bold uppercase tracking-widest mb-2 px-1"
            style={{ color: '#334155', letterSpacing: '0.12em' }}>
            Architecture
          </p>
          <div className="space-y-px">
            {architectures.map(a => {
              const active   = selectedArch === a.key && !compareMode
              const count    = messageCounts[a.key] ?? 0
              const ingested = ingestedArchs.has(a.key)
              const isBGE    = a.key.includes('Multilingual')
              const num      = a.key.split(' ')[0]
              const name     = archMainName(a.key)

              return (
                <button
                  key={a.key}
                  onClick={() => onSelectArch(a.key)}
                  className="w-full flex items-center gap-2 px-2.5 rounded-lg text-left transition-all relative group"
                  style={{
                    paddingTop: 7, paddingBottom: 7,
                    background: active ? 'rgba(124,58,237,0.12)' : 'transparent',
                    border: active
                      ? '1px solid rgba(124,58,237,0.28)'
                      : '1px solid transparent',
                  }}
                  onMouseEnter={e => {
                    if (!active) {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.035)'
                      ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.05)'
                    }
                  }}
                  onMouseLeave={e => {
                    if (!active) {
                      (e.currentTarget as HTMLElement).style.background = 'transparent'
                      ;(e.currentTarget as HTMLElement).style.borderColor = 'transparent'
                    }
                  }}
                >
                  {/* Active left bar */}
                  {active && (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 rounded-r-full flex-shrink-0"
                      style={{ width: 3, height: 20, background: '#a78bfa' }} />
                  )}

                  {/* Number */}
                  <span className="text-[9px] font-bold font-mono w-4 text-right leading-none flex-shrink-0 tabular-nums"
                    style={{ color: active ? '#7c3aed' : '#1e293b' }}>
                    {num}
                  </span>

                  {/* Icon */}
                  <span className="text-sm leading-none flex-shrink-0 transition-opacity"
                    style={{ opacity: active ? 1 : 0.45 }}>
                    {a.icon}
                  </span>

                  {/* Name */}
                  <span className="flex-1 min-w-0 text-[11.5px] font-medium truncate leading-none"
                    style={{ color: active ? '#ede9fe' : '#64748b' }}>
                    {name}
                  </span>

                  {/* Badges */}
                  <div className="flex items-center gap-1 flex-shrink-0">
                    {count > 0 && (
                      <span className="text-[9px] font-bold px-1 py-px rounded-full leading-none"
                        style={{
                          background: active ? 'rgba(124,58,237,0.3)' : 'rgba(255,255,255,0.06)',
                          color: active ? '#c4b5fd' : '#475569',
                        }}>
                        {count}
                      </span>
                    )}
                    {isBGE && !bgeM3Loaded && (
                      <span title="BGE-M3 loading…"
                        className="w-1.5 h-1.5 rounded-full animate-pulse flex-shrink-0"
                        style={{ background: '#f59e0b' }} />
                    )}
                    {ingested && (
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                        style={{ background: '#34d399' }} />
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Divider */}
        <div className="mx-3 flex-shrink-0" style={{ height: 1, background: 'rgba(255,255,255,0.04)' }} />

        {/* Toggles */}
        <div className="px-4 py-3 space-y-2.5 flex-shrink-0">
          {[
            { label: `Compare All (${architectures.length})`, val: compareMode, fn: onCompareToggle },
            { label: 'RAG Evaluation',                        val: enableEval,  fn: onEvalToggle   },
          ].map(({ label, val, fn }) => (
            <label key={label} className="flex items-center justify-between cursor-pointer">
              <span className="text-[11.5px] select-none" style={{ color: '#64748b' }}>{label}</span>
              <button
                onClick={fn}
                className="relative rounded-full flex-shrink-0 transition-all duration-200"
                style={{
                  width: 32, height: 18,
                  background: val ? '#7c3aed' : 'rgba(255,255,255,0.07)',
                  boxShadow: val ? '0 0 8px rgba(124,58,237,0.4)' : 'none',
                }}
              >
                <span
                  className="absolute rounded-full bg-white shadow transition-all duration-200"
                  style={{
                    top: 1, width: 16, height: 16,
                    left: val ? 15 : 1,
                  }}
                />
              </button>
            </label>
          ))}
        </div>

        {/* Divider */}
        <div className="mx-3 flex-shrink-0" style={{ height: 1, background: 'rgba(255,255,255,0.04)' }} />

        {/* Document Manager (injected via children) */}
        <div className="flex-shrink-0">
          {children}
        </div>

        {/* History */}
        {history.length > 0 && (
          <div className="flex-shrink-0">
            <div className="mx-3" style={{ height: 1, background: 'rgba(255,255,255,0.04)' }} />
            <button
              onClick={() => setHistOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-2.5 transition-colors"
              style={{ color: '#475569' }}
              onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = '#94a3b8')}
              onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = '#475569')}
            >
              <span className="flex items-center gap-1.5 text-[11px] font-medium">
                <History size={10} /> Recent ({history.length})
              </span>
              {histOpen ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>
            {histOpen && (
              <div className="px-4 pb-3 space-y-2">
                {[...history].reverse().slice(0, 8).map((h, i) => (
                  <div key={i}>
                    <p className="text-[11px] truncate" style={{ color: '#94a3b8' }}>{h.query}</p>
                    <p className="text-[10px]" style={{ color: '#334155' }}>
                      {h.arch.split(' ')[0]} · {h.elapsed.toFixed(1)}s
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1" />
      </div>

      {/* ── Action buttons — always pinned at bottom ──────────── */}
      <div className="px-3 py-3 flex-shrink-0"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <div ref={exportRef} className="relative">

          {/* Export popover */}
          {exportOpen && (
            <div className="absolute bottom-full mb-2 left-0 right-0 rounded-xl overflow-hidden z-20"
              style={{
                background: '#0d0d1a',
                border: '1px solid rgba(255,255,255,0.1)',
                boxShadow: '0 -8px 32px rgba(0,0,0,0.5)',
              }}>
              <p className="px-3 py-2 text-[9px] font-bold uppercase tracking-widest"
                style={{ color: '#334155', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                Export as .md
              </p>
              {exportOptions.map(opt => (
                <button
                  key={opt.label}
                  onClick={() => { opt.fn(); setExportOpen(false) }}
                  className="w-full text-left px-3 py-2.5 text-xs flex items-center gap-2 transition-colors"
                  style={{ color: '#94a3b8' }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = 'rgba(124,58,237,0.1)'
                    ;(e.currentTarget as HTMLElement).style.color = '#c4b5fd'
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = 'transparent'
                    ;(e.currentTarget as HTMLElement).style.color = '#94a3b8'
                  }}
                >
                  <Download size={10} style={{ color: '#475569' }} />
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-3 gap-1">
            {actions.map(({ icon, label, fn, state }) => (
              <button
                key={label}
                onClick={fn}
                className="flex flex-col items-center gap-1 py-2 rounded-lg text-[10px] font-medium transition-all"
                style={{
                  color: state === 'active' ? '#c4b5fd'
                       : state === 'success' ? '#34d399'
                       : '#475569',
                  background: state === 'active' ? 'rgba(124,58,237,0.1)'
                            : state === 'success' ? 'rgba(52,211,153,0.08)'
                            : 'transparent',
                  border: state === 'active' ? '1px solid rgba(124,58,237,0.25)'
                        : state === 'success' ? '1px solid rgba(52,211,153,0.2)'
                        : '1px solid transparent',
                }}
                onMouseEnter={e => {
                  if (!state) {
                    (e.currentTarget as HTMLElement).style.color = '#cbd5e1'
                    ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)'
                    ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.07)'
                  }
                }}
                onMouseLeave={e => {
                  if (!state) {
                    (e.currentTarget as HTMLElement).style.color = '#475569'
                    ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                    ;(e.currentTarget as HTMLElement).style.borderColor = 'transparent'
                  }
                }}
              >
                {icon}
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  )
}
