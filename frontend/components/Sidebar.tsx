'use client'
import { Trash2, RotateCcw, Download, History, ChevronDown, ChevronUp, BarChart2, Key, Database, Zap } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import type { ArchInfo, DocItem, HistoryItem } from '@/lib/types'

function parseArchName(key: string): { main: string; sub: string } {
  const full = key.split(' ').slice(1).join(' ')
  const i = full.indexOf('(')
  if (i === -1) return { main: full, sub: '' }
  return { main: full.slice(0, i).trim(), sub: full.slice(i) }
}

interface ExportOption { label: string; fn: () => void }

interface Props {
  architectures: ArchInfo[]
  selectedArch: string
  compareMode: boolean
  enableEval: boolean
  ingestedArchs: Set<string>
  messageCounts: Record<string, number>
  docLibrary: DocItem[]
  history: HistoryItem[]
  bgeM3Loaded?: boolean
  onSelectArch: (k: string) => void
  onCompareToggle: () => void
  onEvalToggle: () => void
  onClearChat: () => void
  onReset: () => void
  exportOptions: ExportOption[]
  onClearCache: () => void
  onAnalytics: () => void
  onSettings: () => void
  children: React.ReactNode
}

export default function Sidebar({
  architectures, selectedArch, compareMode, enableEval, ingestedArchs,
  messageCounts, docLibrary, history, bgeM3Loaded,
  onSelectArch, onCompareToggle, onEvalToggle, onClearChat, onReset,
  exportOptions, onClearCache, onAnalytics, onSettings, children,
}: Props) {
  const [histOpen, setHistOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [cacheFeedback, setCacheFeedback] = useState(false)
  const exportRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!exportOpen) return
    const h = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) setExportOpen(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [exportOpen])

  const handleClearCacheClick = () => {
    onClearCache()
    setCacheFeedback(true)
    setTimeout(() => setCacheFeedback(false), 1500)
  }

  return (
    <aside className="w-[288px] bg-surface border-r border-white/[0.06] flex flex-col overflow-hidden flex-shrink-0">

      {/* ── Logo ─────────────────────────────────────────── */}
      <div className="px-4 py-3.5 border-b border-white/[0.06] flex items-center gap-3 flex-shrink-0">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-violet-700 flex items-center justify-center shadow-[0_0_14px_rgba(124,58,237,0.35)] flex-shrink-0">
          <Zap size={14} className="text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-white tracking-tight leading-none">RAG Studio</p>
          <p className="text-[10px] text-slate-600 mt-0.5">10 Architectures</p>
        </div>
        <span className="text-[9px] font-mono font-bold text-violet-400 bg-violet-500/10 border border-violet-500/20 px-1.5 py-0.5 rounded-md flex-shrink-0">v4</span>
      </div>

      {/* ── Scrollable middle (arch + toggles + docs + history) ── */}
      <div className="flex-1 overflow-y-auto min-h-0 scrollbar-thin">

      {/* Architecture list (own inner scroll) */}
      <div className="px-3 pt-3 pb-2.5 border-b border-white/[0.05]">
        <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mb-2 px-1">Architecture</p>
        <div className="space-y-0.5 overflow-y-auto max-h-[268px] scrollbar-thin pr-0.5">
          {architectures.map(a => {
            const active = selectedArch === a.key
            const count = messageCounts[a.key] ?? 0
            const ingested = ingestedArchs.has(a.key)
            const isBGE = a.key.includes('BGE')
            const { main, sub } = parseArchName(a.key)
            const num = a.key.split(' ')[0]
            return (
              <button
                key={a.key}
                onClick={() => onSelectArch(a.key)}
                className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-xl text-left transition-all relative group ${
                  active
                    ? 'bg-violet-500/14 border border-violet-500/30 shadow-[0_0_0_1px_rgba(124,58,237,0.08)]'
                    : 'border border-transparent hover:bg-white/[0.04] hover:border-white/[0.06]'
                }`}
              >
                {active && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 bg-violet-400 rounded-r-full" />
                )}
                {/* Arch number */}
                <span className={`text-[9px] font-bold font-mono w-4 flex-shrink-0 text-right leading-none ${active ? 'text-violet-500' : 'text-slate-700'}`}>
                  {num}
                </span>
                {/* Icon */}
                <span className={`text-[13px] flex-shrink-0 transition-opacity ${active ? 'opacity-100' : 'opacity-50 group-hover:opacity-70'}`}>
                  {a.icon}
                </span>
                {/* Name — 2 lines, no truncation */}
                <div className="flex-1 min-w-0">
                  <p className={`text-[11.5px] font-semibold leading-tight ${active ? 'text-violet-100' : 'text-slate-400 group-hover:text-slate-300'}`}>
                    {main}
                  </p>
                  {sub && (
                    <p className={`text-[10px] leading-tight mt-0.5 ${active ? 'text-violet-400/70' : 'text-slate-600'}`}>
                      {sub}
                    </p>
                  )}
                </div>
                {/* Badges */}
                <div className="flex items-center gap-1 flex-shrink-0">
                  {count > 0 && (
                    <span className={`text-[9px] font-bold px-1 py-px rounded-full leading-none ${
                      active ? 'bg-violet-500/30 text-violet-300' : 'bg-white/[0.07] text-slate-600'
                    }`}>
                      {count}
                    </span>
                  )}
                  {isBGE && !bgeM3Loaded && (
                    <span
                      title="BGE-M3 not loaded yet — will be ready on first use"
                      className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse flex-shrink-0"
                    />
                  )}
                  {ingested && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Toggles ──────────────────────────────────── */}
      <div className="px-4 py-3 border-b border-white/[0.05] space-y-2.5">
        {[
          { label: `Compare All (${architectures.length})`, val: compareMode, fn: onCompareToggle },
          { label: 'RAG Evaluation', val: enableEval, fn: onEvalToggle },
        ].map(({ label, val, fn }) => (
          <label key={label} className="flex items-center justify-between cursor-pointer group">
            <span className="text-xs text-slate-500 group-hover:text-slate-400 transition-colors select-none">{label}</span>
            <button
              onClick={fn}
              className={`relative w-9 h-5 rounded-full transition-all duration-200 flex-shrink-0 ${val ? 'bg-violet-600 shadow-[0_0_8px_rgba(124,58,237,0.4)]' : 'bg-white/[0.08]'}`}
            >
              <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${val ? 'left-4' : 'left-0.5'}`} />
            </button>
          </label>
        ))}
      </div>

      {/* ── Document Manager ─────────── */}
      <div className="border-b border-white/[0.05]">
        {children}
      </div>

      {/* ── History ──────────────────── */}
      {history.length > 0 && (
        <div className="border-b border-white/[0.05]">
          <button
            onClick={() => setHistOpen(o => !o)}
            className="w-full sticky top-0 flex items-center justify-between px-4 py-2.5 text-xs text-slate-500 hover:text-slate-300 transition-colors bg-surface z-10 border-b border-white/[0.03]"
          >
            <span className="flex items-center gap-1.5 font-medium">
              <History size={11} /> Recent ({history.length})
            </span>
            {histOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
          {histOpen && (
            <div className="px-4 py-2.5 space-y-2.5">
              {[...history].reverse().slice(0, 8).map((h, i) => (
                <div key={i} className="space-y-0.5">
                  <p className="text-xs text-slate-400 truncate leading-relaxed">{h.query}</p>
                  <p className="text-[10px] text-slate-600">{h.arch.split(' ')[0]} · {h.elapsed.toFixed(1)}s</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      </div>{/* end scrollable middle */}

      {/* ── Actions (always visible at bottom) ───────── */}
      <div className="px-3 py-3 border-t border-white/[0.06] flex-shrink-0">
        <div ref={exportRef} className="relative">
          {exportOpen && (
            <div className="absolute bottom-full mb-2 left-0 right-0 bg-[#0d0d1a] border border-white/[0.1] rounded-xl overflow-hidden shadow-[0_-8px_32px_rgba(0,0,0,0.4)] z-20">
              <p className="px-3 py-2 text-[10px] font-bold text-slate-600 uppercase tracking-widest border-b border-white/[0.06]">Export as .md</p>
              {exportOptions.map(opt => (
                <button
                  key={opt.label}
                  onClick={() => { opt.fn(); setExportOpen(false) }}
                  className="w-full text-left px-3 py-2.5 text-xs text-slate-400 hover:bg-violet-500/10 hover:text-violet-300 transition-colors flex items-center gap-2"
                >
                  <Download size={10} className="text-slate-600" />
                  {opt.label}
                </button>
              ))}
            </div>
          )}
          <div className="grid grid-cols-3 gap-1">
            {[
              { icon: <Trash2 size={12} />, label: 'Clear', fn: onClearChat, extra: '' },
              { icon: <RotateCcw size={12} />, label: 'Reset', fn: onReset, extra: '' },
              { icon: <Download size={12} />, label: 'Export', fn: () => setExportOpen(o => !o), extra: exportOpen ? 'active' : '' },
              { icon: <Database size={12} />, label: cacheFeedback ? 'Cleared!' : 'Cache', fn: handleClearCacheClick, extra: cacheFeedback ? 'success' : '' },
              { icon: <BarChart2 size={12} />, label: 'Stats', fn: onAnalytics, extra: '' },
              { icon: <Key size={12} />, label: 'API Key', fn: onSettings, extra: '' },
            ].map(({ icon, label, fn, extra }) => (
              <button
                key={label}
                onClick={fn}
                className={`flex flex-col items-center gap-1 py-2.5 rounded-xl text-[10px] font-medium transition-all border ${
                  extra === 'active'
                    ? 'text-violet-300 bg-violet-500/12 border-violet-500/25'
                    : extra === 'success'
                    ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                    : 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05] border-transparent hover:border-white/[0.06]'
                }`}
              >
                {icon}{label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  )
}
