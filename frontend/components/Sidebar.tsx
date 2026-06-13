'use client'
import { Trash2, RotateCcw, Download, History, ChevronDown, ChevronUp, BarChart2, Key, Database, Zap } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import type { ArchInfo, DocItem, HistoryItem } from '@/lib/types'

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
    <aside className="w-[260px] bg-surface border-r border-white/[0.06] flex flex-col overflow-hidden flex-shrink-0">
      {/* Logo / Brand */}
      <div className="px-4 py-4 border-b border-white/[0.06] flex items-center gap-3">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-violet-700 flex items-center justify-center shadow-glow-sm flex-shrink-0">
          <Zap size={13} className="text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-white tracking-tight">RAG Studio</p>
          <p className="text-[10px] text-slate-600">10 Architectures</p>
        </div>
        <span className="text-[9px] font-mono font-bold text-violet-500 bg-violet-500/10 border border-violet-500/20 px-1.5 py-0.5 rounded-md">v4</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Architecture list */}
        <div className="px-3 py-3 border-b border-white/[0.05]">
          <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mb-2 px-1">Architecture</p>
          <div className="space-y-0.5">
            {architectures.map(a => {
              const active = selectedArch === a.key
              const count = messageCounts[a.key] ?? 0
              const ingested = ingestedArchs.has(a.key)
              return (
                <button
                  key={a.key}
                  onClick={() => onSelectArch(a.key)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-xl text-left transition-all text-xs relative ${
                    active
                      ? 'bg-violet-500/15 border border-violet-500/30 text-violet-200'
                      : 'text-slate-500 hover:bg-white/[0.04] hover:text-slate-300 border border-transparent'
                  }`}
                >
                  {active && (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-violet-400 rounded-r-full" />
                  )}
                  <span className="text-sm flex-shrink-0 opacity-80">{a.icon}</span>
                  <span className="truncate flex-1 font-medium">{a.key.split(' ').slice(1).join(' ')}</span>
                  <div className="flex items-center gap-1.5 flex-shrink-0 ml-auto">
                    {count > 0 && (
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full ${active ? 'bg-violet-500/30 text-violet-300' : 'bg-white/[0.06] text-slate-500'}`}>
                        {count}
                      </span>
                    )}
                    {a.key.includes('BGE') && !bgeM3Loaded && (
                      <span
                        title="BGE-M3 model not loaded yet — will load on first use (~30 s)"
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

        {/* Toggles */}
        <div className="px-4 py-3 border-b border-white/[0.05] space-y-3">
          {[
            { label: `Compare All (${architectures.length})`, val: compareMode, fn: onCompareToggle },
            { label: 'RAG Evaluation', val: enableEval, fn: onEvalToggle },
          ].map(({ label, val, fn }) => (
            <label key={label} className="flex items-center justify-between cursor-pointer group">
              <span className="text-xs text-slate-500 group-hover:text-slate-400 transition-colors">{label}</span>
              <button
                onClick={fn}
                className={`relative w-9 h-5 rounded-full transition-all duration-200 flex-shrink-0 ${val ? 'bg-violet-600' : 'bg-white/[0.08]'}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${val ? 'left-4' : 'left-0.5'}`} />
              </button>
            </label>
          ))}
        </div>

        {/* Document manager */}
        <div className="border-b border-white/[0.05]">{children}</div>

        {/* History */}
        {history.length > 0 && (
          <div className="border-b border-white/[0.05]">
            <button
              onClick={() => setHistOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              <span className="flex items-center gap-1.5 font-medium">
                <History size={11} /> Recent ({history.length})
              </span>
              {histOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
            {histOpen && (
              <div className="px-4 pb-3 space-y-2.5">
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
      </div>

      {/* Actions */}
      <div className="px-3 py-3 border-t border-white/[0.06]">
        <div ref={exportRef} className="relative">
          {exportOpen && (
            <div className="absolute bottom-full mb-2 left-0 right-0 bg-[#0f0f22] border border-white/[0.1] rounded-xl overflow-hidden shadow-glow z-20 animate-slide-up">
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
                    : 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05] border-transparent'
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
