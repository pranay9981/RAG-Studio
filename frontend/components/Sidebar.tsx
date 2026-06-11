'use client'
import { Trash2, RotateCcw, Download, History, ChevronDown, ChevronUp, BarChart2, Key, Database } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import type { ArchInfo, DocItem, HistoryItem } from '@/lib/types'

interface ExportOption {
  label: string
  fn: () => void
}

interface Props {
  architectures: ArchInfo[]
  selectedArch: string
  compareMode: boolean
  enableEval: boolean
  ingestedArchs: Set<string>
  messageCounts: Record<string, number>
  docLibrary: DocItem[]
  history: HistoryItem[]
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
  messageCounts, docLibrary, history,
  onSelectArch, onCompareToggle, onEvalToggle, onClearChat, onReset,
  exportOptions, onClearCache, onAnalytics, onSettings, children,
}: Props) {
  const [histOpen, setHistOpen] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [cacheFeedback, setCacheFeedback] = useState(false)
  const exportRef = useRef<HTMLDivElement>(null)

  // Close export dropdown when clicking outside
  useEffect(() => {
    if (!exportOpen) return
    const handler = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [exportOpen])

  const handleClearCacheClick = () => {
    onClearCache()
    setCacheFeedback(true)
    setTimeout(() => setCacheFeedback(false), 1500)
  }

  return (
    <aside className="w-72 bg-[#0d0d18] border-r border-white/[0.06] flex flex-col overflow-hidden flex-shrink-0">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-white/[0.06] flex items-center gap-2.5">
        <div className="w-6 h-6 rounded bg-indigo-500/20 flex items-center justify-center text-xs">⚡</div>
        <span className="text-sm font-semibold text-white/90 tracking-tight">RAG Studio</span>
        <span className="ml-auto text-[10px] font-mono text-slate-600 bg-white/[0.04] px-1.5 py-0.5 rounded">v4</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Architecture selector */}
        <div className="px-3 py-3 border-b border-white/[0.06]">
          <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2 px-1">Architecture</p>
          <div className="space-y-0.5">
            {architectures.map(a => (
              <button
                key={a.key}
                onClick={() => onSelectArch(a.key)}
                className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all text-xs ${
                  selectedArch === a.key
                    ? 'bg-indigo-500/15 border border-indigo-500/25 text-indigo-200'
                    : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 border border-transparent'
                }`}
              >
                <span className="text-sm flex-shrink-0">{a.icon}</span>
                <span className="truncate flex-1">{a.key}</span>
                <div className="flex items-center gap-1.5 flex-shrink-0 ml-auto">
                  {(messageCounts[a.key] ?? 0) > 0 && (
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/20">
                      {messageCounts[a.key]}
                    </span>
                  )}
                  {ingestedArchs.has(a.key) && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Toggles */}
        <div className="px-4 py-3 border-b border-white/[0.06] space-y-2.5">
          {[
            { label: `Compare (${architectures.length})`, val: compareMode, fn: onCompareToggle },
            { label: 'RAG Evaluation', val: enableEval, fn: onEvalToggle },
          ].map(({ label, val, fn }) => (
            <label key={label} className="flex items-center justify-between cursor-pointer">
              <span className="text-xs text-slate-400">{label}</span>
              <button onClick={fn} className={`w-9 h-5 rounded-full transition-colors ${val ? 'bg-indigo-500' : 'bg-white/10'} relative flex-shrink-0`}>
                <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all ${val ? 'left-4' : 'left-0.5'}`} />
              </button>
            </label>
          ))}
        </div>

        {/* Document manager slot */}
        <div className="border-b border-white/[0.06]">{children}</div>

        {/* History */}
        {history.length > 0 && (
          <div className="border-b border-white/[0.06]">
            <button
              onClick={() => setHistOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200"
            >
              <span className="flex items-center gap-1.5"><History size={11} /> History ({history.length})</span>
              {histOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
            {histOpen && (
              <div className="px-4 pb-3 space-y-2">
                {[...history].reverse().slice(0, 10).map((h, i) => (
                  <div key={i} className="text-xs">
                    <p className="text-slate-300 truncate">{h.query}</p>
                    <p className="text-slate-600">{h.arch.split(' ')[0]} · {h.elapsed.toFixed(1)}s</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-3 py-3 border-t border-white/[0.06]">
        {/* Export dropdown */}
        <div ref={exportRef} className="relative">
          {exportOpen && (
            <div className="absolute bottom-full mb-1.5 left-0 right-0 bg-[#13131f] border border-white/[0.1] rounded-lg overflow-hidden shadow-xl z-20">
              <p className="px-3 py-1.5 text-[10px] font-medium text-slate-500 uppercase tracking-wider border-b border-white/[0.06]">Export as Markdown</p>
              {exportOptions.map(opt => (
                <button
                  key={opt.label}
                  onClick={() => { opt.fn(); setExportOpen(false) }}
                  className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-indigo-500/10 hover:text-indigo-200 transition-colors flex items-center gap-2"
                >
                  <Download size={10} className="text-slate-500" />
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-3 gap-1">
            {[
              { icon: <Trash2 size={12} />, label: 'Clear', fn: onClearChat, active: false },
              { icon: <RotateCcw size={12} />, label: 'Reset', fn: onReset, active: false },
              {
                icon: <Download size={12} />,
                label: 'Export',
                fn: () => setExportOpen(o => !o),
                active: exportOpen,
              },
              {
                icon: <Database size={12} />,
                label: cacheFeedback ? 'Cleared!' : 'Cache',
                fn: handleClearCacheClick,
                active: cacheFeedback,
              },
              { icon: <BarChart2 size={12} />, label: 'Stats', fn: onAnalytics, active: false },
              { icon: <Key size={12} />, label: 'API Key', fn: onSettings, active: false },
            ].map(({ icon, label, fn, active }) => (
              <button
                key={label}
                onClick={fn}
                className={`flex flex-col items-center gap-1 py-2 rounded-lg transition-colors text-[10px] ${
                  active
                    ? 'text-indigo-300 bg-indigo-500/10 border border-indigo-500/20'
                    : 'text-slate-500 hover:text-slate-200 hover:bg-white/[0.05] border border-transparent'
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
