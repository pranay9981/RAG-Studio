'use client'
import { Trash2, RotateCcw, Download, History, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import type { ArchInfo, DocItem, HistoryItem } from '@/lib/types'

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
  onExport: () => void
  children: React.ReactNode // DocumentManager slot
}

export default function Sidebar({ architectures, selectedArch, compareMode, enableEval, ingestedArchs, messageCounts, docLibrary, history, onSelectArch, onCompareToggle, onEvalToggle, onClearChat, onReset, onExport, children }: Props) {
  const [histOpen, setHistOpen] = useState(false)

  return (
    <aside className="w-72 bg-[#0d0d18] border-r border-white/[0.06] flex flex-col overflow-hidden flex-shrink-0">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-white/[0.06] flex items-center gap-2.5">
        <div className="w-6 h-6 rounded bg-indigo-500/20 flex items-center justify-center text-xs">⚡</div>
        <span className="text-sm font-semibold text-white/90 tracking-tight">RAG Studio</span>
        <span className="ml-auto text-[10px] font-mono text-slate-600 bg-white/[0.04] px-1.5 py-0.5 rounded">v2</span>
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
                className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all text-xs ${selectedArch === a.key ? 'bg-indigo-500/15 border border-indigo-500/25 text-indigo-200' : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 border border-transparent'}`}
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
            { label: 'Compare all 8', val: compareMode, fn: onCompareToggle },
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

        {/* Doc library */}
        {docLibrary.length > 0 && (
          <div className="px-4 py-3 border-b border-white/[0.06]">
            <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2">Ingested Docs</p>
            <div className="space-y-1.5">
              {[...new Map(docLibrary.map(d => [d.name, d])).values()].map((d, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-slate-400">
                  <span className="text-slate-600">📄</span>
                  <span className="truncate flex-1">{d.name.split('/').pop()}</span>
                  <span className="text-slate-600 flex-shrink-0">{d.chunks}c</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="border-b border-white/[0.06]">
            <button onClick={() => setHistOpen(o => !o)} className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200">
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
      <div className="px-3 py-3 border-t border-white/[0.06] flex gap-1.5">
        {[
          { icon: <Trash2 size={13} />, label: 'Clear', fn: onClearChat },
          { icon: <RotateCcw size={13} />, label: 'Reset', fn: onReset },
          { icon: <Download size={13} />, label: 'Export', fn: onExport },
        ].map(({ icon, label, fn }) => (
          <button key={label} onClick={fn} className="flex-1 flex flex-col items-center gap-1 py-2 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-white/[0.05] transition-colors text-[10px]">
            {icon}{label}
          </button>
        ))}
      </div>
    </aside>
  )
}
