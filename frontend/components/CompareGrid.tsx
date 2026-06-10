'use client'
import { useState } from 'react'
import type { CompareResult, ArchInfo } from '@/lib/types'
import { Loader2, Maximize2, X } from 'lucide-react'
import MarkdownContent from './MarkdownContent'
import EvalScorecard from './EvalScorecard'

interface Props {
  results: CompareResult[]
  architectures: ArchInfo[]
  loading: boolean
}

function ExpandModal({ result, info, onClose }: { result: CompareResult; info?: ArchInfo; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-[#131320] border border-white/[0.1] rounded-2xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col gap-4 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">{info?.icon || '🤖'}</span>
            <span className="text-sm font-semibold text-slate-200">{result.arch_key.split(' ').slice(1).join(' ')}</span>
            <span className="text-[11px] font-mono text-slate-500 bg-white/[0.04] px-1.5 py-0.5 rounded">
              {result.error ? 'error' : `${result.elapsed.toFixed(2)}s`}
            </span>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
            <X size={16} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1 text-sm leading-relaxed pr-1">
          {result.error
            ? <p className="text-red-400">{result.error}</p>
            : <MarkdownContent content={result.answer || 'No answer generated.'} />
          }
        </div>
      </div>
    </div>
  )
}

export default function CompareGrid({ results, architectures, loading }: Props) {
  const archMap = Object.fromEntries(architectures.map(a => [a.key, a]))
  const [expanded, setExpanded] = useState<string | null>(null)

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 gap-3 text-slate-400">
        <Loader2 size={18} className="animate-spin" />
        <span className="text-sm">Running all 8 architectures…</span>
      </div>
    )
  }

  const expandedResult = results.find(r => r.arch_key === expanded)

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 p-1">
        {results.map(r => {
          const info = archMap[r.arch_key]
          return (
            <div key={r.arch_key} className="bg-surface border border-white/[0.06] rounded-xl flex flex-col gap-0 animate-fade-in overflow-hidden">
              {/* Card header */}
              <div className="flex items-center justify-between px-4 pt-3 pb-2 border-b border-white/[0.04]">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-base flex-shrink-0">{info?.icon || '🤖'}</span>
                  <span className="text-xs font-medium text-slate-300 truncate">{r.arch_key.split(' ').slice(1).join(' ')}</span>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${r.error ? 'text-red-400 bg-red-400/10' : 'text-slate-500 bg-white/[0.04]'}`}>
                    {r.error ? 'error' : `${r.elapsed.toFixed(2)}s`}
                  </span>
                  <button
                    onClick={() => setExpanded(r.arch_key)}
                    className="text-slate-600 hover:text-slate-300 transition-colors"
                    title="Expand full answer"
                  >
                    <Maximize2 size={12} />
                  </button>
                </div>
              </div>

              {/* Scrollable content */}
              <div className="px-4 py-3 overflow-y-auto max-h-[220px] text-xs leading-relaxed scrollbar-thin">
                {r.error
                  ? <p className="text-red-400">{r.error}</p>
                  : <MarkdownContent content={r.answer || 'No answer generated.'} />
                }
              </div>

              {/* Eval scores */}
              {r.eval && (
                <div className="px-4 pb-3 border-t border-white/[0.04] pt-2">
                  <EvalScorecard scores={r.eval} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {expanded && expandedResult && (
        <ExpandModal
          result={expandedResult}
          info={archMap[expanded]}
          onClose={() => setExpanded(null)}
        />
      )}
    </>
  )
}
