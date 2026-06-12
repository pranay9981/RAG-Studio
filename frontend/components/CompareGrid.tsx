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
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fade-in" onClick={onClose}>
      <div
        className="bg-surface border border-white/[0.1] rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col shadow-glow overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-5 py-4 border-b border-white/[0.07] flex-shrink-0">
          <div className="w-8 h-8 rounded-xl bg-violet-500/15 border border-violet-500/20 flex items-center justify-center text-base flex-shrink-0">
            {info?.icon || '🤖'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-slate-200 truncate">
              {result.arch_key.split(' ').slice(1).join(' ')}
            </p>
            <p className="text-[11px] text-slate-500">{result.error ? 'Error occurred' : `${result.elapsed.toFixed(2)}s`}</p>
          </div>
          <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-300 hover:bg-white/[0.06] transition-all">
            <X size={14} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1 px-5 py-4 text-sm leading-relaxed prose-dark">
          {result.error
            ? <p className="text-red-400">{result.error}</p>
            : <MarkdownContent content={result.answer || 'No answer generated.'} />
          }
        </div>
        {result.eval && (
          <div className="px-5 pb-4 border-t border-white/[0.07] flex-shrink-0">
            <EvalScorecard scores={result.eval} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function CompareGrid({ results, architectures, loading }: Props) {
  const archMap = Object.fromEntries(architectures.map(a => [a.key, a]))
  const [expanded, setExpanded] = useState<string | null>(null)

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 rounded-full border-2 border-violet-500/30 animate-ping" />
          <div className="absolute inset-2 rounded-full bg-violet-500/20 flex items-center justify-center">
            <Loader2 size={16} className="text-violet-400 animate-spin" />
          </div>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-slate-300">Running all 10 architectures</p>
          <p className="text-xs text-slate-600 mt-1">Results will appear as they complete…</p>
        </div>
      </div>
    )
  }

  const expandedResult = results.find(r => r.arch_key === expanded)

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3 p-1">
        {results.map(r => {
          const info = archMap[r.arch_key]
          const archLabel = r.arch_key.split(' ').slice(1).join(' ')
          return (
            <div key={r.arch_key}
              className="bg-surface border border-white/[0.07] rounded-2xl flex flex-col animate-slide-up overflow-hidden hover:border-white/[0.12] transition-all group shadow-card"
            >
              {/* Header */}
              <div className="flex items-center gap-2.5 px-4 pt-3.5 pb-3 border-b border-white/[0.05]">
                <div className="w-7 h-7 rounded-lg bg-violet-500/15 border border-violet-500/20 flex items-center justify-center text-sm flex-shrink-0">
                  {info?.icon || '🤖'}
                </div>
                <span className="text-xs font-semibold text-slate-200 truncate flex-1">{archLabel}</span>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded-md ${r.error ? 'text-red-400 bg-red-400/10' : 'text-slate-500 bg-white/[0.04]'}`}>
                    {r.error ? 'ERR' : `${r.elapsed.toFixed(2)}s`}
                  </span>
                  <button
                    onClick={() => setExpanded(r.arch_key)}
                    className="w-5 h-5 rounded-md flex items-center justify-center text-slate-600 hover:text-violet-400 hover:bg-violet-500/10 transition-all"
                  >
                    <Maximize2 size={10} />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="px-4 py-3 overflow-y-auto flex-1 text-xs leading-relaxed scrollbar-thin min-h-[120px] max-h-[200px]">
                {r.error
                  ? <p className="text-red-400 text-xs">{r.error}</p>
                  : <MarkdownContent content={r.answer || 'No answer generated.'} />
                }
              </div>

              {/* Eval scores */}
              {r.eval && (
                <div className="px-4 pb-3 border-t border-white/[0.05] pt-2.5">
                  <EvalScorecard scores={r.eval} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {expanded && expandedResult && (
        <ExpandModal result={expandedResult} info={archMap[expanded]} onClose={() => setExpanded(null)} />
      )}
    </>
  )
}
