import type { CompareResult, ArchInfo } from '@/lib/types'
import { Loader2 } from 'lucide-react'

interface Props {
  results: CompareResult[]
  architectures: ArchInfo[]
  loading: boolean
}

export default function CompareGrid({ results, architectures, loading }: Props) {
  const archMap = Object.fromEntries(architectures.map(a => [a.key, a]))

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 gap-3 text-slate-400">
        <Loader2 size={18} className="animate-spin" />
        <span className="text-sm">Running all 8 architectures…</span>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 p-1">
      {results.map(r => {
        const info = archMap[r.arch_key]
        return (
          <div key={r.arch_key} className="bg-surface border border-white/[0.06] rounded-xl p-4 flex flex-col gap-3 animate-fade-in">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-base">{info?.icon || '🤖'}</span>
                <span className="text-xs font-medium text-slate-300">{r.arch_key.split(' ').slice(1).join(' ')}</span>
              </div>
              <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${r.error ? 'text-red-400 bg-red-400/10' : 'text-slate-500 bg-white/[0.04]'}`}>
                {r.error ? 'error' : `${r.elapsed.toFixed(2)}s`}
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed flex-1 line-clamp-6">
              {r.error || r.answer || 'No answer generated.'}
            </p>
          </div>
        )
      })}
    </div>
  )
}
