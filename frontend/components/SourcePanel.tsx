'use client'
import { useState } from 'react'
import { ChevronDown, ChevronUp, BookOpen, Hash } from 'lucide-react'
import type { Source } from '@/lib/types'

interface Props { sources: Source[] }

export default function SourcePanel({ sources }: Props) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null

  const unique = [...new Map(sources.map(s => [s.source, s])).values()]

  return (
    <div className="mt-2 rounded-xl overflow-hidden border border-white/[0.06] bg-white/[0.015]">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs hover:bg-white/[0.03] transition-colors group"
      >
        <span className="flex items-center gap-2 text-slate-400 group-hover:text-slate-300">
          <BookOpen size={11} className="text-violet-400" />
          <span className="font-medium">{sources.length} chunk{sources.length !== 1 ? 's' : ''}</span>
          <span className="text-slate-600">from</span>
          <span className="text-violet-400">{unique.length} source{unique.length !== 1 ? 's' : ''}</span>
        </span>
        <span className="text-slate-600">
          {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        </span>
      </button>

      {open && (
        <div className="border-t border-white/[0.06] divide-y divide-white/[0.04]">
          {sources.map((src, i) => (
            <div key={i} className="px-3.5 py-3 group/src hover:bg-white/[0.02] transition-colors">
              <div className="flex items-center gap-2 mb-1.5">
                <div className="w-5 h-5 rounded-md bg-violet-500/15 border border-violet-500/25 flex items-center justify-center flex-shrink-0">
                  <Hash size={9} className="text-violet-400" />
                </div>
                <span className="text-xs font-medium text-violet-300 truncate flex-1">
                  {src.source?.split('/').pop()?.split('\\').pop() || src.source || 'Unknown'}
                </span>
                {src.score != null && (
                  <span className="text-[10px] font-mono text-slate-600 bg-white/[0.04] px-1.5 py-0.5 rounded-md flex-shrink-0">
                    {(src.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 line-clamp-3 leading-relaxed pl-7">
                {src.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
