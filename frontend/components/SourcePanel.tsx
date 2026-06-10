'use client'
import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText } from 'lucide-react'
import type { Source } from '@/lib/types'

interface Props { sources: Source[] }

export default function SourcePanel({ sources }: Props) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null

  return (
    <div className="mt-2 rounded-lg border border-white/[0.06] overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <FileText size={12} />
          {sources.length} source chunk{sources.length !== 1 ? 's' : ''} retrieved
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div className="border-t border-white/[0.06] divide-y divide-white/[0.04]">
          {sources.map((src, i) => (
            <div key={i} className="px-3 py-2.5 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-indigo-300 truncate max-w-[70%]">
                  {src.source?.split('/').pop() || src.source || 'Unknown'}
                </span>
                {src.score != null && (
                  <span className="text-[10px] text-slate-500 font-mono">score {src.score.toFixed(2)}</span>
                )}
              </div>
              <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed">
                {src.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
