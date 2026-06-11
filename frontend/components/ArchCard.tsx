'use client'
import { useState } from 'react'
import { ChevronDown, ChevronUp, Sparkles } from 'lucide-react'
import type { ArchInfo } from '@/lib/types'

interface Props { arch: ArchInfo }

export default function ArchCard({ arch }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`border-b border-white/[0.05] transition-colors ${open ? 'bg-violet-500/[0.03]' : 'hover:bg-white/[0.015]'}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-5 py-3.5 text-left"
      >
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500/20 to-violet-900/20 border border-violet-500/20 flex items-center justify-center text-base flex-shrink-0">
          {arch.icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-100 truncate">{arch.key.split(' ').slice(1).join(' ')}</p>
          <p className="text-xs text-slate-500 truncate mt-0.5">{arch.tagline}</p>
        </div>
        <span className={`transition-transform duration-200 text-slate-600 flex-shrink-0 ${open ? 'rotate-180' : ''}`}>
          <ChevronDown size={14} />
        </span>
      </button>

      {open && (
        <div className="px-5 pb-4 pt-1 animate-fade-in grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">How it works</p>
            <p className="text-xs text-slate-400 leading-relaxed">{arch.how}</p>
          </div>
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1">
              <Sparkles size={9} className="text-violet-400" /> Best for
            </p>
            <p className="text-xs text-violet-300 leading-relaxed">{arch.best_for}</p>
          </div>
        </div>
      )}
    </div>
  )
}
