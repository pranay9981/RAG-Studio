'use client'
import { useState } from 'react'
import { ChevronDown, Sparkles } from 'lucide-react'
import type { ArchInfo } from '@/lib/types'

interface Props { arch: ArchInfo }

export default function ArchCard({ arch }: Props) {
  const [open, setOpen] = useState(false)
  const num = arch.key.split(' ')[0]
  const name = arch.key.split(' ').slice(1).join(' ')

  return (
    <div className={`flex-1 min-w-0 transition-colors ${open ? 'bg-violet-500/[0.04]' : ''}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3.5 px-5 py-3.5 text-left group"
      >
        {/* Arch number */}
        <span className="text-[11px] font-mono font-bold text-violet-500/40 w-5 flex-shrink-0 tabular-nums select-none">
          {num}
        </span>

        {/* Icon */}
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-violet-800/10 border border-violet-500/20 flex items-center justify-center text-lg flex-shrink-0 shadow-[0_0_14px_rgba(124,58,237,0.1)] group-hover:shadow-[0_0_18px_rgba(124,58,237,0.18)] transition-shadow">
          {arch.icon}
        </div>

        {/* Name + tagline */}
        <div className="flex-1 min-w-0">
          <p className="text-[15px] font-bold text-slate-100 leading-tight">{name}</p>
          <p className="text-xs text-slate-500 mt-0.5 leading-snug">{arch.tagline}</p>
        </div>

        {/* Status chip + chevron */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          <span className="flex items-center gap-1.5 text-[10px] font-semibold text-emerald-400/80 bg-emerald-400/8 border border-emerald-400/15 px-2.5 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_5px_rgba(52,211,153,0.6)]" />
            Active
          </span>
          <ChevronDown
            size={14}
            className={`text-slate-600 group-hover:text-slate-400 transition-all duration-200 ${open ? 'rotate-180' : ''}`}
          />
        </div>
      </button>

      {open && (
        <div className="px-5 pb-4 pt-3 border-t border-white/[0.05] grid grid-cols-2 gap-6 bg-violet-500/[0.02]">
          <div className="space-y-1.5">
            <p className="text-[9px] font-bold text-slate-600 uppercase tracking-widest">How it works</p>
            <p className="text-xs text-slate-400 leading-relaxed">{arch.how}</p>
          </div>
          <div className="space-y-1.5">
            <p className="text-[9px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-1">
              <Sparkles size={8} className="text-violet-400" /> Best for
            </p>
            <p className="text-xs text-violet-300 leading-relaxed">{arch.best_for}</p>
          </div>
        </div>
      )}
    </div>
  )
}
