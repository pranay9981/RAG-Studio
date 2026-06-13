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
    <div className={`flex-1 min-w-0 transition-all ${open ? 'bg-violet-500/[0.03]' : ''}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-4 px-5 py-4 text-left group"
      >
        {/* Arch number */}
        <span className="text-[11px] font-mono font-bold text-violet-500/50 w-5 flex-shrink-0 tabular-nums select-none leading-none">
          {num}
        </span>

        {/* Icon with glow ring */}
        <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-violet-500/25 to-violet-900/20 border border-violet-500/25 flex items-center justify-center text-xl flex-shrink-0 shadow-[0_0_16px_rgba(124,58,237,0.15)] group-hover:shadow-[0_0_22px_rgba(124,58,237,0.25)] transition-shadow">
          {arch.icon}
        </div>

        {/* Name + tagline */}
        <div className="flex-1 min-w-0">
          <p className="text-[15px] font-bold text-white leading-tight tracking-tight">{name}</p>
          <p className="text-xs text-slate-500 mt-0.5 leading-snug">{arch.tagline}</p>
        </div>

        {/* Chevron only — Active chip lives in page.tsx */}
        <ChevronDown
          size={15}
          className={`text-slate-600 group-hover:text-slate-400 transition-all duration-200 flex-shrink-0 ${open ? 'rotate-180 text-violet-400' : ''}`}
        />
      </button>

      {open && (
        <div className="px-5 pb-5 pt-3.5 border-t border-white/[0.05] grid grid-cols-2 gap-6 bg-gradient-to-b from-violet-500/[0.03] to-transparent">
          <div className="space-y-2">
            <p className="text-[9px] font-bold text-slate-600 uppercase tracking-widest">How it works</p>
            <p className="text-xs text-slate-400 leading-relaxed">{arch.how}</p>
          </div>
          <div className="space-y-2">
            <p className="text-[9px] font-bold text-violet-500/70 uppercase tracking-widest flex items-center gap-1.5">
              <Sparkles size={8} /> Best for
            </p>
            <p className="text-xs text-violet-300/90 leading-relaxed">{arch.best_for}</p>
          </div>
        </div>
      )}
    </div>
  )
}
