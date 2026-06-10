'use client'
import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { ArchInfo } from '@/lib/types'

interface Props { arch: ArchInfo }

export default function ArchCard({ arch }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-white/[0.06]">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-6 py-3 hover:bg-white/[0.02] transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span className="text-lg">{arch.icon}</span>
          <div>
            <p className="text-sm font-medium text-slate-100">{arch.key}</p>
            <p className="text-xs text-slate-500">{arch.tagline}</p>
          </div>
        </div>
        {open ? <ChevronUp size={14} className="text-slate-500 flex-shrink-0" /> : <ChevronDown size={14} className="text-slate-500 flex-shrink-0" />}
      </button>
      {open && (
        <div className="px-6 pb-4 grid grid-cols-[1fr_auto] gap-6">
          <div>
            <p className="text-xs font-medium text-slate-400 mb-1">How it works</p>
            <p className="text-xs text-slate-500 leading-relaxed">{arch.how}</p>
          </div>
          <div className="min-w-[160px]">
            <p className="text-xs font-medium text-slate-400 mb-1">Best for</p>
            <p className="text-xs text-indigo-300 leading-relaxed">{arch.best_for}</p>
          </div>
        </div>
      )}
    </div>
  )
}
