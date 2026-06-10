'use client'
import { CheckCircle, Loader2 } from 'lucide-react'

interface Props {
  steps: string[]
  tokens: string
  isStreaming: boolean
  archIcon: string
  archLabel: string
}

export default function BrainWorking({ steps, tokens, isStreaming, archIcon, archLabel }: Props) {
  return (
    <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-4 space-y-2 animate-fade-in">
      <div className="flex items-center gap-2 mb-1">
        {isStreaming
          ? <Loader2 size={13} className="text-indigo-400 animate-spin flex-shrink-0" />
          : <CheckCircle size={13} className="text-green-400 flex-shrink-0" />}
        <span className="text-xs font-medium text-indigo-300">
          {archIcon} {archLabel}
        </span>
      </div>

      {steps.map((step, i) => (
        <div key={i} className="flex items-start gap-2 text-xs text-slate-400">
          <span className="text-green-400 mt-0.5 flex-shrink-0">✓</span>
          <span>{step}</span>
        </div>
      ))}

      {tokens && (
        <div className="mt-3 pt-3 border-t border-white/[0.06] text-sm text-slate-200 leading-relaxed prose-dark whitespace-pre-wrap">
          {tokens}
          {isStreaming && <span className="cursor-blink" />}
        </div>
      )}
    </div>
  )
}
