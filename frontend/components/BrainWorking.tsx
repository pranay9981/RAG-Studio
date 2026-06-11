'use client'
import { CheckCircle2, Cpu } from 'lucide-react'

interface Props {
  steps: string[]
  tokens: string
  isStreaming: boolean
  archIcon: string
  archLabel: string
}

export default function BrainWorking({ steps, tokens, isStreaming, archIcon, archLabel }: Props) {
  return (
    <div className="rounded-2xl border border-violet-500/20 bg-gradient-to-br from-violet-500/5 to-transparent p-4 space-y-3 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-2.5">
        <div className="relative flex-shrink-0">
          <div className={`w-6 h-6 rounded-lg flex items-center justify-center ${isStreaming ? 'bg-violet-500/20' : 'bg-emerald-500/20'}`}>
            <Cpu size={13} className={isStreaming ? 'text-violet-400' : 'text-emerald-400'} />
          </div>
          {isStreaming && (
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
          )}
        </div>
        <span className="text-xs font-semibold text-slate-300">{archLabel}</span>
        {isStreaming && (
          <div className="ml-auto flex items-center gap-1">
            {[0,1,2].map(i => (
              <span key={i} className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-dot-bounce"
                style={{ animationDelay: `${i * 0.16}s` }} />
            ))}
          </div>
        )}
        {!isStreaming && steps.length > 0 && (
          <CheckCircle2 size={13} className="ml-auto text-emerald-400 flex-shrink-0" />
        )}
      </div>

      {/* Steps */}
      {steps.length > 0 && (
        <div className="space-y-1.5 pl-1">
          {steps.map((step, i) => (
            <div key={i} className="flex items-start gap-2 text-xs">
              <span className="text-emerald-400 mt-0.5 flex-shrink-0 text-[10px]">✓</span>
              <span className="text-slate-400 leading-relaxed">{step}</span>
            </div>
          ))}
        </div>
      )}

      {/* Streaming tokens */}
      {tokens && (
        <div className="mt-1 pt-3 border-t border-white/[0.06] text-sm text-slate-200 leading-[1.8] prose-dark">
          {tokens}
          {isStreaming && <span className="cursor-blink" />}
        </div>
      )}
    </div>
  )
}
