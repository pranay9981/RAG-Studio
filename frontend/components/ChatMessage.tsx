import type { ChatMessage as Msg } from '@/lib/types'
import SourcePanel from './SourcePanel'
import EvalScorecard from './EvalScorecard'

interface Props { message: Msg; archIcon?: string }

export default function ChatMessage({ message, archIcon }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[75%] px-4 py-3 rounded-2xl rounded-tr-sm bg-indigo-500/15 border border-indigo-500/25 text-sm text-slate-100 leading-relaxed">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 animate-fade-in">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-slate-500">{archIcon || '🤖'} {message.arch}</span>
        {message.elapsed != null && (
          <span className="text-[10px] font-mono text-slate-600">{message.elapsed.toFixed(2)}s</span>
        )}
      </div>
      <div className="max-w-[88%] px-4 py-3 rounded-2xl rounded-tl-sm bg-surface border border-white/[0.06] text-sm text-slate-200 leading-relaxed prose-dark whitespace-pre-wrap">
        {message.content}
      </div>
      {message.sources && message.sources.length > 0 && (
        <div className="max-w-[88%]">
          <SourcePanel sources={message.sources} />
        </div>
      )}
      {message.eval && (
        <div className="max-w-[88%]">
          <EvalScorecard scores={message.eval} />
        </div>
      )}
    </div>
  )
}
