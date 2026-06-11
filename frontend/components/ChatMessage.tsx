'use client'
import { ThumbsUp, ThumbsDown, Zap, Bot } from 'lucide-react'
import type { ChatMessage as Msg } from '@/lib/types'
import SourcePanel from './SourcePanel'
import EvalScorecard from './EvalScorecard'
import MarkdownContent from './MarkdownContent'

interface Props {
  message: Msg
  archIcon?: string
  onFeedback?: (messageId: string, rating: number) => void
}

export default function ChatMessage({ message, archIcon, onFeedback }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[78%] relative">
          <div className="px-4 py-3 rounded-2xl rounded-tr-sm bg-gradient-to-br from-violet-600/20 to-violet-800/10 border border-violet-500/25 text-sm text-slate-100 leading-relaxed">
            {message.content}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 animate-fade-in">
      {/* Avatar */}
      <div className="flex-shrink-0 mt-0.5">
        <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-violet-600/30 to-violet-900/30 border border-violet-500/20 flex items-center justify-center text-sm">
          {archIcon || <Bot size={13} className="text-violet-400" />}
        </div>
      </div>

      <div className="flex-1 min-w-0 max-w-[88%]">
        {/* Meta row */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-medium text-slate-400">{message.arch}</span>
          {message.elapsed != null && (
            <span className="text-[10px] font-mono text-slate-600 bg-white/[0.04] px-1.5 py-0.5 rounded-md">
              {message.elapsed.toFixed(2)}s
            </span>
          )}
          {message.cached && (
            <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 bg-amber-400/10 text-amber-400 rounded-md border border-amber-400/20 font-semibold">
              <Zap size={8} />CACHED
            </span>
          )}
        </div>

        {/* Message bubble */}
        <div className="px-4 py-3.5 rounded-2xl rounded-tl-sm bg-surface border border-white/[0.07] text-sm leading-relaxed shadow-card">
          <MarkdownContent content={message.content} />
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <SourcePanel sources={message.sources} />
        )}

        {/* Eval */}
        {message.eval && <EvalScorecard scores={message.eval} />}

        {/* Feedback */}
        {onFeedback && (
          <div className="flex items-center gap-1.5 mt-2">
            <button
              onClick={() => onFeedback(message.id, message.feedback === 'up' ? 0 : 1)}
              title={message.feedback === 'up' ? 'Remove rating' : 'Helpful'}
              className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] transition-all ${
                message.feedback === 'up'
                  ? 'text-emerald-400 bg-emerald-400/12 border border-emerald-400/25'
                  : 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05] border border-transparent'
              }`}
            >
              <ThumbsUp size={11} />
              {message.feedback === 'up' && <span>Helpful</span>}
            </button>
            <button
              onClick={() => onFeedback(message.id, message.feedback === 'down' ? 0 : -1)}
              title={message.feedback === 'down' ? 'Remove rating' : 'Unhelpful'}
              className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] transition-all ${
                message.feedback === 'down'
                  ? 'text-red-400 bg-red-400/12 border border-red-400/25'
                  : 'text-slate-600 hover:text-slate-300 hover:bg-white/[0.05] border border-transparent'
              }`}
            >
              <ThumbsDown size={11} />
              {message.feedback === 'down' && <span>Unhelpful</span>}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
