'use client'
import { ThumbsUp, ThumbsDown, Zap } from 'lucide-react'
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
        {message.cached && (
          <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 bg-yellow-400/10 text-yellow-400 rounded border border-yellow-400/20 font-medium">
            <Zap size={8} />cached
          </span>
        )}
      </div>
      <div className="max-w-[88%] px-4 py-3 rounded-2xl rounded-tl-sm bg-surface border border-white/[0.06] text-sm leading-relaxed">
        <MarkdownContent content={message.content} />
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
      {onFeedback && (
        <div className="flex items-center gap-1.5 mt-0.5">
          <button
            onClick={() => onFeedback(message.id, 1)}
            className={`p-1.5 rounded-lg transition-colors text-xs ${
              message.feedback === 'up'
                ? 'text-green-400 bg-green-400/15 border border-green-400/25'
                : 'text-slate-600 hover:text-slate-400 hover:bg-white/[0.04]'
            }`}
            title="Good answer"
          >
            <ThumbsUp size={12} />
          </button>
          <button
            onClick={() => onFeedback(message.id, -1)}
            className={`p-1.5 rounded-lg transition-colors text-xs ${
              message.feedback === 'down'
                ? 'text-red-400 bg-red-400/15 border border-red-400/25'
                : 'text-slate-600 hover:text-slate-400 hover:bg-white/[0.04]'
            }`}
            title="Bad answer"
          >
            <ThumbsDown size={12} />
          </button>
          {message.feedback && (
            <span className="text-[10px] text-slate-600">
              {message.feedback === 'up' ? 'Marked helpful' : 'Marked unhelpful'}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
