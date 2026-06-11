import type { EvalScore } from '@/lib/types'

interface Props { scores: EvalScore }

function ScoreBar({ label, score }: { label: string; score: number | null | undefined }) {
  if (score == null) return null
  const pct = Math.round((score / 10) * 100)
  const color = score >= 7 ? '#10b981' : score >= 4 ? '#f59e0b' : '#ef4444'
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="text-[10px] text-slate-500 w-14 flex-shrink-0">{label}</span>
      <div className="flex-1 score-bar">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[11px] font-semibold font-mono w-5 text-right flex-shrink-0" style={{ color }}>
        {score}
      </span>
    </div>
  )
}

export default function EvalScorecard({ scores }: Props) {
  const avg = [scores.faithfulness, scores.relevance, scores.context_precision, scores.context_recall]
    .filter(v => v != null && v > 0)
  const avgScore = avg.length ? Math.round(avg.reduce((a, b) => a + b!, 0) / avg.length) : null
  const avgColor = avgScore == null ? '#475569' : avgScore >= 7 ? '#10b981' : avgScore >= 4 ? '#f59e0b' : '#ef4444'

  return (
    <div className="mt-2.5 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">RAG Eval</span>
        {avgScore != null && (
          <span className="text-[11px] font-bold font-mono px-2 py-0.5 rounded-md" style={{ color: avgColor, background: `${avgColor}18` }}>
            {avgScore}/10 avg
          </span>
        )}
      </div>
      <ScoreBar label="Faithful" score={scores.faithfulness} />
      <ScoreBar label="Relevant" score={scores.relevance} />
      <ScoreBar label="Precision" score={scores.context_precision} />
      <ScoreBar label="Recall" score={scores.context_recall} />
    </div>
  )
}
