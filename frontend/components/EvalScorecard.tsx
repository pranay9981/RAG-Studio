import type { EvalScore } from '@/lib/types'

interface Props { scores: EvalScore }

function ScorePill({ label, score }: { label: string; score: number }) {
  const color = score >= 7 ? 'text-green-400 bg-green-400/10 border-green-400/20'
    : score >= 4 ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
    : 'text-red-400 bg-red-400/10 border-red-400/20'
  return (
    <div className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg border text-xs font-medium ${color}`}>
      <span className="text-base font-bold">{score}</span>
      <span className="text-[10px] opacity-80">{label}</span>
    </div>
  )
}

export default function EvalScorecard({ scores }: Props) {
  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="text-[10px] text-slate-500 mr-1">RAG Eval</span>
      <ScorePill label="Faithful" score={scores.faithfulness} />
      <ScorePill label="Relevant" score={scores.relevance} />
      <ScorePill label="Precision" score={scores.context_precision} />
    </div>
  )
}
