'use client'
import { X, Clock, MessageSquare, ThumbsUp, Zap } from 'lucide-react'
import type { AnalyticsData } from '@/lib/types'

interface Props {
  data: AnalyticsData
  onClose: () => void
}

function Bar({ value, max = 10, color = 'bg-indigo-500' }: { value: number | null | undefined; max?: number; color?: string }) {
  if (value == null) return <span className="text-[10px] text-slate-600">n/a</span>
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono text-slate-400 flex-shrink-0 w-5 text-right">{value}</span>
    </div>
  )
}

const ARCH_LABELS: Record<string, string> = {
  '01 Hybrid RAG (Dense + Sparse)': '01 Hybrid',
  '02 Graph RAG (Knowledge Graphs)': '02 Graph',
  '03 Agentic RAG (LangGraph)': '03 Agentic',
  '04 Corrective RAG (CRAG)': '04 CRAG',
  '05 Multimodal RAG (Vision + Text)': '05 Multimodal',
  '06 Multilingual RAG (BGE-M3)': '06 Multilingual',
  '07 RAG-Fusion (Query Expansion)': '07 Fusion',
  '08 HyDE RAG (Hypothetical Document)': '08 HyDE',
  '09 Structured RAG (CSV/Excel)': '09 Structured',
  '10 Self-RAG (Reflection + Critique)': '10 Self-RAG',
}

export default function AnalyticsDashboard({ data, onClose }: Props) {
  const entries = Object.entries(data.data)
  const hasData = entries.length > 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-[#0e0e1a] border border-white/[0.1] rounded-2xl overflow-hidden shadow-2xl flex flex-col"
        style={{ width: '90vw', maxWidth: '900px', maxHeight: '85vh' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-200">Analytics Dashboard</span>
            <span className="text-xs text-slate-500">per-architecture performance</span>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {!hasData ? (
            <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
              <p className="text-3xl">📊</p>
              <p className="text-sm font-medium text-slate-300">No data yet</p>
              <p className="text-xs text-slate-500 max-w-xs">Start asking questions to build analytics. Enable RAG Evaluation for quality scores.</p>
            </div>
          ) : (
            <>
              {/* Stats table */}
              <div>
                <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-3">Architecture Performance</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[10px] text-slate-500 border-b border-white/[0.06]">
                        <th className="text-left pb-2 pr-4 font-medium">Architecture</th>
                        <th className="text-left pb-2 pr-4 font-medium">Queries</th>
                        <th className="text-left pb-2 pr-4 font-medium">Avg Latency</th>
                        <th className="text-left pb-2 pr-4 font-medium w-24">Faithful</th>
                        <th className="text-left pb-2 pr-4 font-medium w-24">Relevant</th>
                        <th className="text-left pb-2 pr-4 font-medium w-24">Precision</th>
                        <th className="text-left pb-2 pr-4 font-medium w-24">Recall</th>
                        <th className="text-left pb-2 pr-4 font-medium">Cache</th>
                        <th className="text-left pb-2 font-medium">Feedback</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {entries.map(([key, d]) => (
                        <tr key={key} className="hover:bg-white/[0.02] transition-colors">
                          <td className="py-2.5 pr-4 text-slate-300 font-medium whitespace-nowrap">
                            {ARCH_LABELS[key] || key.slice(0, 12)}
                          </td>
                          <td className="py-2.5 pr-4">
                            <span className="flex items-center gap-1 text-slate-400">
                              <MessageSquare size={10} />
                              {d.query_count}
                            </span>
                          </td>
                          <td className="py-2.5 pr-4">
                            <span className="flex items-center gap-1 text-slate-400">
                              <Clock size={10} />
                              {d.avg_elapsed}s
                            </span>
                          </td>
                          <td className="py-2.5 pr-4 w-24"><Bar value={d.avg_faithfulness} color="bg-green-500" /></td>
                          <td className="py-2.5 pr-4 w-24"><Bar value={d.avg_relevance} color="bg-blue-500" /></td>
                          <td className="py-2.5 pr-4 w-24"><Bar value={d.avg_context_precision} color="bg-purple-500" /></td>
                          <td className="py-2.5 pr-4 w-24"><Bar value={d.avg_context_recall} color="bg-cyan-500" /></td>
                          <td className="py-2.5 pr-4">
                            <span className="flex items-center gap-1 text-yellow-400">
                              <Zap size={9} />{d.cache_hits}
                            </span>
                          </td>
                          <td className="py-2.5">
                            {d.feedback_total > 0 ? (
                              <span className="flex items-center gap-1 text-green-400">
                                <ThumbsUp size={9} />
                                {Math.round((d.feedback_positive / d.feedback_total) * 100)}%
                              </span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Recent queries */}
              {data.recent.length > 0 && (
                <div>
                  <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-3">Recent Queries</p>
                  <div className="space-y-1.5">
                    {data.recent.map((r, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                        <span className="text-slate-500 flex-shrink-0 w-20 truncate">{ARCH_LABELS[r.arch_key] || r.arch_key.slice(0, 10)}</span>
                        <span className="flex-1 text-slate-300 truncate">{r.query}</span>
                        <span className="text-slate-600 flex-shrink-0 font-mono">{r.elapsed.toFixed(1)}s</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
