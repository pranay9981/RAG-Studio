'use client'
import { X } from 'lucide-react'
import type { ArchInfo } from '@/lib/types'

interface Props {
  arch: ArchInfo
  onClose: () => void
}

const PIPELINE_STEPS: Record<string, string[]> = {
  '01 Hybrid RAG (Dense + Sparse)': ['Embed Query', 'Dense Retrieval', 'BM25 Retrieval', 'RRF Fusion (k=60)', 'Cross-Encoder Rerank', 'Context Eval', 'Gemini Generate'],
  '02 Graph RAG (Knowledge Graphs)': ['Extract Entities', 'Graph Traversal', 'Dense Retrieval', 'Context Eval', 'Combine Context', 'Gemini Generate'],
  '03 Agentic RAG (LangGraph)': ['Planner Agent', 'Route Decision', 'Vector / Web / Direct', 'Multi-hop Decompose', 'Reasoner Agent', 'Gemini Generate'],
  '04 Corrective RAG (CRAG)': ['Retrieve', 'Evaluate Quality', 'CORRECT → Generate', 'AMBIGUOUS → Rewrite + Web', 'INCORRECT → Web Search', 'Gemini Generate'],
  '05 Multimodal RAG (Vision + Text)': ['Embed Query', 'Retrieve Text + Images', 'Build Multimodal Prompt', 'Gemini Vision Generate'],
  '06 Multilingual RAG (BGE-M3)': ['Cross-lingual Embed', 'Cross-lingual Retrieval', 'Cross-Encoder Rerank', 'Context Eval', 'Gemini Generate (same language)'],
  '07 RAG-Fusion (Query Expansion)': ['Generate 4 Sub-queries', 'Retrieve for Each (×4)', 'RRF Fusion', 'Context Eval', 'Gemini Generate'],
  '08 HyDE RAG (Hypothetical Document)': ['Generate Hypothetical Answer', 'Embed Hypothetical', 'Retrieve Real Docs', 'Context Eval', 'Gemini Generate'],
}

const ADAPTIVE_FEATURES = [
  { icon: '⚡', label: 'Semantic Cache', desc: 'Similar past queries return instantly (cosine > 0.92)' },
  { icon: '🔍', label: 'Context Quality Eval', desc: 'Checks if retrieved context is CORRECT/AMBIGUOUS/INCORRECT before generating' },
  { icon: '🌐', label: 'Web Fallback', desc: 'Auto-fetches from DuckDuckGo if local context is insufficient' },
  { icon: '👍', label: 'Feedback Tracking', desc: 'Thumbs up/down ratings stored and reflected in analytics' },
]

export default function ArchExplainer({ arch, onClose }: Props) {
  const steps = PIPELINE_STEPS[arch.key] || []

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-[#0e0e1a] border border-white/[0.1] rounded-2xl overflow-hidden shadow-2xl flex flex-col"
        style={{ width: '90vw', maxWidth: '700px', maxHeight: '85vh' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{arch.icon}</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{arch.label}</p>
              <p className="text-xs text-slate-500">{arch.tagline}</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Pipeline flow */}
          {steps.length > 0 && (
            <div>
              <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-3">Pipeline Flow</p>
              <div className="flex flex-wrap items-center gap-1.5">
                {steps.map((step, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    {i > 0 && <span className="text-slate-600 text-xs">→</span>}
                    <span className="px-2.5 py-1 bg-white/[0.04] rounded-lg border border-white/[0.08] text-xs text-slate-300 whitespace-nowrap">
                      {step}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* How it works */}
          <div>
            <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2">How It Works</p>
            <p className="text-sm text-slate-300 leading-relaxed">{arch.how}</p>
          </div>

          {/* Best for */}
          <div className="px-4 py-3 bg-indigo-500/5 border border-indigo-500/20 rounded-xl">
            <p className="text-[10px] font-medium text-indigo-400 uppercase tracking-wider mb-1">Best For</p>
            <p className="text-sm text-slate-300">{arch.best_for}</p>
          </div>

          {/* Adaptive features (common to all) */}
          <div>
            <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-3">Adaptive RAG Features (All Architectures)</p>
            <div className="grid grid-cols-2 gap-2">
              {ADAPTIVE_FEATURES.map(f => (
                <div key={f.label} className="flex gap-2.5 px-3 py-2.5 bg-white/[0.02] rounded-lg border border-white/[0.06]">
                  <span className="text-base flex-shrink-0">{f.icon}</span>
                  <div>
                    <p className="text-xs font-medium text-slate-200">{f.label}</p>
                    <p className="text-[11px] text-slate-500 leading-snug mt-0.5">{f.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
