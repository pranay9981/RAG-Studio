export interface ArchInfo {
  key: string
  icon: string
  label: string
  tagline: string
  how: string
  best_for: string
  state_key: string
}

export interface Source {
  text: string
  source: string
  score?: number
}

export interface EvalScore {
  faithfulness: number
  relevance: number
  context_precision: number
  context_recall: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  arch?: string
  elapsed?: number
  sources?: Source[]
  eval?: EvalScore
  feedback?: 'up' | 'down'
  cached?: boolean
  chunk_ids?: string[]
}

export interface DocItem {
  name: string
  chunks: number
}

export interface HistoryItem {
  query: string
  arch: string
  elapsed: number
  answer: string
}

export interface StreamingState {
  steps: string[]
  tokens: string
  sources: Source[]
  isStreaming: boolean
}

export interface CompareResult {
  arch_key: string
  answer: string
  elapsed: number
  error?: string
  eval?: EvalScore
}

export interface AnalyticsArchData {
  query_count: number
  avg_elapsed: number
  avg_faithfulness: number | null
  avg_relevance: number | null
  avg_context_precision: number | null
  avg_context_recall: number | null
  cache_hits: number
  feedback_positive: number
  feedback_total: number
}

export interface AnalyticsData {
  data: Record<string, AnalyticsArchData>
  recent: Array<{ arch_key: string; query: string; elapsed: number; ts: number }>
}
