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
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  arch?: string
  elapsed?: number
  sources?: Source[]
  eval?: EvalScore
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
