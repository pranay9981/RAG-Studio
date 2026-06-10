import type { ArchInfo, Source, EvalScore, CompareResult, DocItem, HistoryItem, AnalyticsData } from './types'

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8000'

export async function getArchitectures(): Promise<ArchInfo[]> {
  const r = await fetch(`${BASE}/api/architectures`)
  return r.json()
}

export async function getSession(sessionId: string) {
  const r = await fetch(`${BASE}/api/sessions/${sessionId}`)
  return r.json()
}

export async function resetSession(sessionId: string) {
  await fetch(`${BASE}/api/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function ingestFile(file: File, archKeys: string[], sessionId = 'default'): Promise<{ chunks: number; source: string }> {
  const fd = new FormData()
  fd.append('session_id', sessionId)
  fd.append('arch_keys', JSON.stringify(archKeys))
  fd.append('file', file)
  const r = await fetch(`${BASE}/api/ingest`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error((await r.json()).detail || 'Ingest failed')
  return r.json()
}

export async function ingestUrl(url: string, archKeys: string[], sessionId = 'default'): Promise<{ chunks: number; source: string }> {
  const fd = new FormData()
  fd.append('session_id', sessionId)
  fd.append('arch_keys', JSON.stringify(archKeys))
  fd.append('url', url)
  const r = await fetch(`${BASE}/api/ingest`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error((await r.json()).detail || 'Ingest failed')
  return r.json()
}

export function streamQuery(
  query: string,
  archKey: string,
  sessionId = 'default',
  callbacks: {
    onStep: (s: string) => void
    onToken: (t: string) => void
    onSources: (s: Source[]) => void
    onDone: (answer: string, elapsed: number, cached: boolean) => void
    onError: (e: string) => void
  },
): () => void {
  const params = new URLSearchParams({ session_id: sessionId, query, arch_key: archKey })
  const es = new EventSource(`${BASE}/api/query?${params}`)

  es.onmessage = (e) => {
    const d = JSON.parse(e.data)
    if (d.type === 'step') callbacks.onStep(d.content)
    else if (d.type === 'token') callbacks.onToken(d.content)
    else if (d.type === 'sources') callbacks.onSources(d.content)
    else if (d.type === 'done') { callbacks.onDone(d.answer, d.elapsed, d.cached || false); es.close() }
    else if (d.type === 'error') { callbacks.onError(d.content); es.close() }
  }
  es.onerror = () => { callbacks.onError('Connection error'); es.close() }

  return () => es.close()
}

export async function compareAll(query: string, sessionId = 'default'): Promise<CompareResult[]> {
  const r = await fetch(`${BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  })
  const data = await r.json()
  return data.results
}

export async function evaluateAnswer(query: string, answer: string, sources: Source[], archKey = ''): Promise<EvalScore> {
  const r = await fetch(`${BASE}/api/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, answer, sources, arch_key: archKey }),
  })
  return r.json()
}

export async function submitFeedback(query: string, archKey: string, chunkIds: string[], rating: number): Promise<void> {
  await fetch(`${BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, arch_key: archKey, chunk_ids: chunkIds, rating }),
  })
}

export async function getAnalytics(): Promise<AnalyticsData> {
  const r = await fetch(`${BASE}/api/analytics`)
  return r.json()
}

export async function getGraphHtml(): Promise<string> {
  const r = await fetch(`${BASE}/api/graph`)
  const data = await r.json()
  return data.html || ''
}

export async function getHistory(sessionId = 'default'): Promise<HistoryItem[]> {
  const r = await fetch(`${BASE}/api/history?session_id=${sessionId}`)
  const data = await r.json()
  return data.history
}

export async function deleteDocument(source: string): Promise<{ deleted: number }> {
  const r = await fetch(`${BASE}/api/documents`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  })
  if (!r.ok) throw new Error((await r.json()).detail || 'Delete failed')
  return r.json()
}

export async function getConfigStatus(): Promise<{ has_key: boolean }> {
  const r = await fetch(`${BASE}/api/config/status`)
  return r.json()
}

export async function setApiKey(apiKey: string): Promise<{ status: string }> {
  const r = await fetch(`${BASE}/api/config/apikey`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey }),
  })
  if (!r.ok) throw new Error((await r.json()).detail || 'Failed to set API key')
  return r.json()
}
