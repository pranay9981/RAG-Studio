import type { ArchInfo, Source, EvalScore, CompareResult, DocItem, AnalyticsData, HistoryItem } from './types'

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8000'

export async function getArchitectures(): Promise<ArchInfo[]> {
  try {
    const r = await fetch(`${BASE}/api/architectures`)
    if (!r.ok) return []
    return r.json()
  } catch {
    return []
  }
}

export async function resetSession(sessionId: string) {
  try {
    await fetch(`${BASE}/api/sessions/${sessionId}`, { method: 'DELETE' })
  } catch {}
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
  let intentionallyClosed = false

  const timeoutId = setTimeout(() => {
    if (!intentionallyClosed) {
      intentionallyClosed = true
      es.close()
      callbacks.onError('Request timed out after 2 minutes')
    }
  }, 120000)

  es.onmessage = (e) => {
    let d: Record<string, unknown>
    try { d = JSON.parse(e.data) } catch { return }
    if (d.type === 'step') callbacks.onStep(d.content as string)
    else if (d.type === 'token') callbacks.onToken(d.content as string)
    else if (d.type === 'sources') callbacks.onSources(d.content as Source[])
    else if (d.type === 'done') {
      clearTimeout(timeoutId)
      intentionallyClosed = true
      callbacks.onDone(d.answer as string, d.elapsed as number, (d.cached as boolean) || false)
      es.close()
    }
    else if (d.type === 'error') {
      clearTimeout(timeoutId)
      intentionallyClosed = true
      callbacks.onError(d.content as string)
      es.close()
    }
  }
  es.onerror = () => {
    clearTimeout(timeoutId)
    if (intentionallyClosed) return
    callbacks.onError('Connection error — is the backend running?')
    es.close()
  }

  return () => { clearTimeout(timeoutId); intentionallyClosed = true; es.close() }
}

export async function compareAll(query: string, sessionId = 'default'): Promise<CompareResult[]> {
  const r = await fetch(`${BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  })
  if (!r.ok) throw new Error(`Compare failed: ${r.status}`)
  const data = await r.json()
  return (data.results ?? []) as CompareResult[]
}

export async function evaluateAnswer(query: string, answer: string, sources: Source[], archKey = ''): Promise<EvalScore> {
  try {
    const r = await fetch(`${BASE}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, answer, sources, arch_key: archKey }),
    })
    if (!r.ok) throw new Error(`Eval failed: ${r.status}`)
    return r.json()
  } catch (e) {
    console.error('[evaluateAnswer]', e)
    return { faithfulness: 0, relevance: 0, context_precision: 0, context_recall: 0 }
  }
}

export async function submitFeedback(query: string, archKey: string, chunkIds: string[], rating: number): Promise<void> {
  try {
    await fetch(`${BASE}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, arch_key: archKey, chunk_ids: chunkIds, rating }),
    })
  } catch {}
}

export async function getAnalytics(): Promise<AnalyticsData> {
  try {
    const r = await fetch(`${BASE}/api/analytics`)
    if (!r.ok) throw new Error(`Analytics failed: ${r.status}`)
    return r.json()
  } catch (e) {
    console.error('[getAnalytics]', e)
    return { data: {}, recent: [] }
  }
}

export async function getGraphHtml(): Promise<string> {
  try {
    const r = await fetch(`${BASE}/api/graph`)
    if (!r.ok) return ''
    const data = await r.json()
    return data.html || ''
  } catch {
    return ''
  }
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

export async function listDocuments(archKey: string): Promise<DocItem[]> {
  try {
    const r = await fetch(`${BASE}/api/documents?arch_key=${encodeURIComponent(archKey)}`)
    if (!r.ok) return []
    const data = await r.json()
    return (data.documents ?? []).map((d: { source: string; chunks: number }) => ({
      name: d.source,
      chunks: d.chunks,
    }))
  } catch {
    return []
  }
}

export async function getConfigStatus(): Promise<{ has_key: boolean }> {
  try {
    const r = await fetch(`${BASE}/api/config/status`)
    if (!r.ok) return { has_key: false }
    return r.json()
  } catch {
    return { has_key: false }
  }
}

export async function getHistory(): Promise<HistoryItem[]> {
  try {
    const r = await fetch(`${BASE}/api/history`)
    if (!r.ok) return []
    const data = await r.json()
    return (data.history ?? []).map((h: { query: string; arch: string; elapsed: number; answer: string }) => ({
      query: h.query,
      arch: h.arch,
      elapsed: h.elapsed,
      answer: h.answer,
    }))
  } catch {
    return []
  }
}

export async function clearCache(): Promise<{ deleted: number }> {
  const r = await fetch(`${BASE}/api/cache`, { method: 'DELETE' })
  if (!r.ok) throw new Error('Clear cache failed')
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
