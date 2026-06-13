'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import { Send, Loader2, Network, X, HelpCircle } from 'lucide-react'
import ApiKeyModal from '@/components/ApiKeyModal'
import Sidebar from '@/components/Sidebar'
import ArchCard from '@/components/ArchCard'
import ChatMessage from '@/components/ChatMessage'
import BrainWorking from '@/components/BrainWorking'
import SourcePanel from '@/components/SourcePanel'
import CompareGrid from '@/components/CompareGrid'
import DocumentManager from '@/components/DocumentManager'
import AnalyticsDashboard from '@/components/AnalyticsDashboard'
import ArchExplainer from '@/components/ArchExplainer'
import {
  getArchitectures, streamQuery, compareAll, evaluateAnswer,
  resetSession, getGraphHtml, submitFeedback, getAnalytics, getConfigStatus,
  listDocuments, getHistory, clearCache, getHealth,
} from '@/lib/api'
import type { ArchInfo, ChatMessage as Msg, Source, EvalScore, DocItem, HistoryItem, CompareResult, AnalyticsData } from '@/lib/types'
import { v4 as uuidv4 } from 'uuid'

function downloadMd(content: string, filename: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/markdown' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `${filename}.md`
  a.click()
  URL.revokeObjectURL(url)
}

const SESSION_ID = 'default'
const GRAPH_ARCH = '02 Graph RAG (Knowledge Graphs)'
const COMPARE_KEY = '__compare__'
const LS_KEY = 'rag-studio-messages'

export default function Page() {
  const [archs, setArchs] = useState<ArchInfo[]>([])
  const [selectedArch, setSelectedArch] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const [enableEval, setEnableEval] = useState(false)

  const [allMessages, setAllMessages] = useState<Record<string, Msg[]>>({})
  const [allCompareResults, setAllCompareResults] = useState<Record<string, CompareResult[]>>({})

  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [steps, setSteps] = useState<string[]>([])
  const [tokens, setTokens] = useState('')
  const [liveSource, setLiveSource] = useState<Source[]>([])
  const [ingestedArchs, setIngestedArchs] = useState<Set<string>>(new Set())
  const [docLibrary, setDocLibrary] = useState<DocItem[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [graphHtml, setGraphHtml] = useState('')
  const [showGraph, setShowGraph] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null)
  const [showExplainer, setShowExplainer] = useState(false)
  const [showApiKey, setShowApiKey] = useState(false)
  const [bgeM3Loaded, setBgeM3Loaded] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => { cleanupRef.current?.() }
  }, [])

  // Load architectures + hydrate doc library + check API key
  useEffect(() => {
    getArchitectures().then(list => {
      setArchs(list)
      if (list.length) {
        setSelectedArch(list[0].key)
        // Hydrate doc library from ChromaDB (persisted across server restarts)
        listDocuments(list[0].key).then(docs => {
          if (docs.length > 0) {
            setDocLibrary(docs)
            setIngestedArchs(new Set(list.map(a => a.key)))
          }
        })
      }
    })
    getConfigStatus().then(({ has_key }) => { if (!has_key) setShowApiKey(true) }).catch(() => {})
    getHistory().then(h => { if (h.length) setHistory(h) })
    // Poll BGE-M3 load status every 5 s until loaded
    getHealth().then(h => setBgeM3Loaded(h.bge_m3_loaded))
    const bgeInterval = setInterval(() => {
      getHealth().then(h => {
        setBgeM3Loaded(h.bge_m3_loaded)
        if (h.bge_m3_loaded) clearInterval(bgeInterval)
      })
    }, 5000)
    return () => clearInterval(bgeInterval)
  }, [])

  // Session persistence — restore messages from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(LS_KEY)
      if (saved) setAllMessages(JSON.parse(saved))
    } catch {}
  }, [])

  // Persist messages to localStorage on every change
  useEffect(() => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(allMessages))
    } catch {}
  }, [allMessages])

  const chatKey = compareMode ? COMPARE_KEY : selectedArch
  const currentMessages = allMessages[chatKey] ?? []
  const currentCompareResults = allCompareResults[chatKey] ?? []

  const messageCounts: Record<string, number> = {}
  for (const [k, msgs] of Object.entries(allMessages)) {
    messageCounts[k] = msgs.filter(m => m.role === 'assistant').length
  }

  const appendMsg = useCallback((key: string, msg: Msg) => {
    setAllMessages(prev => ({ ...prev, [key]: [...(prev[key] ?? []), msg] }))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentMessages, tokens, steps, currentCompareResults])

  const currentArch = archs.find(a => a.key === selectedArch)

  const handleOpenGraph = useCallback(async () => {
    const html = await getGraphHtml()
    setGraphHtml(html)
    setShowGraph(true)
  }, [])

  const handleOpenAnalytics = useCallback(async () => {
    try {
      const data = await getAnalytics()
      setAnalyticsData(data)
      setShowAnalytics(true)
    } catch {}
  }, [])

  const handleFeedback = useCallback(async (messageId: string, rating: number) => {
    const msgs = allMessages[chatKey] ?? []
    const msgIdx = msgs.findIndex(m => m.id === messageId)
    if (msgIdx === -1) return
    const msg = msgs[msgIdx]
    if (msg.role !== 'assistant') return

    const lastUserMsg = msgs.slice(0, msgIdx).reverse().find(m => m.role === 'user')
    const query = lastUserMsg?.content || ''

    try {
      await submitFeedback(query, msg.arch || selectedArch, msg.chunk_ids || [], rating)
      setAllMessages(prev => ({
        ...prev,
        [chatKey]: (prev[chatKey] ?? []).map(m =>
          m.id === messageId
            ? { ...m, feedback: rating === 0 ? undefined : rating > 0 ? 'up' : 'down' }
            : m
        ),
      }))
    } catch {}
  }, [allMessages, chatKey, selectedArch, compareMode])

  const handleSend = useCallback(async () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')

    const key = compareMode ? COMPARE_KEY : selectedArch
    appendMsg(key, { id: uuidv4(), role: 'user', content: q })

    if (compareMode) {
      setCompareLoading(true)
      setAllCompareResults(prev => ({ ...prev, [key]: [] }))
      try {
        const results = await compareAll(q, SESSION_ID)
        let final = results
        if (enableEval) {
          final = await Promise.all(
            results.map(async r => {
              if (r.error || !r.answer) return r
              try {
                const evalScore = await evaluateAnswer(q, r.answer, r.sources ?? [], r.arch_key)
                return { ...r, eval: evalScore }
              } catch { return r }
            })
          )
        }
        setAllCompareResults(prev => ({ ...prev, [key]: final }))
        const maxElapsed = results.length ? Math.max(...results.map(r => r.elapsed)) : 0
        setHistory(prev => [...prev, { query: q, arch: 'Compare All', elapsed: maxElapsed, answer: '' }])
      } catch (e) {
        appendMsg(key, {
          id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(),
          role: 'assistant',
          content: `Compare failed: ${e instanceof Error ? e.message : 'Unknown error'}`,
          arch: 'Compare All',
          ts: new Date().toISOString(),
        })
      } finally {
        setCompareLoading(false)
      }
      return
    }

    setIsStreaming(true)
    setSteps([])
    setTokens('')
    setLiveSource([])

    let collectedSources: Source[] = []

    let localTokens = ''
    cleanupRef.current = streamQuery(q, selectedArch, SESSION_ID, {
      onStep: s => setSteps(prev => [...prev, s]),
      onToken: t => { localTokens += t; setTokens(prev => prev + t) },
      onSources: s => { collectedSources = s; setLiveSource(s) },
      onDone: async (answer, elapsed, cached) => {
        const finalContent = answer || localTokens
        let evalScore: EvalScore | undefined
        if (enableEval && finalContent) {
          try { evalScore = await evaluateAnswer(q, finalContent, collectedSources, selectedArch) } catch {}
        }
        appendMsg(selectedArch, {
          id: uuidv4(),
          role: 'assistant',
          content: finalContent,
          arch: selectedArch,
          elapsed,
          sources: collectedSources,
          eval: evalScore,
          cached,
          chunk_ids: collectedSources.map(s => s.text.slice(0, 80)),
        })
        setHistory(prev => [...prev, { query: q, arch: selectedArch, elapsed, answer: answer.slice(0, 120) }])
        setIsStreaming(false)
        setSteps([])
        setTokens('')
        setLiveSource([])
      },
      onError: err => {
        appendMsg(selectedArch, { id: uuidv4(), role: 'assistant', content: `⚠️ ${err}`, arch: selectedArch })
        setIsStreaming(false)
        setSteps([])
        setTokens('')
      },
    })
  }, [input, isStreaming, compareMode, selectedArch, enableEval, appendMsg])

  const handleClearChat = () => {
    setAllMessages(prev => ({ ...prev, [chatKey]: [] }))
    setAllCompareResults(prev => ({ ...prev, [chatKey]: [] }))
  }

  const handleReset = async () => {
    await resetSession(SESSION_ID)
    setAllMessages({})
    setAllCompareResults({})
    setIngestedArchs(new Set())
    setDocLibrary([])
    setHistory([])
    setGraphHtml('')
    try { localStorage.removeItem(LS_KEY) } catch {}
  }

  const handleDeleteDoc = useCallback((source: string) => {
    setDocLibrary(prev => prev.filter(d => d.name !== source))
  }, [])

  const handleExportCurrentChat = useCallback(() => {
    if (compareMode) {
      if (!currentCompareResults.length) return
      const compareMsgs = allMessages[COMPARE_KEY] ?? []
      const lastUserMsg = [...compareMsgs].reverse().find(m => m.role === 'user')
      const query = lastUserMsg?.content ?? 'Query'
      const md = `# RAG Studio — Compare Mode\n\n**Query:** ${query}\n\n` +
        currentCompareResults.map(r => {
          const evalStr = r.eval
            ? `\n\n**Eval:** Faithful ${r.eval.faithfulness}/10 · Relevant ${r.eval.relevance}/10 · Precision ${r.eval.context_precision}/10 · Recall ${r.eval.context_recall}/10`
            : ''
          const errStr = r.error ? `\n\n⚠️ Error: ${r.error}` : ''
          return `## ${r.arch_key}\n\n*${r.elapsed}s*\n\n${r.answer || '(no answer)'}${evalStr}${errStr}`
        }).join('\n\n---\n\n')
      downloadMd(md, 'rag-compare-mode')
    } else {
      if (!currentMessages.length) return
      const title = selectedArch
      const md = `# RAG Studio — ${title}\n\n` +
        currentMessages
          .map(m => `**${m.role === 'user' ? 'You' : m.arch || 'Assistant'}:**\n${m.content}`)
          .join('\n\n---\n\n')
      downloadMd(md, `rag-${title.slice(0, 20).replace(/\s+/g, '-').toLowerCase()}`)
    }
  }, [currentMessages, currentCompareResults, compareMode, selectedArch, allMessages])

  const handleExportAllChats = useCallback(() => {
    const sections: string[] = []
    for (const [key, msgs] of Object.entries(allMessages)) {
      if (key === COMPARE_KEY) {
        const compareResults = allCompareResults[COMPARE_KEY] ?? []
        const lastUserMsg = [...msgs].reverse().find(m => m.role === 'user')
        const query = lastUserMsg?.content ?? ''
        if (compareResults.length) {
          sections.push(
            `# Compare Mode\n\n**Query:** ${query}\n\n` +
            compareResults.map(r => {
              const evalStr = r.eval
                ? `\n\n**Eval:** Faithful ${r.eval.faithfulness}/10 · Relevant ${r.eval.relevance}/10 · Precision ${r.eval.context_precision}/10 · Recall ${r.eval.context_recall}/10`
                : ''
              const errStr = r.error ? `\n\n⚠️ Error: ${r.error}` : ''
              return `## ${r.arch_key}\n\n*${r.elapsed}s*\n\n${r.answer || '(no answer)'}${evalStr}${errStr}`
            }).join('\n\n---\n\n')
          )
        } else if (query) {
          sections.push(`# Compare Mode\n\n**You:**\n${query}`)
        }
        continue
      }
      if (!msgs.length) continue
      sections.push(
        `# ${key}\n\n` +
        msgs.map(m => `**${m.role === 'user' ? 'You' : m.arch || 'Assistant'}:**\n${m.content}`)
          .join('\n\n---\n\n')
      )
    }
    if (!sections.length) return
    downloadMd(sections.join('\n\n\n---\n\n\n'), 'rag-all-chats')
  }, [allMessages, allCompareResults])

  const handleExportCompare = useCallback(() => {
    if (!currentCompareResults.length) return
    const compareMsgs = allMessages[COMPARE_KEY] ?? []
    const lastUserMsg = [...compareMsgs].reverse().find(m => m.role === 'user')
    const query = lastUserMsg?.content ?? 'Query'
    const md = `# Compare Results\n\n**Query:** ${query}\n\n` +
      currentCompareResults.map(r => {
        const evalStr = r.eval
          ? `\n\n**Eval:** Faithful ${r.eval.faithfulness}/10 · Relevant ${r.eval.relevance}/10 · Precision ${r.eval.context_precision}/10 · Recall ${r.eval.context_recall}/10`
          : ''
        const errStr = r.error ? `\n\n⚠️ Error: ${r.error}` : ''
        return `## ${r.arch_key}\n\n*${r.elapsed}s*\n\n${r.answer || '(no answer)'}${evalStr}${errStr}`
      }).join('\n\n---\n\n')
    downloadMd(md, 'rag-compare-results')
  }, [currentCompareResults, allMessages])

  const handleClearCache = useCallback(async () => {
    try { await clearCache() } catch {}
  }, [])

  const showGraphBtn = selectedArch === GRAPH_ARCH && ingestedArchs.has(GRAPH_ARCH) && !compareMode
  const isEmpty = currentMessages.length === 0 && !isStreaming && currentCompareResults.length === 0

  return (
    <div className="flex h-screen bg-base overflow-hidden">
      <Sidebar
        architectures={archs}
        selectedArch={selectedArch}
        compareMode={compareMode}
        enableEval={enableEval}
        ingestedArchs={ingestedArchs}
        messageCounts={messageCounts}
        docLibrary={docLibrary}
        history={history}
        onSelectArch={k => { if (!isStreaming) setSelectedArch(k) }}
        onCompareToggle={() => setCompareMode(o => !o)}
        onEvalToggle={() => setEnableEval(o => !o)}
        onClearChat={handleClearChat}
        onReset={handleReset}
        exportOptions={[
          { label: 'Current Chat', fn: handleExportCurrentChat },
          { label: 'All Chats', fn: handleExportAllChats },
          { label: 'Compare Results', fn: handleExportCompare },
        ]}
        onClearCache={handleClearCache}
        onAnalytics={handleOpenAnalytics}
        onSettings={() => setShowApiKey(true)}
        bgeM3Loaded={bgeM3Loaded}
      >
        <DocumentManager
          archKeys={archs.map(a => a.key)}
          onIngested={(source, chunks) => {
            setDocLibrary(prev => [...prev, { name: source, chunks }])
            setIngestedArchs(new Set(archs.map(a => a.key)))
          }}
          docLibrary={docLibrary}
          onDocDeleted={handleDeleteDoc}
        />
      </Sidebar>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        {currentArch && !compareMode && (
          <div className="flex items-center justify-between pr-4">
            <ArchCard arch={currentArch} />
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => setShowExplainer(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-slate-400 text-xs font-medium hover:bg-white/[0.07] hover:text-slate-200 transition-colors"
              >
                <HelpCircle size={12} />
                How it works
              </button>
              {showGraphBtn && (
                <button
                  onClick={handleOpenGraph}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-500/10 border border-violet-500/25 text-violet-300 text-xs font-medium hover:bg-violet-500/20 transition-colors"
                >
                  <Network size={13} />
                  Knowledge Graph
                </button>
              )}
            </div>
          </div>
        )}
        {compareMode && (
          <div className="px-6 py-3 border-b border-white/[0.06] bg-violet-500/[0.06] flex items-center gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse-slow" />
            <span className="text-xs font-semibold text-violet-300">Compare Mode</span>
            <span className="text-xs text-slate-500">— all {archs.length} architectures run simultaneously</span>
          </div>
        )}

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {isEmpty && (
            <div className="flex flex-col items-center justify-center h-full gap-5 text-center select-none">
              <div className="relative">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-600/30 to-violet-900/20 border border-violet-500/25 flex items-center justify-center shadow-glow">
                  <span className="text-2xl">⚡</span>
                </div>
                <div className="absolute inset-0 rounded-2xl bg-violet-500/10 blur-xl -z-10" />
              </div>
              <div className="space-y-1.5">
                <p className="text-base font-bold text-slate-200">RAG Studio</p>
                <p className="text-sm text-slate-500 max-w-xs leading-relaxed">
                  {compareMode
                    ? 'Upload a document, then ask a question to compare all 10 architectures side by side.'
                    : 'Upload a document or paste a URL, then start asking questions.'}
                </p>
              </div>
              {ingestedArchs.size === 0 && (
                <div className="flex items-center gap-2 text-xs text-slate-600 bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-2.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
                  Upload a document from the sidebar to begin
                </div>
              )}
            </div>
          )}

          {currentMessages.map(msg => (
            <ChatMessage
              key={msg.id}
              message={msg}
              archIcon={archs.find(a => a.key === msg.arch)?.icon}
              onFeedback={msg.role === 'assistant' ? handleFeedback : undefined}
            />
          ))}

          {isStreaming && (
            <div className="animate-fade-in">
              <BrainWorking steps={steps} tokens={tokens} isStreaming={isStreaming} archIcon={currentArch?.icon || '🤖'} archLabel={currentArch?.label || selectedArch} />
              {liveSource.length > 0 && <div className="max-w-[88%] mt-2"><SourcePanel sources={liveSource} /></div>}
            </div>
          )}

          {currentCompareResults.length > 0 && (
            <div className="animate-fade-in">
              <p className="text-xs text-slate-500 mb-3">
                Results for: <span className="text-slate-300">{currentMessages.filter(m => m.role === 'user').slice(-1)[0]?.content}</span>
              </p>
              <CompareGrid results={currentCompareResults} architectures={archs} loading={false} />
            </div>
          )}

          {compareLoading && <CompareGrid results={[]} architectures={archs} loading={true} />}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-6 py-4 border-t border-white/[0.06] bg-surface">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
                onInput={e => {
                  const el = e.currentTarget
                  el.style.height = 'auto'
                  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
                }}
                placeholder={
                  ingestedArchs.size === 0
                    ? 'Upload a document first…'
                    : 'Ask a question… (Enter ↵ to send)'
                }
                rows={1}
                className="w-full resize-none bg-white/[0.04] border border-white/[0.08] rounded-2xl px-4 py-3 pr-4 text-sm text-slate-200 placeholder:text-slate-600 outline-none focus:border-violet-500/40 focus:bg-violet-500/[0.03] transition-all leading-relaxed"
                style={{ maxHeight: '120px', overflowY: 'auto' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming || compareLoading}
              className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-violet-700 hover:from-violet-500 hover:to-violet-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center shadow-glow-sm"
            >
              {isStreaming || compareLoading
                ? <Loader2 size={15} className="animate-spin text-white" />
                : <Send size={15} className="text-white" />}
            </button>
          </div>
          <p className="text-center text-[10px] text-slate-700 mt-2">Shift+Enter for new line</p>
        </div>
      </div>

      {/* Knowledge Graph Modal */}
      {showGraph && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm" onClick={() => setShowGraph(false)}>
          <div
            className="bg-[#0e0e1a] border border-white/[0.1] rounded-2xl overflow-hidden shadow-2xl flex flex-col"
            style={{ width: '90vw', height: '85vh' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <Network size={15} className="text-violet-400" />
                <span className="text-sm font-semibold text-slate-200">Knowledge Graph</span>
                <span className="text-xs text-slate-500">Graph RAG entity relationships</span>
              </div>
              <button onClick={() => setShowGraph(false)} className="text-slate-500 hover:text-slate-300 transition-colors">
                <X size={16} />
              </button>
            </div>
            {graphHtml
              ? <iframe srcDoc={graphHtml} className="flex-1 w-full border-0" sandbox="allow-scripts" title="Knowledge Graph" />
              : <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">No graph data yet — ingest a document first</div>
            }
          </div>
        </div>
      )}

      {/* Analytics Modal */}
      {showAnalytics && analyticsData && (
        <AnalyticsDashboard data={analyticsData} onClose={() => setShowAnalytics(false)} />
      )}

      {/* Arch Explainer Modal */}
      {showExplainer && currentArch && (
        <ArchExplainer arch={currentArch} onClose={() => setShowExplainer(false)} />
      )}

      {/* API Key Modal */}
      {showApiKey && (
        <ApiKeyModal
          onClose={() => setShowApiKey(false)}
          onKeySet={() => setShowApiKey(false)}
        />
      )}
    </div>
  )
}
