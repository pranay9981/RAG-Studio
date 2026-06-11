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
  listDocuments,
} from '@/lib/api'
import type { ArchInfo, ChatMessage as Msg, Source, EvalScore, DocItem, HistoryItem, CompareResult, AnalyticsData } from '@/lib/types'
import { v4 as uuidv4 } from 'uuid'

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

  const bottomRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

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
          m.id === messageId ? { ...m, feedback: rating > 0 ? 'up' : 'down' } : m
        ),
      }))
    } catch {}
  }, [allMessages, chatKey, selectedArch])

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
                const evalScore = await evaluateAnswer(q, r.answer, [], r.arch_key)
                return { ...r, eval: evalScore }
              } catch { return r }
            })
          )
        }
        setAllCompareResults(prev => ({ ...prev, [key]: final }))
        setHistory(prev => [...prev, { query: q, arch: 'Compare All', elapsed: Math.max(...results.map(r => r.elapsed)), answer: '' }])
      } finally { setCompareLoading(false) }
      return
    }

    setIsStreaming(true)
    setSteps([])
    setTokens('')
    setLiveSource([])

    let collectedSources: Source[] = []

    cleanupRef.current = streamQuery(q, selectedArch, SESSION_ID, {
      onStep: s => setSteps(prev => [...prev, s]),
      onToken: t => setTokens(prev => prev + t),
      onSources: s => { collectedSources = s; setLiveSource(s) },
      onDone: async (answer, elapsed, cached) => {
        let evalScore: EvalScore | undefined
        if (enableEval && answer) {
          try { evalScore = await evaluateAnswer(q, answer, collectedSources, selectedArch) } catch {}
        }
        appendMsg(selectedArch, {
          id: uuidv4(),
          role: 'assistant',
          content: answer || tokens,
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
  }, [input, isStreaming, compareMode, selectedArch, enableEval, tokens, appendMsg])

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
    setDocLibrary(prev => prev.filter(d => {
      const label = d.name.split('/').pop()?.split('\\').pop() || d.name
      return label !== source && d.name !== source
    }))
  }, [])

  const handleExport = () => {
    const md = currentMessages
      .map(m => `**${m.role === 'user' ? 'You' : m.arch || 'Assistant'}:**\n${m.content}`)
      .join('\n\n---\n\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }))
    a.download = `rag-chat-${compareMode ? 'compare' : selectedArch.slice(0, 20)}.md`
    a.click()
  }

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
        onExport={handleExport}
        onAnalytics={handleOpenAnalytics}
        onSettings={() => setShowApiKey(true)}
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
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-medium hover:bg-indigo-500/20 transition-colors"
                >
                  <Network size={13} />
                  Knowledge Graph
                </button>
              )}
            </div>
          </div>
        )}
        {compareMode && (
          <div className="px-6 py-3 border-b border-white/[0.06] bg-indigo-500/5 flex items-center gap-2">
            <span className="text-xs font-medium text-indigo-300">🔍 Compare Mode — all {archs.length} architectures run simultaneously</span>
          </div>
        )}

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {isEmpty && (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
              <p className="text-4xl">⚡</p>
              <p className="text-lg font-semibold text-slate-300">RAG Studio</p>
              <p className="text-sm text-slate-500 max-w-sm">
                {compareMode
                  ? 'Upload a document, then ask a question to compare all architectures side by side.'
                  : 'Upload a PDF, TXT, CSV, DOCX, image, or paste a URL — then start asking questions.'}
              </p>
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
        <div className="px-6 py-4 border-t border-white/[0.06] bg-[#0d0d18]">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
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
                  : 'Ask a question… (Enter to send, Shift+Enter for newline)'
              }
              rows={1}
              className="flex-1 resize-none bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder:text-slate-600 outline-none focus:border-indigo-500/40 transition-colors leading-relaxed"
              style={{ maxHeight: '120px', overflowY: 'auto' }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming || compareLoading}
              className="flex-shrink-0 w-10 h-10 rounded-xl bg-indigo-500 hover:bg-accent-h disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {isStreaming || compareLoading
                ? <Loader2 size={16} className="animate-spin text-white" />
                : <Send size={16} className="text-white" />}
            </button>
          </div>
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
                <Network size={15} className="text-indigo-400" />
                <span className="text-sm font-semibold text-slate-200">Knowledge Graph</span>
                <span className="text-xs text-slate-500">Graph RAG entity relationships</span>
              </div>
              <button onClick={() => setShowGraph(false)} className="text-slate-500 hover:text-slate-300 transition-colors">
                <X size={16} />
              </button>
            </div>
            {graphHtml
              ? <iframe srcDoc={graphHtml} className="flex-1 w-full border-0" sandbox="allow-scripts allow-same-origin" title="Knowledge Graph" />
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
