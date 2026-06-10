'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import { Send, Loader2, Network, X } from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import ArchCard from '@/components/ArchCard'
import ChatMessage from '@/components/ChatMessage'
import BrainWorking from '@/components/BrainWorking'
import SourcePanel from '@/components/SourcePanel'
import CompareGrid from '@/components/CompareGrid'
import DocumentManager from '@/components/DocumentManager'
import { getArchitectures, streamQuery, compareAll, evaluateAnswer, resetSession, getGraphHtml } from '@/lib/api'
import type { ArchInfo, ChatMessage as Msg, Source, EvalScore, DocItem, HistoryItem, CompareResult } from '@/lib/types'
import { v4 as uuidv4 } from 'uuid'

const SESSION_ID = 'default'
const GRAPH_ARCH = '02 Graph RAG (Knowledge Graphs)'

export default function Page() {
  const [archs, setArchs] = useState<ArchInfo[]>([])
  const [selectedArch, setSelectedArch] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const [enableEval, setEnableEval] = useState(false)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [steps, setSteps] = useState<string[]>([])
  const [tokens, setTokens] = useState('')
  const [liveSource, setLiveSource] = useState<Source[]>([])
  const [ingestedArchs, setIngestedArchs] = useState<Set<string>>(new Set())
  const [docLibrary, setDocLibrary] = useState<DocItem[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [compareResults, setCompareResults] = useState<CompareResult[]>([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [graphHtml, setGraphHtml] = useState('')
  const [showGraph, setShowGraph] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    getArchitectures().then(list => {
      setArchs(list)
      if (list.length) setSelectedArch(list[0].key)
    })
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, tokens, steps, compareResults])

  const currentArch = archs.find(a => a.key === selectedArch)

  const handleOpenGraph = useCallback(async () => {
    const html = await getGraphHtml()
    setGraphHtml(html)
    setShowGraph(true)
  }, [])

  const handleSend = useCallback(async () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')

    const userMsg: Msg = { id: uuidv4(), role: 'user', content: q }
    setMessages(prev => [...prev, userMsg])

    if (compareMode) {
      setCompareLoading(true)
      setCompareResults([])
      try {
        const results = await compareAll(q, SESSION_ID)

        if (enableEval) {
          const withEval = await Promise.all(
            results.map(async r => {
              if (r.error || !r.answer) return r
              try {
                const evalScore = await evaluateAnswer(q, r.answer, [])
                return { ...r, eval: evalScore }
              } catch { return r }
            })
          )
          setCompareResults(withEval)
        } else {
          setCompareResults(results)
        }

        setHistory(prev => [...prev, {
          query: q,
          arch: 'Compare All',
          elapsed: Math.max(...results.map(r => r.elapsed)),
          answer: '',
        }])
      } finally { setCompareLoading(false) }
      return
    }

    setIsStreaming(true)
    setSteps([])
    setTokens('')
    setLiveSource([])

    let finalAnswer = ''
    let elapsed = 0
    let collectedSources: Source[] = []

    cleanupRef.current = streamQuery(q, selectedArch, SESSION_ID, {
      onStep: s => setSteps(prev => [...prev, s]),
      onToken: t => setTokens(prev => prev + t),
      onSources: s => { collectedSources = s; setLiveSource(s) },
      onDone: async (answer, el) => {
        finalAnswer = answer
        elapsed = el
        let evalScore: EvalScore | undefined

        if (enableEval && answer) {
          try { evalScore = await evaluateAnswer(q, answer, collectedSources) } catch {}
        }

        const assistantMsg: Msg = {
          id: uuidv4(),
          role: 'assistant',
          content: answer || tokens,
          arch: selectedArch,
          elapsed,
          sources: collectedSources,
          eval: evalScore,
        }
        setMessages(prev => [...prev, assistantMsg])
        setHistory(prev => [...prev, { query: q, arch: selectedArch, elapsed, answer: answer.slice(0, 120) }])
        setIsStreaming(false)
        setSteps([])
        setTokens('')
        setLiveSource([])
      },
      onError: err => {
        setMessages(prev => [...prev, { id: uuidv4(), role: 'assistant', content: `⚠️ ${err}`, arch: selectedArch }])
        setIsStreaming(false)
        setSteps([])
        setTokens('')
      },
    })
  }, [input, isStreaming, compareMode, selectedArch, enableEval, tokens])

  const handleReset = async () => {
    await resetSession(SESSION_ID)
    setMessages([])
    setIngestedArchs(new Set())
    setDocLibrary([])
    setHistory([])
    setCompareResults([])
    setGraphHtml('')
  }

  const handleExport = () => {
    const md = messages.map(m => `**${m.role === 'user' ? 'You' : 'Assistant'}:**\n${m.content}`).join('\n\n---\n\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }))
    a.download = 'rag-chat.md'
    a.click()
  }

  const showGraphBtn = selectedArch === GRAPH_ARCH && ingestedArchs.has(GRAPH_ARCH) && !compareMode

  return (
    <div className="flex h-screen bg-base overflow-hidden">
      <Sidebar
        architectures={archs}
        selectedArch={selectedArch}
        compareMode={compareMode}
        enableEval={enableEval}
        ingestedArchs={ingestedArchs}
        docLibrary={docLibrary}
        history={history}
        onSelectArch={setSelectedArch}
        onCompareToggle={() => setCompareMode(o => !o)}
        onEvalToggle={() => setEnableEval(o => !o)}
        onClearChat={() => { setMessages([]); setCompareResults([]) }}
        onReset={handleReset}
        onExport={handleExport}
      >
        <DocumentManager
          archKeys={archs.map(a => a.key)}
          onIngested={(source, chunks) => {
            setDocLibrary(prev => [...prev, { name: source, chunks }])
            setIngestedArchs(new Set(archs.map(a => a.key)))
          }}
        />
      </Sidebar>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Arch info card or compare header */}
        {currentArch && !compareMode && (
          <div className="flex items-center justify-between pr-4">
            <ArchCard arch={currentArch} />
            {showGraphBtn && (
              <button
                onClick={handleOpenGraph}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-medium hover:bg-indigo-500/20 transition-colors flex-shrink-0"
              >
                <Network size={13} />
                Knowledge Graph
              </button>
            )}
          </div>
        )}
        {compareMode && (
          <div className="px-6 py-3 border-b border-white/[0.06] bg-indigo-500/5 flex items-center gap-2">
            <span className="text-xs font-medium text-indigo-300">🔍 Compare Mode — all 8 architectures run simultaneously</span>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.length === 0 && !isStreaming && compareResults.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
              <p className="text-4xl">⚡</p>
              <p className="text-lg font-semibold text-slate-300">RAG Studio</p>
              <p className="text-sm text-slate-500 max-w-sm">Upload a document, then ask a question. Switch between 8 RAG architectures to compare how each one answers.</p>
            </div>
          )}

          {messages.map(msg => (
            <ChatMessage key={msg.id} message={msg} archIcon={archs.find(a => a.key === msg.arch)?.icon} />
          ))}

          {isStreaming && (
            <div className="animate-fade-in">
              <BrainWorking steps={steps} tokens={tokens} isStreaming={isStreaming} archIcon={currentArch?.icon || '🤖'} archLabel={currentArch?.label || selectedArch} />
              {liveSource.length > 0 && <div className="max-w-[88%] mt-2"><SourcePanel sources={liveSource} /></div>}
            </div>
          )}

          {compareResults.length > 0 && (
            <div className="animate-fade-in">
              <p className="text-xs text-slate-500 mb-3">Results for: <span className="text-slate-300">{messages[messages.length - 1]?.content}</span></p>
              <CompareGrid results={compareResults} architectures={archs} loading={false} />
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
              placeholder={ingestedArchs.size === 0 ? 'Upload a document first, then ask a question…' : 'Ask a question… (Enter to send, Shift+Enter for newline)'}
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
    </div>
  )
}
