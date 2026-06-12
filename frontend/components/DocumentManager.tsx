'use client'
import { useState, useRef } from 'react'
import { Upload, Link, Loader2, Trash2, FileText, X } from 'lucide-react'
import { ingestFile, ingestUrl, deleteDocument } from '@/lib/api'
import type { DocItem } from '@/lib/types'

interface Props {
  archKeys: string[]
  onIngested: (source: string, chunks: number) => void
  docLibrary: DocItem[]
  onDocDeleted: (source: string) => void
}

export default function DocumentManager({ archKeys, onIngested, docLibrary, onDocDeleted }: Props) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState('')
  const [msgType, setMsgType] = useState<'ok'|'err'>('ok')
  const [deletingSource, setDeletingSource] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFiles = async (files: File[]) => {
    if (!files.length) return
    setLoading(true); setMsg('')
    let totalChunks = 0; let errors = 0
    const errorMessages: string[] = []
    for (const file of files) {
      try {
        const res = await ingestFile(file, archKeys)
        totalChunks += res.chunks
        onIngested(res.source, res.chunks)
      } catch (e: unknown) {
        errors++
        errorMessages.push(`${file.name}: ${e instanceof Error ? e.message : 'unknown error'}`)
      }
    }
    if (errors > 0) {
      setMsgType('err')
      setMsg(`${files.length - errors} ingested · ${errors} failed — ${errorMessages.slice(0,2).join('; ')}`)
    } else {
      setMsgType('ok')
      setMsg(`${files.length} file${files.length > 1 ? 's' : ''} → ${totalChunks} chunks`)
    }
    setLoading(false)
  }

  const handleUrl = async () => {
    if (!url.trim()) return
    setLoading(true); setMsg('')
    try {
      const res = await ingestUrl(url.trim(), archKeys)
      setMsgType('ok')
      setMsg(`${res.chunks} chunks ingested`)
      onIngested(res.source, res.chunks)
      setUrl('')
    } catch (e: unknown) {
      setMsgType('err')
      setMsg(e instanceof Error ? e.message : 'Failed')
    }
    setLoading(false)
  }

  const handleDelete = async (source: string) => {
    setDeletingSource(source)
    try {
      await deleteDocument(source)
      onDocDeleted(source)
    } catch (e: unknown) {
      setMsgType('err')
      setMsg(`Delete failed: ${e instanceof Error ? e.message : 'unknown'}`)
    }
    setDeletingSource(null)
  }

  const uniqueDocs = [...new Map(docLibrary.map(d => [d.name, d])).values()]

  return (
    <div className="px-3 py-3 space-y-3">
      <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest px-1">Documents</p>

      {/* Drop zone */}
      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(Array.from(e.dataTransfer.files)) }}
        className={`relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-violet-500/60 bg-violet-500/10'
            : 'border-white/[0.08] hover:border-violet-500/40 hover:bg-violet-500/5'
        }`}
      >
        <div className="w-8 h-8 rounded-xl bg-violet-500/15 border border-violet-500/20 flex items-center justify-center mx-auto mb-2">
          <Upload size={13} className="text-violet-400" />
        </div>
        <p className="text-xs font-medium text-slate-400">Drop files or click</p>
        <p className="text-[10px] text-slate-600 mt-0.5">PDF · TXT · DOCX · CSV · XLSX · PNG · JPG</p>
      </div>
      <input
        ref={fileRef} type="file" multiple
        accept=".pdf,.txt,.docx,.csv,.xlsx,.xls,.png,.jpg,.jpeg"
        className="hidden"
        onChange={e => handleFiles(Array.from(e.target.files ?? []))}
      />

      {/* URL input */}
      <div className="flex gap-1.5">
        <input
          value={url} onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleUrl()}
          placeholder="https://…"
          className="flex-1 text-xs bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-slate-300 placeholder:text-slate-700 outline-none focus:border-violet-500/40 focus:bg-violet-500/5 transition-all"
        />
        <button
          onClick={handleUrl}
          disabled={!url.trim() || loading}
          className="px-2.5 py-2 rounded-xl bg-violet-500/15 border border-violet-500/25 text-violet-400 hover:bg-violet-500/25 disabled:opacity-40 transition-all"
        >
          <Link size={12} />
        </button>
      </div>

      {/* Status */}
      {loading && (
        <div className="flex items-center gap-2 text-xs text-slate-500 px-1">
          <Loader2 size={11} className="animate-spin text-violet-400" />Processing…
        </div>
      )}
      {msg && !loading && (
        <div className={`flex items-start gap-2 text-xs px-3 py-2 rounded-xl border ${
          msgType === 'ok'
            ? 'text-emerald-400 bg-emerald-500/8 border-emerald-500/15'
            : 'text-red-400 bg-red-500/8 border-red-500/15'
        }`}>
          <span className="flex-1">{msg}</span>
          <button onClick={() => setMsg('')} className="flex-shrink-0 opacity-60 hover:opacity-100">
            <X size={10} />
          </button>
        </div>
      )}

      {/* Ingested list */}
      {uniqueDocs.length > 0 && (
        <div className="space-y-1 pt-1">
          <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest px-1">Ingested</p>
          {uniqueDocs.map((d) => {
            const label = d.name.split('/').pop()?.split('\\').pop() || d.name
            const isDeleting = deletingSource === d.name
            return (
              <div key={d.name} className="flex items-center gap-2 group px-2 py-1.5 rounded-xl hover:bg-white/[0.03] transition-colors">
                <div className="w-5 h-5 rounded-lg bg-violet-500/10 border border-violet-500/15 flex items-center justify-center flex-shrink-0">
                  <FileText size={9} className="text-violet-400" />
                </div>
                <span className="text-[11px] text-slate-400 truncate flex-1" title={d.name}>{label}</span>
                <span className="text-[10px] font-mono text-slate-700 flex-shrink-0">{d.chunks}c</span>
                <button
                  onClick={() => handleDelete(d.name)}
                  disabled={isDeleting}
                  className="opacity-0 group-hover:opacity-100 flex-shrink-0 w-5 h-5 rounded-lg flex items-center justify-center text-slate-700 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-40 transition-all"
                >
                  {isDeleting ? <Loader2 size={9} className="animate-spin" /> : <Trash2 size={9} />}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
