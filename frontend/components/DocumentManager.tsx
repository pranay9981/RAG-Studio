'use client'
import { useState, useRef } from 'react'
import { Upload, Link, Loader2, Trash2, FileText } from 'lucide-react'
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
  const [deletingSource, setDeletingSource] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFiles = async (files: File[]) => {
    if (!files.length) return
    setLoading(true); setMsg('')
    let totalChunks = 0; let errors = 0
    for (const file of files) {
      try {
        const res = await ingestFile(file, archKeys)
        totalChunks += res.chunks
        onIngested(res.source, res.chunks)
      } catch { errors++ }
    }
    setMsg(errors === 0
      ? `✓ ${files.length} file${files.length > 1 ? 's' : ''} → ${totalChunks} chunks ingested`
      : `✓ ${files.length - errors} ingested, ${errors} failed`)
    setLoading(false)
  }

  const handleUrl = async () => {
    if (!url.trim()) return
    setLoading(true); setMsg('')
    try {
      const res = await ingestUrl(url.trim(), archKeys)
      setMsg(`✓ ${res.chunks} chunks ingested`)
      onIngested(res.source, res.chunks)
      setUrl('')
    } catch (e: any) { setMsg(`✗ ${e.message}`) }
    finally { setLoading(false) }
  }

  const handleDelete = async (source: string) => {
    setDeletingSource(source)
    try {
      await deleteDocument(source)
      onDocDeleted(source)
    } catch (e: any) {
      setMsg(`✗ Delete failed: ${e.message}`)
    } finally {
      setDeletingSource(null)
    }
  }

  // Deduplicate by name
  const uniqueDocs = [...new Map(docLibrary.map(d => [d.name, d])).values()]

  return (
    <div className="px-4 py-3 space-y-2.5">
      <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">Upload Document</p>

      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFiles(Array.from(e.dataTransfer.files)) }}
        className="border border-dashed border-white/[0.1] rounded-lg p-3 text-center cursor-pointer hover:border-indigo-500/40 hover:bg-indigo-500/5 transition-colors"
      >
        <Upload size={14} className="mx-auto mb-1 text-slate-500" />
        <p className="text-[11px] text-slate-500">Drop files or click · multiple OK</p>
        <p className="text-[10px] text-slate-600 mt-0.5">PDF · TXT · DOCX · CSV · XLSX · PNG · JPG</p>
      </div>
      <input
        ref={fileRef} type="file" multiple
        accept=".pdf,.txt,.docx,.csv,.xlsx,.xls,.png,.jpg,.jpeg"
        className="hidden"
        onChange={e => handleFiles(Array.from(e.target.files ?? []))}
      />

      <div className="flex gap-1.5">
        <input
          value={url} onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleUrl()}
          placeholder="https://…"
          className="flex-1 text-xs bg-white/[0.04] border border-white/[0.08] rounded-lg px-2.5 py-2 text-slate-300 placeholder:text-slate-600 outline-none focus:border-indigo-500/40"
        />
        <button onClick={handleUrl} disabled={!url.trim() || loading} className="px-2.5 py-2 rounded-lg bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-500/30 disabled:opacity-40 transition-colors">
          <Link size={12} />
        </button>
      </div>

      {loading && <div className="flex items-center gap-1.5 text-xs text-slate-400"><Loader2 size={11} className="animate-spin" />Processing…</div>}
      {msg && !loading && <p className={`text-xs ${msg.startsWith('✓') ? 'text-green-400' : 'text-red-400'}`}>{msg}</p>}

      {/* Ingested files list */}
      {uniqueDocs.length > 0 && (
        <div className="pt-1">
          <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1.5">Ingested Files</p>
          <div className="space-y-1">
            {uniqueDocs.map((d, i) => {
              const label = d.name.split('/').pop()?.split('\\').pop() || d.name
              const isDeleting = deletingSource === label
              return (
                <div key={i} className="flex items-center gap-2 group">
                  <FileText size={10} className="text-slate-600 flex-shrink-0" />
                  <span className="text-[11px] text-slate-400 truncate flex-1" title={d.name}>{label}</span>
                  <span className="text-[10px] text-slate-600 flex-shrink-0">{d.chunks}c</span>
                  <button
                    onClick={() => handleDelete(label)}
                    disabled={isDeleting}
                    className="opacity-0 group-hover:opacity-100 flex-shrink-0 p-0.5 text-slate-600 hover:text-red-400 disabled:opacity-40 transition-all"
                    title="Delete from all architectures"
                  >
                    {isDeleting
                      ? <Loader2 size={10} className="animate-spin" />
                      : <Trash2 size={10} />
                    }
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
