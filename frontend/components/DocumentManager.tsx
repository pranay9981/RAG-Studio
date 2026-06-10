'use client'
import { useState, useRef } from 'react'
import { Upload, Link, Loader2 } from 'lucide-react'
import { ingestFile, ingestUrl } from '@/lib/api'

interface Props {
  archKeys: string[]
  onIngested: (source: string, chunks: number) => void
}

export default function DocumentManager({ archKeys, onIngested }: Props) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  // Always ingest into all 8 architectures so switching modes never requires re-upload
  const targets = archKeys

  const handleFile = async (file: File) => {
    setLoading(true); setMsg('')
    try {
      const res = await ingestFile(file, targets)
      setMsg(`✓ ${res.chunks} chunks ingested`)
      onIngested(res.source, res.chunks)
    } catch (e: any) { setMsg(`✗ ${e.message}`) }
    finally { setLoading(false) }
  }

  const handleUrl = async () => {
    if (!url.trim()) return
    setLoading(true); setMsg('')
    try {
      const res = await ingestUrl(url.trim(), targets)
      setMsg(`✓ ${res.chunks} chunks ingested`)
      onIngested(res.source, res.chunks)
      setUrl('')
    } catch (e: any) { setMsg(`✗ ${e.message}`) }
    finally { setLoading(false) }
  }

  return (
    <div className="px-4 py-3 space-y-2.5">
      <p className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">Upload Document</p>

      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
        className="border border-dashed border-white/[0.1] rounded-lg p-3 text-center cursor-pointer hover:border-indigo-500/40 hover:bg-indigo-500/5 transition-colors"
      >
        <Upload size={14} className="mx-auto mb-1 text-slate-500" />
        <p className="text-[11px] text-slate-500">Drop file or click</p>
        <p className="text-[10px] text-slate-600 mt-0.5">PDF · TXT · DOCX · PNG · JPG</p>
      </div>
      <input ref={fileRef} type="file" accept=".pdf,.txt,.docx,.png,.jpg,.jpeg" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />

      <div className="flex gap-1.5">
        <input
          value={url} onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleUrl()}
          placeholder="https://..." className="flex-1 text-xs bg-white/[0.04] border border-white/[0.08] rounded-lg px-2.5 py-2 text-slate-300 placeholder:text-slate-600 outline-none focus:border-indigo-500/40"
        />
        <button onClick={handleUrl} disabled={!url.trim() || loading} className="px-2.5 py-2 rounded-lg bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-500/30 disabled:opacity-40 transition-colors">
          <Link size={12} />
        </button>
      </div>

      {loading && <div className="flex items-center gap-1.5 text-xs text-slate-400"><Loader2 size={11} className="animate-spin" />Processing…</div>}
      {msg && !loading && <p className={`text-xs ${msg.startsWith('✓') ? 'text-green-400' : 'text-red-400'}`}>{msg}</p>}
    </div>
  )
}
