'use client'
import { useState } from 'react'
import { Key, X, ExternalLink, Loader2, Check, AlertCircle } from 'lucide-react'
import { setApiKey } from '@/lib/api'

interface Props {
  onClose: () => void
  onKeySet: () => void
}

export default function ApiKeyModal({ onClose, onKeySet }: Props) {
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleSave = async () => {
    if (!key.trim()) return
    setLoading(true)
    setError('')
    try {
      await setApiKey(key.trim())
      setSuccess(true)
      setTimeout(() => {
        onKeySet()
        onClose()
      }, 700)
    } catch (e: any) {
      setError(e.message || 'Failed to set API key')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div
        className="bg-[#0e0e1a] border border-white/[0.1] rounded-2xl shadow-2xl w-full max-w-md mx-4 p-6"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/20 flex items-center justify-center flex-shrink-0">
              <Key size={15} className="text-indigo-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Configure Groq API Key</h2>
              <p className="text-[11px] text-slate-500 mt-0.5">Required to query any RAG architecture</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-600 hover:text-slate-300 transition-colors mt-0.5"
          >
            <X size={15} />
          </button>
        </div>

        <div className="space-y-4">
          {/* Key input */}
          <div>
            <label className="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1.5 block">
              Groq API Key
            </label>
            <div className="relative">
              <input
                type={show ? 'text' : 'password'}
                value={key}
                onChange={e => setKey(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSave()}
                placeholder="gsk_..."
                className="w-full text-sm bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2.5 pr-16 text-slate-200 placeholder:text-slate-600 outline-none focus:border-indigo-500/40 transition-colors font-mono"
              />
              <button
                onClick={() => setShow(o => !o)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-500 hover:text-slate-300 transition-colors font-mono"
              >
                {show ? 'hide' : 'show'}
              </button>
            </div>
          </div>

          {/* Get key link */}
          <a
            href="https://console.groq.com/keys"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition-colors w-fit"
          >
            <ExternalLink size={11} />
            Get a free key from Groq Console
          </a>

          {/* Info note */}
          <div className="flex items-start gap-2 bg-white/[0.03] border border-white/[0.06] rounded-xl px-3 py-2.5">
            <AlertCircle size={12} className="text-slate-500 mt-0.5 flex-shrink-0" />
            <p className="text-[11px] text-slate-500 leading-relaxed">
              The key is stored in the server process memory only — it is never written to disk or sent
              anywhere other than Groq's API. Restart the server and re-enter to change it.
            </p>
          </div>

          {/* Error */}
          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">
              {error}
            </p>
          )}

          {/* Actions */}
          <div className="flex gap-2.5 pt-1">
            <button
              onClick={onClose}
              className="flex-1 py-2.5 rounded-xl border border-white/[0.08] text-xs text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] transition-colors"
            >
              Skip for now
            </button>
            <button
              onClick={handleSave}
              disabled={!key.trim() || loading || success}
              className="flex-1 py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 text-xs text-white font-medium transition-colors flex items-center justify-center gap-2"
            >
              {loading && <Loader2 size={12} className="animate-spin" />}
              {success && <Check size={12} />}
              {success ? 'Saved!' : loading ? 'Saving…' : 'Save Key'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
