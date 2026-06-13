'use client'
import { useState } from 'react'
import { ChevronDown, Sparkles, HelpCircle, Network } from 'lucide-react'
import type { ArchInfo } from '@/lib/types'

interface Props {
  arch:             ArchInfo
  onExplainerOpen?: () => void
  showGraphBtn?:    boolean
  onGraphOpen?:     () => void
}

export default function ArchCard({ arch, onExplainerOpen, showGraphBtn, onGraphOpen }: Props) {
  const [open, setOpen] = useState(false)

  const num       = arch.key.split(' ')[0]
  const fullName  = arch.key.split(' ').slice(1).join(' ')
  const parenIdx  = fullName.indexOf('(')
  const mainName  = parenIdx === -1 ? fullName : fullName.slice(0, parenIdx).trim()
  const techStr   = parenIdx === -1 ? '' : fullName.slice(parenIdx + 1, fullName.lastIndexOf(')')).trim()
  const techTags  = techStr ? techStr.split('+').map(t => t.trim()).filter(Boolean) : []

  return (
    <div className="flex-shrink-0 relative" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>

      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 pointer-events-none"
        style={{
          height: 1,
          background: 'linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.6) 30%, rgba(139,92,246,0.3) 70%, transparent 100%)',
        }} />

      {/* Background gradient — violet wash on left */}
      <div className="absolute inset-0 pointer-events-none"
        style={{
          background: 'linear-gradient(90deg, rgba(124,58,237,0.07) 0%, rgba(124,58,237,0.02) 40%, transparent 100%)',
        }} />

      {/* Main row */}
      <div className="relative flex items-center gap-0" style={{ background: 'rgba(13,13,26,0.6)' }}>

        {/* Icon + name — clickable expand area */}
        <button
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-4 px-5 py-4 text-left flex-1 min-w-0 group"
        >
          {/* Icon with layered glow */}
          <div className="relative flex-shrink-0">
            <div className="absolute inset-0 rounded-2xl blur-md"
              style={{ background: 'rgba(124,58,237,0.25)', transform: 'scale(1.15)' }} />
            <div className="relative flex items-center justify-center text-2xl rounded-2xl"
              style={{
                width: 48, height: 48,
                background: 'linear-gradient(135deg, rgba(30,16,64,0.9) 0%, rgba(15,10,36,0.95) 100%)',
                border: '1px solid rgba(124,58,237,0.35)',
                boxShadow: '0 0 20px rgba(124,58,237,0.18), inset 0 1px 0 rgba(255,255,255,0.06)',
              }}>
              {arch.icon}
            </div>
          </div>

          {/* Text block */}
          <div className="flex-1 min-w-0">
            {/* Number + name row */}
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] font-mono font-bold tabular-nums flex-shrink-0"
                style={{ color: 'rgba(124,58,237,0.45)' }}>
                {num}
              </span>
              <h2 className="text-[15px] font-bold leading-tight tracking-tight truncate"
                style={{ color: '#f1f5f9' }}>
                {mainName}
              </h2>
            </div>

            {/* Tagline */}
            <p className="text-[11.5px] leading-snug mt-0.5 truncate"
              style={{ color: '#475569' }}>
              {arch.tagline}
            </p>

            {/* Tech tags */}
            {techTags.length > 0 && (
              <div className="flex items-center gap-1 mt-1.5 flex-wrap">
                {techTags.map(tag => (
                  <span key={tag}
                    className="text-[9px] font-mono leading-none px-1.5 py-0.5 rounded-md"
                    style={{
                      color: 'rgba(167,139,250,0.55)',
                      background: 'rgba(124,58,237,0.07)',
                      border: '1px solid rgba(124,58,237,0.13)',
                    }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Expand chevron */}
          <ChevronDown
            size={14}
            className="flex-shrink-0 transition-transform duration-200"
            style={{
              color: open ? '#a78bfa' : '#334155',
              transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
            }}
          />
        </button>

        {/* Vertical divider */}
        <div className="flex-shrink-0 self-stretch"
          style={{ width: 1, background: 'rgba(255,255,255,0.05)', margin: '10px 0' }} />

        {/* Right controls */}
        <div className="flex items-center gap-2 px-5 flex-shrink-0">
          {/* Active chip */}
          <span className="flex items-center gap-1.5 text-[10px] font-semibold rounded-full px-2.5 py-1"
            style={{
              color: '#34d399',
              background: 'rgba(52,211,153,0.08)',
              border: '1px solid rgba(52,211,153,0.2)',
            }}>
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{
                background: '#34d399',
                boxShadow: '0 0 7px rgba(52,211,153,0.8)',
              }} />
            Active
          </span>

          {/* How it works */}
          <button
            onClick={onExplainerOpen}
            className="flex items-center gap-1.5 text-[11px] font-medium rounded-lg px-3 py-1.5 transition-all"
            style={{
              color: '#64748b',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
            onMouseEnter={e => {
              ;(e.currentTarget as HTMLElement).style.color = '#cbd5e1'
              ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)'
            }}
            onMouseLeave={e => {
              ;(e.currentTarget as HTMLElement).style.color = '#64748b'
              ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'
            }}
          >
            <HelpCircle size={11} />
            How it works
          </button>

          {/* Knowledge Graph (conditional) */}
          {showGraphBtn && (
            <button
              onClick={onGraphOpen}
              className="flex items-center gap-1.5 text-[11px] font-medium rounded-lg px-3 py-1.5 transition-all"
              style={{
                color: '#c4b5fd',
                background: 'rgba(124,58,237,0.1)',
                border: '1px solid rgba(124,58,237,0.25)',
              }}
              onMouseEnter={e => {
                ;(e.currentTarget as HTMLElement).style.background = 'rgba(124,58,237,0.18)'
              }}
              onMouseLeave={e => {
                ;(e.currentTarget as HTMLElement).style.background = 'rgba(124,58,237,0.1)'
              }}
            >
              <Network size={11} />
              Knowledge Graph
            </button>
          )}
        </div>
      </div>

      {/* Expandable details */}
      {open && (
        <div
          className="grid grid-cols-2 gap-6 px-5 pt-4 pb-5 animate-fade-in"
          style={{
            background: 'linear-gradient(180deg, rgba(124,58,237,0.04) 0%, transparent 100%)',
            borderTop: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <div className="space-y-2">
            <p className="text-[9px] font-bold uppercase tracking-widest"
              style={{ color: '#334155', letterSpacing: '0.12em' }}>
              How it works
            </p>
            <p className="text-[11.5px] leading-relaxed" style={{ color: '#94a3b8' }}>
              {arch.how}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-[9px] font-bold uppercase tracking-widest flex items-center gap-1.5"
              style={{ color: 'rgba(124,58,237,0.6)', letterSpacing: '0.12em' }}>
              <Sparkles size={8} /> Best for
            </p>
            <p className="text-[11.5px] leading-relaxed" style={{ color: 'rgba(196,181,253,0.8)' }}>
              {arch.best_for}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
