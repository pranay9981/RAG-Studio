'use client'
import React from 'react'

// Render inline tokens: **bold**, *italic*, `code`
function renderInline(text: string, prefix: string): React.ReactNode {
  const tokens: React.ReactNode[] = []
  const re = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g
  let last = 0, m: RegExpExecArray | null, k = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index))
    const raw = m[0]
    const id = `${prefix}-${k++}`
    if (raw.startsWith('**'))
      tokens.push(<strong key={id} className="font-semibold text-slate-100">{raw.slice(2, -2)}</strong>)
    else if (raw.startsWith('`'))
      tokens.push(<code key={id} className="bg-white/[0.07] rounded px-1 py-0.5 text-[11px] font-mono text-indigo-300">{raw.slice(1, -1)}</code>)
    else
      tokens.push(<em key={id} className="italic text-slate-300">{raw.slice(1, -1)}</em>)
    last = m.index + raw.length
  }
  if (last < text.length) tokens.push(text.slice(last))
  if (tokens.length === 0) return null
  if (tokens.length === 1) return tokens[0]
  return <React.Fragment>{tokens}</React.Fragment>
}

export default function MarkdownContent({ content, className }: { content: string; className?: string }) {
  const lines = content.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0, k = 0
  const nextKey = () => String(k++)

  while (i < lines.length) {
    const line = lines[i].trimEnd()

    // blank
    if (!line.trim()) { i++; continue }

    // heading
    const hm = line.match(/^(#{1,3}) (.+)/)
    if (hm) {
      const level = hm[1].length
      const cls =
        level === 1 ? 'text-base font-bold text-white mt-3 mb-1' :
        level === 2 ? 'text-sm font-bold text-slate-100 mt-2 mb-1' :
                      'text-sm font-semibold text-slate-200 mt-2 mb-0.5'
      const Tag = `h${level}` as 'h1' | 'h2' | 'h3'
      nodes.push(<Tag key={nextKey()} className={cls}>{renderInline(hm[2], nextKey())}</Tag>)
      i++; continue
    }

    // horizontal rule
    if (/^-{3,}$/.test(line.trim())) {
      nodes.push(<hr key={nextKey()} className="border-white/[0.06] my-2" />)
      i++; continue
    }

    // unordered list — collect consecutive bullet lines
    if (/^[*-] /.test(line)) {
      const items: React.ReactNode[] = []
      while (i < lines.length && /^[*-] /.test(lines[i])) {
        items.push(<li key={nextKey()}>{renderInline(lines[i].replace(/^[*-] /, ''), nextKey())}</li>)
        i++
      }
      nodes.push(<ul key={nextKey()} className="list-disc list-inside space-y-0.5 mb-2 text-slate-300">{items}</ul>)
      continue
    }

    // ordered list
    if (/^\d+\. /.test(line)) {
      const items: React.ReactNode[] = []
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        items.push(<li key={nextKey()}>{renderInline(lines[i].replace(/^\d+\. /, ''), nextKey())}</li>)
        i++
      }
      nodes.push(<ol key={nextKey()} className="list-decimal list-inside space-y-0.5 mb-2 text-slate-300">{items}</ol>)
      continue
    }

    // blockquote
    if (/^> /.test(line)) {
      const items: React.ReactNode[] = []
      while (i < lines.length && /^> /.test(lines[i])) {
        items.push(<p key={nextKey()} className="mb-0.5">{renderInline(lines[i].slice(2), nextKey())}</p>)
        i++
      }
      nodes.push(
        <blockquote key={nextKey()} className="border-l-2 border-indigo-500/40 pl-3 text-slate-400 italic mb-2">
          {items}
        </blockquote>
      )
      continue
    }

    // fenced code block
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      if (i < lines.length) {
        i++ // skip closing ``` only if it exists (prevents drop on unterminated block)
      }
      nodes.push(
        <pre key={nextKey()} className="bg-white/[0.05] rounded-lg p-3 overflow-x-auto my-2">
          <code className={`text-xs font-mono text-indigo-200${lang ? ` language-${lang}` : ''}`}>
            {codeLines.join('\n')}
          </code>
        </pre>
      )
      continue
    }

    // paragraph — collect until blank or block-level line
    const para: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,3} |[*-] |\d+\. |> |-{3,}$|```)/.test(lines[i])
    ) {
      para.push(lines[i].trimEnd())
      i++
    }
    if (para.length) {
      nodes.push(
        <p key={nextKey()} className="mb-1.5 last:mb-0 text-slate-200 leading-relaxed">
          {renderInline(para.join('\n'), nextKey())}
        </p>
      )
    }
  }

  return <div className={className ?? ''}>{nodes}</div>
}
