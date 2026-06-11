import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'RAG Studio — 10 Architectures',
  description: 'Compare 10 state-of-the-art RAG architectures on your documents',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-base text-slate-200 antialiased">{children}</body>
    </html>
  )
}
