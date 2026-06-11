import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base:      '#07070f',
        surface:   '#0d0d1a',
        surface2:  '#13132a',
        surface3:  '#1a1a35',
        accent:    '#7c3aed',
        'accent-h':'#6d28d9',
        'accent-s':'#8b5cf6',
        border:    'rgba(255,255,255,0.07)',
      },
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      boxShadow: {
        'glow-sm': '0 0 12px rgba(124,58,237,0.25)',
        'glow':    '0 0 24px rgba(124,58,237,0.3)',
        'glow-lg': '0 0 40px rgba(124,58,237,0.35)',
        'card':    '0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05)',
      },
      animation: {
        'pulse-slow':  'pulse 2.5s cubic-bezier(0.4,0,0.6,1) infinite',
        'fade-in':     'fadeIn 0.25s ease-out',
        'slide-up':    'slideUp 0.3s ease-out',
        'glow-pulse':  'glowPulse 2s ease-in-out infinite',
        'dot-bounce':  'dotBounce 1.4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:    { from: { opacity:'0', transform:'translateY(6px)' }, to: { opacity:'1', transform:'translateY(0)' } },
        slideUp:   { from: { opacity:'0', transform:'translateY(12px)' }, to: { opacity:'1', transform:'translateY(0)' } },
        glowPulse: { '0%,100%': { opacity:'0.6' }, '50%': { opacity:'1' } },
        dotBounce: {
          '0%,80%,100%': { transform:'scale(0)', opacity:'0.3' },
          '40%': { transform:'scale(1)', opacity:'1' },
        },
      },
    },
  },
  plugins: [],
}
export default config
