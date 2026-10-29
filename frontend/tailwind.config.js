/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Clinical Blue - Primary palette
        'clinical-blue': {
          900: '#0A2540',
          700: '#1E4976',
          500: '#2E6BA8',
          100: '#E3F2FD',
        },
        // Diagnostic Teal - AI/Analysis palette
        'diagnostic-teal': {
          700: '#006B7D',
          500: '#00ACC1',
          100: '#E0F7FA',
        },
        // Risk indicators (semantic)
        'risk': {
          low: '#2E7D32',
          medium: '#F57C00',
          high: '#C62828',
        },
        // Neutrals - Cool grays with blue tint
        'slate': {
          950: '#0F172A',
          900: '#1E293B',
          800: '#1E293B',
          700: '#334155',
          600: '#475569',
          500: '#64748B',
          400: '#94A3B8',
          300: '#CBD5E1',
          200: '#E2E8F0',
          100: '#F1F5F9',
          50: '#F8FAFC',
        },
        // Surface colors
        'surface': {
          base: '#FAFBFC',
          elevated: '#FFFFFF',
        },
        // Legacy compatibility (will be phased out)
        'primary': '#1E4976',
        'secondary': '#00ACC1',
        'background': '#FAFBFC',
        'error': '#C62828',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Inter', 'sans-serif'],
        mono: ['"SF Mono"', 'Consolas', 'Monaco', 'monospace'],
      },
      fontSize: {
        'xs': ['0.6875rem', { lineHeight: '1.5', letterSpacing: '0.01em' }],
        'sm': ['0.8125rem', { lineHeight: '1.5', letterSpacing: '0.005em' }],
        'base': ['0.9375rem', { lineHeight: '1.6', letterSpacing: '0' }],
        'lg': ['1.125rem', { lineHeight: '1.4', letterSpacing: '-0.01em' }],
        'xl': ['1.5rem', { lineHeight: '1.3', letterSpacing: '-0.015em' }],
        '2xl': ['2rem', { lineHeight: '1.2', letterSpacing: '-0.02em' }],
      },
      spacing: {
        '1': '0.5rem',   // 8px
        '2': '1rem',     // 16px
        '3': '1.5rem',   // 24px
        '4': '2rem',     // 32px
        '6': '3rem',     // 48px
        '8': '4rem',     // 64px
      },
      borderRadius: {
        'sm': '6px',
        'md': '10px',
        'lg': '16px',
      },
      boxShadow: {
        'sm': '0 1px 2px rgba(15, 23, 42, 0.04)',
        'md': '0 2px 8px rgba(15, 23, 42, 0.06)',
        'lg': '0 8px 24px rgba(15, 23, 42, 0.08)',
      },
      transitionDuration: {
        'quick': '150ms',
        'default': '250ms',
        'gentle': '350ms',
      },
      transitionTimingFunction: {
        'out': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'in-out': 'cubic-bezier(0.45, 0, 0.55, 1)',
      },
      backdropBlur: {
        'xl': '20px',
      },
    },
  },
  plugins: [
    // Custom plugin for tabular numerals
    function({ addUtilities }) {
      addUtilities({
        '.tabular-nums': {
          'font-variant-numeric': 'tabular-nums',
        },
      })
    },
  ],
}
