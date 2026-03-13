/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        up: '#22c55e',
        down: '#ef4444',
        neutral: '#6b7280',
      },
    },
  },
  plugins: [],
}
