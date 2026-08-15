/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./apps/web/index.html",
    "./apps/web/src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        studio: {
          50: '#f4f6fb',
          100: '#e4e8f5',
          800: '#111827',
          900: '#0b0f19',
          950: '#05070d',
        },
        cyan: {
          400: '#38bdf8',
          500: '#06b6d4',
        },
      },
    },
  },
  plugins: [],
}
