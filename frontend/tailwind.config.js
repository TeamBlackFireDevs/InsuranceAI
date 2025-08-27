/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',   // 👈 enable class-based dark mode
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    extend: {},
  },
  plugins: [require('@tailwindcss/typography')],
}

