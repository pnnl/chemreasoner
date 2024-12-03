/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte}'],
  theme: {
    fontFamily: {
      title: ['"Exo 2"', 'ui-sans-serif', 'system-ui'],
      body: ['"Alegreya Sans"', 'ui-sans-serif', 'system-ui'],
      mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', '"Liberation Mono"', '"Courier New"', 'monospace'],
      icon: ['Material Icons'],
    },
    extend: {},
  },
  plugins: [],
}

