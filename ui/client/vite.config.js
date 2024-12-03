import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      $lib: path.resolve('./src/lib'),
    },
  },
  server: {
    proxy: {
      // '/api': 'http://localhost:7071',
      '/api': 'http://localhost:8000',
      // '/api': 'https://chem-reasoner-query.azurewebsites.net',
    },
  },
})
