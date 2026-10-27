import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Set API_URL when the backend runs with another --host or --port.
const api = process.env.API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [vue()],
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 950 },
  server: { proxy: { '/api': api } },
})
