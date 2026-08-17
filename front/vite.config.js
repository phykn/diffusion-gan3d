import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 950 },
  server: { proxy: { '/generate': 'http://127.0.0.1:8000', '/health': 'http://127.0.0.1:8000' } },
})
