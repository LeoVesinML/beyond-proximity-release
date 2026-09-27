import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    // @sparkjsdev/spark is loaded from CDN (npm package missing WASM binary)
    // so no need to pre-bundle it
    exclude: [],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/scenes': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
