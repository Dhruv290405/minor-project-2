import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API calls to the Node gateway so the browser can use a same-origin
    // relative path (/api/...) in dev without CORS or a hardcoded host.
    proxy: {
      '/api': 'http://localhost:5000',
    },
  },
})
