import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // base:"/aila-demo-app" //for the demo app in github pages, delete this line for local development
  server: {
    watch: {
      usePolling: true // Use polling for file changes, useful in some environments like Docker
    },
    host: true,
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:8000', // Adjust to your backend URL
    //     changeOrigin: true,
    //     secure: false,
    //   },
    //   '/ws': {
    //     target: 'ws://localhost:8000', // WebSocket endpoint
    //     ws: true,
    //   }
    // }
  }
})
