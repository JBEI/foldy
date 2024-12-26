// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // If CRA served your app from a sub-path, you might need base: '/sub-path/'
  // base: '/my-app/',
})