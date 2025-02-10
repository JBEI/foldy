// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer';
import { Plugin } from 'vite';

export default defineConfig({
  plugins: [react(),
    visualizer({
      open: true,
      gzipSize: true,
    }) as Plugin,
  ],
  build: {
    target: 'es2015',
    minify: 'esbuild',
    sourcemap: false, // Set to true only when needed
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js/dist/plotly'],
          pdbeMolstar: ['pdbe-molstar/build/pdbe-molstar-plugin'],   
          vendor: [
            'react',
            'react-dom',
            'antd',
            'd3',
          ],
          // Add more chunks as needed
        },
      },
    },
  },
  // If CRA served your app from a sub-path, you might need base: '/sub-path/'
  // base: '/my-app/',
})