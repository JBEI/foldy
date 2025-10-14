// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer';
import { Plugin } from 'vite';
import reactSwc from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [reactSwc(),
    visualizer({
      open: true,
      gzipSize: true,
    }) as Plugin,
  ],
  build: {
    target: 'es2015',
    minify: 'esbuild',
    sourcemap: false, // Set to true only when needed
    chunkSizeWarningLimit: 1000, // Increase warning limit
    reportCompressedSize: false, // Disable compressed size reporting
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
  optimizeDeps: {
    include: ['react', 'react-dom', 'antd', 'd3', 'plotly.js'],
    exclude: ['pdbe-molstar']
  },

  // If CRA served your app from a sub-path, you might need base: '/sub-path/'
  // base: '/my-app/',
})
