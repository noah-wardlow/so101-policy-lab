import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
// The mujoco-react Vite plugin generates the typed model register (control-name
// unions etc.) from the MJCF at build time.
import { mujocoReact } from 'mujoco-react/vite';

export default defineConfig({
  plugins: [
    mujocoReact({
      models: {
        so101: 'public/models/so101/SO101.xml',
      },
    }),
    tailwindcss(),
    react(),
  ],
  // onnxruntime-web ships prebuilt wasm; let Vite serve it and avoid pre-bundling.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  // Cross-origin isolation enables SharedArrayBuffer -> multithreaded WASM and
  // the fast onnxruntime-web path (browser ACT inference) + MuJoCo threads.
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    chunkSizeWarningLimit: 6000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/@react-three/')) return 'r3f';
          if (id.includes('node_modules/three/')) return 'three';
          if (id.includes('node_modules/onnxruntime-web')) return 'ort';
        },
      },
    },
  },
  // Single copies of React/three so R3F + mujoco-react share one instance.
  resolve: {
    dedupe: ['react', 'react-dom', 'three', '@react-three/fiber', '@react-three/drei'],
  },
});
