import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Tailwind v4 is integrated through its official Vite plugin.
// https://tailwindcss.com/docs/installation/using-vite
export default defineConfig({
  plugins: [react(), tailwindcss()],
})
