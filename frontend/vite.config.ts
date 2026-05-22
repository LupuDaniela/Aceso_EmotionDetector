import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: { '@': path.resolve(__dirname, './src') },
    },
    server: {
        port: 5173,
        proxy: {
            '/auth/google': 'http://localhost:8000',
            '/auth/login':        'http://localhost:8000',
            '/auth/register':     'http://localhost:8000',
            '/auth/forgot-password': 'http://localhost:8000',
            '/auth/reset-password':  'http://localhost:8000',
            '/auth/me':           'http://localhost:8000',
            '/api':               'http://localhost:8000',
        },
    },
})