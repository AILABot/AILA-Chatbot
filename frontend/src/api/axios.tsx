/**
 * @packageDocumentation
 *
 * Axios API Client
 * =================
 * A single, reusable Axios instance for talking to the backend — so every request
 * acknowledges the **Head of the Table** with the correct `baseURL` and cookies.
 *
 * Why centralize Axios config?
 * - Single source of truth for `baseURL`
 * - Consistent credential handling (`withCredentials: true` for JWT cookies)
 * - One place to add interceptors, logging, and retry policies later
 *
 * Environment & Deployment
 * ------------------------
 * - Local dev: defaults to `http://localhost:8080`
 * - Reverse proxy (Vite / Nginx): use relative `/` base
 * - Hosted prod: point at your deployed API (e.g., `https://api.example.com`)
 *
 * **CORS & Cookies**
 * - Because `withCredentials: true`, the backend must set:
 *   - `Access-Control-Allow-Credentials: true`
 *   - A matching `Access-Control-Allow-Origin` (not `*`) to the frontend origin
 *   - HttpOnly `token` cookie on login
 *
 * Testing & Overrides
 * -------------------
 * - Per-request base override: `api.get('/ping', { baseURL: 'http://127.0.0.1:8080' })`
 * - Create a fresh client (e.g., for tests): `const testApi = createApi()`
 * - Mocking: use `msw` or `axios-mock-adapter`
 *
 * @remarks
 * This module exports a preconfigured {@link AxiosInstance} (`api`) and a factory
 * {@link createApi} for bespoke clients (e.g., different base URLs in tests).
 *
 * @example Quick start
 * ```ts
 * import api from './api'
 * const res = await api.get('/user_conversations')
 * console.log(res.data)
 * ```
 *
 * @example POST with JSON body
 * ```ts
 * const payload = { username: 'roman', password: 'acknowledge-me' }
 * const res = await api.post('/login', payload)
 * ```
 *
 * @example File upload (multipart/form-data)
 * ```ts
 * const form = new FormData()
 * form.append('files', fileInput.files![0])
 * form.append('message', 'Describe the files attached by the user')
 * form.append('conversation_type', 'lawsuit')
 * const res = await api.post('/request', form, {
 *   headers: { 'Content-Type': 'multipart/form-data' },
 * })
 * ```
 *
 * @example Strongly-typed response (TypeScript generics)
 * ```ts
 * type Conversation = { id: string; conversation_name: string }
 * const { data } = await api.get<Conversation[]>('/user_conversations', { params: { username: 'roman' } })
 * ```
 *
 * @example Creating a custom client (e.g., in tests)
 * ```ts
 * import { createApi } from './api'
 * const testApi = createApi()
 * testApi.defaults.baseURL = 'http://localhost:8081' // point to a stub server
 * ```
 *
 * @see {@link createApi | Factory helper} to spawn isolated clients
 */

import axios, { type AxiosInstance } from 'axios'

/**
 * Create a pre-configured Axios client for backend API calls.
 *
 * @remarks
 * Default configuration:
 *  - `baseURL`: `http://localhost:8080` (local development)
 *    - Alternative presets (commented below):
 *      - `'/'` (use dev proxy or reverse proxy in prod)
 *      - `'https://your-prod-api.example.com'`
 *  - `withCredentials`: `true` — sends/receives cookies (e.g., HttpOnly JWT)
 *
 * @returns A preconfigured {@link AxiosInstance}.
 *
 * @example Switch target at runtime
 * ```ts
 * const client = createApi()
 * client.defaults.baseURL = import.meta.env.VITE_API_URL // from your .env
 * ```
 *
 * @example Per-request override without new client
 * ```ts
 * await api.get('/health', { baseURL: 'https://override-host.example.com' })
 * ```
 *
 * @example Add interceptors (logging, auth errors) — do this where you bootstrap the app
 * ```ts
 * api.interceptors.response.use(
 *   res => res,
 *   err => {
 *     if (err.response?.status === 401) {
 *       // e.g., redirect to login
 *     }
 *     return Promise.reject(err)
 *   }
 * )
 * ```
 */
function createApi(): AxiosInstance {
  return axios.create({
    // baseURL: 'https://ailademo.fly.dev', // ✅ Production deployment
    baseURL: 'http://localhost:8080', // ✅ Local development
    // baseURL: '/', // ✅ Default: relative path (frontend proxy)
    // baseURL: 'http://ailabot.upatras.gr'
    withCredentials: true, // Include cookies for auth
  })
}

/**
 * Exported Axios client for general use across the app.
 *
 * @remarks
 * Import this in your React hooks and components to call backend endpoints.
 * If you need a separate client (e.g., different base URL, custom headers),
 * use {@link createApi}.
 *
 * @example
 * ```ts
 * import { api } from './api'
 * const res = await api.get('/messages', { params: { conversation_id: 'abc-123' } })
 * ```
 */
export const api: AxiosInstance = createApi()

export default api
export { createApi }
export type { AxiosInstance }
