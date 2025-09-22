/**
 * @packageDocumentation
 *
 * @remarks
 * Application bootstrap for **AILA**. Creates the React 18 root and wires the
 * top‑level providers:
 * 
 * What happens here:
 * - Create React 18 root
 * - Provide TanStack Query client for data fetching & caching
 * - Provide AuthProvider for global auth/conversation state
 * - Wire up React Router (BrowserRouter) for client-side navigation
 *
 *  * IMPORTANT:
 * - Use BrowserRouter from 'react-router-dom' (NOT 'react-router').
 * - Provider order:
 *     QueryClientProvider (network/cache)
 *       └── AuthProvider (auth state)
 *             └── BrowserRouter (routing context)
 *                   └── <App />
 * 
 * 1. {@link QueryClientProvider} — network & cache layer (TanStack Query)
 * 2. {@link AuthProvider} — authentication and conversation state
 * 3. {@link BrowserRouter} — client‑side routing context
 * 4. {@link App} — application shell and routes
 *
 * **Provider order** matters: Query/cache → Auth → Router → App.
 *
 * @example
 * ```tsx
 * const queryClient = new QueryClient();
 *
 * createRoot(document.getElementById('root')!).render(
 *  <React.StrictMode>
 *    <QueryClientProvider client={queryClient}>
 *      <AuthProvider>
 *        <Router>
 *          <App />
 *        </Router>
 *      </AuthProvider>
 *    </QueryClientProvider>
 * </React.StrictMode>
 * );
 * ```
 */

import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { BrowserRouter as Router } from "react-router-dom"; // ✅ correct package
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./context/AuthContext.jsx";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <App />
        </Router>
      </AuthProvider>
    </QueryClientProvider>
  </React.StrictMode>
);
