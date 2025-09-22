/**
 * index.tsx
 * ============================================================
 * Frontend entrypoint for the AILA Chatbot Demo.
 *
 * Responsibilities
 * ----------------
 * - Bootstraps React (concurrent root via React 18).
 * - Wraps the app with:
 *   - <BrowserRouter> for client-side routing
 *   - <AuthProvider> to expose authentication & app state globally
 * - Applies global styles (Tailwind / index.css).
 *
 * Notes
 * -----
 * - Ensure BrowserRouter is imported from 'react-router-dom' (NOT 'react-router').
 *   If you see navigation issues, switch:
 *     import { BrowserRouter } from 'react-router-dom';
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { AuthProvider } from './context/AuthContext';
// ✅ Prefer this:
// import { BrowserRouter } from 'react-router-dom';
// ⛔ Current import may cause issues in some setups:
import { BrowserRouter } from 'react-router';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

root.render(
    <React.StrictMode>
        {/* Router context for route resolution & navigation */}
        <BrowserRouter>
            {/* Global auth/conversation provider for the entire app tree */}
            <AuthProvider>
                <App />
            </AuthProvider>
        </BrowserRouter>
    </React.StrictMode>
);
