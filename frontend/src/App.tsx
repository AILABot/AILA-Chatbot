/**
 * @packageDocumentation
 *
 * Top-level React application component.
 *
 * ## Responsibilities
 * - Define the application routes using React Router v6.
 * - Protect sensitive routes (like `/chat`) behind authentication.
 * - Render shared layout (the `<Template />` banner) across all pages.
 *
 * ## Routing Table
 * - `/login`     → Login page
 * - `/register`  → Registration page (with verification flow)
 * - `/chat`      → Main Chat UI (requires authentication)
 * - `/`          → Redirects to `/chat`
 * - `*`          → Fallback to {@link Login} (could be replaced with a 404)
 *
 * ## Components
 * - {@link Template}: A static header/banner with project logos.
 * - PrivateRoute: Higher-order wrapper that guards protected pages.
 */


import { Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Chat from './pages/Chat';
import Register from './pages/Register';
import { useAuth } from './context/AuthContext';
import type { JSX } from 'react';
import { Template } from './components/Template';

/**
 * Protects routes that require authentication.
 *
 * @remarks
 * Uses {@link useAuth} to check if a user is logged in and verified.
 *
 * Logic:
 * 1. Waits for `loading` to finish (verifying session).
 * 2. Grants access if `user` exists and `user.verified === true`.
 * 3. Otherwise redirects to `/login`.
 *
 * @param children - The component tree to render when authenticated.
 * @returns The wrapped element if access is granted, else a redirect.
 *
 * @example
 * ```tsx
 * <Route
 * path='/chat'
 * element={<PrivateRoute><Chat /></PrivateRoute>}
 * />
 * ```
 */
function PrivateRoute({ children }: { children: JSX.Element }) {
  const { user, loading } = useAuth();

  if (loading) {
    // While verifying cookie/session with backend
    return <div>Loading...</div>;
  }

  // Debug output (can be removed in production)
  // console.log(user);

  return user && user.verified === true
    ? children
    : <Navigate to='/login' replace />;
}

/**
 * Root React component.
 *
 * @remarks
 * Defines the application’s routes using React Router v6 and renders
 * the global header banner via {@link Template}.
 *
 * **Routing table:**
 * - `/login` → {@link Login} page
 * - `/register` → {@link Register} page
 * - `/chat` → {@link Chat} page (protected)
 * - `/` → Redirects to `/chat`
 * - `*` → Fallback to {@link Login} (could be replaced with 404)
 *
 * @returns The top-level application component.
 *
 * @example
 * ```tsx
 * import { createRoot } from 'react-dom/client';
 * import App from './App';
 *
 * createRoot(document.getElementById('root')!).render(<App />);
 * ```
 */
export function App() {
  return (
    <div>
      {/* Global banner visible on all routes */}
      <Template />

      {/* Define routes */}
      <Routes>
        {/* Public routes */}
        <Route path='/login' element={<Login />} />
        <Route path='/register' element={<Register />} />

        {/* Protected route */}
        <Route path='/chat' element={<PrivateRoute><Chat /></PrivateRoute>} />

        {/* Redirects & fallbacks */}
        <Route path='/' element={<Navigate to='/chat' />} />
        <Route path="*" element={<Login />} />
        {/* TODO: Replace with a proper 404 Not Found page */}
      </Routes>
    </div>
  );
}

export default App;
