/**
* @packageDocumentation
*
* @remarks
* Renders the login page. Authenticates via {@link useAuth}.
*
* Features:
* - Username & password input fields
* - Calls the AuthContext `loginUser` to authenticate
* - Handles both success and error flows:
*   - If user is verified → navigate to /chat
*   - If user is NOT verified → resend code & redirect to /register
*   - If authentication fails → display error message
* - Resets username/password fields on error
* - Uses TailwindCSS for styling 
* 
* Behavior:
* - Success + verified → redirect `/chat`
* - Success + unverified → resend code & redirect `/register`
* - Failure → show error
*
* Uses TailwindCSS for styling.
* 
* Dependencies:
* - useAuth (AuthContext)
* - react-router-dom (navigation)
*
* @example
* ```tsx
* import Login from './pages/Login';
* <Route path="/login" element={<Login />} />
* ```
*/

import { useState, useRef, useEffect } from "react";
import { useAuth } from "../context/AuthContext.jsx";
import { Link, useNavigate } from "react-router-dom";



/**
 * Login component — renders the sign-in page.
 *
 * @remarks
 * Provides username/password login with three outcome flows:
 * 1. **Success & verified** → navigates to `/chat`.
 * 2. **Success but unverified** → resends verification code, then redirects to `/register`.
 * 3. **Failure** → shows an error message and resets input fields.
 *
 * ## Responsibilities
 * - Collect username & password from the user.
 * - Call {@link AuthContextType.loginUser | loginUser} via {@link useAuth}.
 * - Call {@link AuthContextType.resendCode | resendCode} if the user is unverified.
 * - Manage local state for inputs, loading, and errors.
 * - Redirect the user appropriately based on login outcome.
 *
 * ## Props
 * None. This component consumes global state/actions from {@link useAuth}.
 *
 * ## Returns
 * A styled login form wrapped in TailwindCSS classes.
 *
 * @example
 * ```tsx
 * import { Login } from "./pages/Login";
 *
 * <Route path="/login" element={<Login />} />
 * ```
 */
export const Login = () => {
    // ------------------ State Management ------------------
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const { loginUser, resendCode } = useAuth();
    const [errorMsg, setErrorMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    // ------------------ Refs ------------------
    const userRef = useRef<HTMLInputElement>(null);
    const errRef = useRef<HTMLParagraphElement>(null);

    const navigate = useNavigate();

    // ------------------ Effects ------------------
    useEffect(() => {
        userRef.current?.focus();
    }, []);

    // ------------------ Handlers ------------------

    /**
     * Handles login form submission.
     *
     * param e React.FormEvent
     * - Prevents default form submit
     * - Calls login API
     * - Handles three cases:
     *   1. Auth success & verified → navigate to chat
     *   2. Auth success & unverified → resend code + go to register
     *   3. Auth failed → show error message
     */
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setErrorMessage("");

        try {
            const res = await loginUser(username, password);
            // Case 1: Success & verified
            if (res && "user_details" in res) {
                const { verified, email } = res.user_details;
                if (verified) {
                    navigate("/chat");
                } else {
                    // Case 2: Not verified → resend code
                    await resendCode(username, email);
                    navigate("/register");
                }

                // Case 3: Error returned by API
            } else if (res && "error_message" in res) {
                setErrorMessage(res.error_message);
                errRef.current?.focus();
                setUsername("");
                setPassword("");
            }
            // Unknown failure
            else {
                // Login failed — wrong credentials, user not found, etc.
                setErrorMessage("Login failed. Something happened");
                errRef.current?.focus();
                setUsername("");
                setPassword("");
            }
        } catch (err) {
            console.error("Login error:", err);
            setErrorMessage(String(err));
            errRef.current?.focus();
            setUsername("");
            setPassword("");
        } finally {
            setIsLoading(false);
        }
    };

    // ------------------ Render ------------------
    return (
        <div className="min-h-screen bg-gray-100 text-gray-800 px-4 py-8">
            {/* Login Form */}
            <div className="w-full max-w-md bg-white shadow-lg rounded-xl p-10 mx-auto mt-16">
                <h1 className="text-3xl font-semibold text-center mb-6">Sign In</h1>

                {errorMsg && (
                    <p
                        ref={errRef}
                        className="text-red-700 bg-red-100 border border-red-300 p-3 rounded text-sm mb-4"
                        aria-live="assertive"
                    >
                        {errorMsg}
                    </p>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                        <label htmlFor="username" className="block text-sm font-medium mb-1">
                            Username
                        </label>
                        <input
                            type="text"
                            name="username"
                            id="username"
                            autoComplete="off"
                            required
                            ref={userRef}
                            value={username}
                            onChange={(e) => {
                                setUsername(e.target.value);
                                if (errorMsg) setErrorMessage("");
                            }}
                            className="w-full px-3 py-2 bg-white text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Enter your username"
                        />
                    </div>

                    <div>
                        <label htmlFor="password" className="block text-sm font-medium mb-1">
                            Password
                        </label>
                        <input
                            type="password"
                            name="password"
                            id="password"
                            autoComplete="off"
                            required
                            value={password}
                            onChange={(e) => {
                                setPassword(e.target.value);
                                if (errorMsg) setErrorMessage("");
                            }}
                            className="w-full px-3 py-2 bg-white text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Enter your password"
                        />
                    </div>

                    <div className="pt-2">
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded transition-transform transform hover:scale-105 cursor-pointer text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isLoading ? "Signing In..." : "Submit"}
                        </button>
                    </div>
                    <div className="pt-4 text-sm text-center text-gray-600">
                        <span>If you do not have an account yet...</span>{' '}
                        <Link
                            to="/register"
                            className="font-semibold text-blue-600 hover:underline hover:text-blue-800 transition-colors"
                        >
                            Sign Up here
                        </Link>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Login;
