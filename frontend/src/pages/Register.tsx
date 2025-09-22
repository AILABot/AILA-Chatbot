/**
* @packageDocumentation
*
* @remarks
* React component that handles **user registration** and **email verification**.
*
* Features:
* - New user registration with:
*   - username
*   - email
*   - password (double-entry check)
* - Server-side password policy enforced
* - Email verification workflow:
*   - After successful registration, user must verify via 6-digit code sent to email
*   - Code expires in 120s (timer with resend functionality)
*   - Resend code button reactivates only after expiration
* - Graceful error handling and accessibility with `aria-live`
* - TailwindCSS styled form with clear instructions
*
* States:
* - `verified`:
*    undefined → user not yet registered
*    false → registered but awaiting email verification
*    true → registration + verification completed → redirected to `/chat`
*
* Dependencies:
* - useAuth (AuthContext → RegisterUser, verifyCodeUser, resendCode)
* - react-router-dom (navigation)
*
* @example
* ```tsx
* import Register from './pages/Register';
* <Route path="/register" element={<Register />} />
* ```
*/

import { useState, useRef, useEffect } from "react";
import { useAuth } from "../context/AuthContext.jsx";
import { Link, useNavigate } from "react-router-dom";



/**
 * Register component — renders the registration & verification flow.
 *
 * @remarks
 * Provides a two-step flow:
 * 1. **Registration** — collects username, email, and password, validates inputs,
 *    and calls {@link AuthContextType.registerUser | registerUser}.
 * 2. **Verification** — prompts for a one-time code sent by email, calls
 *    {@link AuthContextType.verifyCodeUser | verifyCodeUser}, and allows resending via
 *    {@link AuthContextType.resendCode | resendCode}.
 *
 * ## Responsibilities
 * - Collect user credentials (username, email, password + confirm).
 * - Validate password match and enforce backend policy.
 * - Submit registration request to backend via `registerUser`.
 * - Transition to verification phase if registration succeeds.
 * - Display countdown timer and manage resend code functionality.
 * - On successful verification → navigate to `/chat`.
 *
 * ## Props
 * None. Uses global state/actions via {@link useAuth}.
 *
 * ## Returns
 * A React element that conditionally renders either:
 * - Registration form (step 1), or
 * - Email verification form (step 2).
 *
 * ## Example
 * ```tsx
 * import { Register } from "./pages/Register";
 *
 * <Route path="/register" element={<Register />} />
 * ```
 */
export const Register = () => {
    // ------------------ State ------------------
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [password_1, setNewPassword] = useState("");
    const [role, setRole] = useState("");
    const [email, setEmail] = useState("");
    const { RegisterUser, verifyCodeUser, resendCode, user } = useAuth();
    const [errorMsg, setErrorMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [verified, setVerified] = useState<boolean | undefined>(undefined);
    const [verificationCode, setVerificationCode] = useState("");

    // Countdown timer for resend functionality
    const [timeLeft, setTimeLeft] = useState(120);
    const [timerKey, setTimerKey] = useState(0);

    // ------------------ Refs ------------------
    const userRef = useRef<HTMLInputElement>(null);
    const errRef = useRef<HTMLParagraphElement>(null);

    const navigate = useNavigate();

    // ------------------ Effects ------------------
    useEffect(() => {
        userRef.current?.focus();
    }, []);

    // Prefill username/email if user context exists
    useEffect(() => {
        if (user?.verified === false && verified === undefined) {
            setUsername(user.username);
            setEmail(user.email);
        }
    }, [])

    // Reset timer when entering verification phase
    useEffect(() => {
        if (verified === false) {
            setTimeLeft(120);
            setTimerKey(prev => prev + 1); // restart interval logic
        }
    }, [verified]);

    // Timer countdown logic
    useEffect(() => {
        const interval = setInterval(() => {
            setTimeLeft(prev => {
                if (prev <= 1) {
                    clearInterval(interval);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [timerKey]);

    useEffect(() => {
    }, [username, email, password, password_1]);

    // ------------------ Handlers ------------------

    /**
     * Handle verification code submission.
     * - Calls backend to validate code
     * - On success → mark user verified and navigate to `/chat`
     * - On failure → reset code field and display error
     */
    const handleVerificationCode = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            const res = await verifyCodeUser(username, verificationCode);
            if (typeof res === 'boolean' && res === true) {
                setVerified(true);
                navigate('/chat');
            } else if (typeof res === 'object' && 'error_message' in res) {
                setVerificationCode("");
                setErrorMessage(res.error_message);
            }
            else {
                setVerificationCode("");
                setErrorMessage("User Verification failed");
            }
        } catch (err) {
            setVerificationCode("");
            setErrorMessage("User Verification failed");
            errRef.current?.focus();
        } finally {
            setIsLoading(false);
        }
    }

    /**
     * Resend verification code.
     * - Only available once timer has expired
     */
    const rescendCode = async () => {
        setIsLoading(true);
        try {
            await resendCode(username, email);
            setTimeLeft(120);
            setTimerKey(prev => prev + 1);
        } catch (err) {
            setErrorMessage("Something went wrong");
            errRef.current?.focus();
        } finally {
            setIsLoading(false);
        }
    }

    /**
     * Handle registration form submission.
     * - Validates password match
     * - Calls backend RegisterUser
     * - Sets verification state if registration successful
     * - Displays error if failure
     */
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        if (password === password_1) {
            try {
                const res = await RegisterUser(username, password_1, email, role);
                if (res === true) {
                    setVerified(false);
                    setPassword("");
                    setNewPassword("");
                    setRole("");
                }
                else if (res && "error_message" in res) {
                    setUsername("");
                    setPassword("");
                    setNewPassword("");
                    setEmail("");
                    setRole("");
                    setErrorMessage(res.error_message);
                    errRef.current?.focus();
                }
                else {
                    setUsername("");
                    setPassword("");
                    setNewPassword("");
                    setEmail("");
                    setRole("");
                    setErrorMessage("Registratiion failed. Something happened");
                    errRef.current?.focus();
                }

            } catch (err) {
                setUsername("");
                setPassword("");
                setNewPassword("");
                setRole("");
                setEmail("")
                setErrorMessage(`Registratiion failed. Something happened ${err}`);
                errRef.current?.focus();
            } finally {
                setIsLoading(false);
            }
        } else {
            setUsername("");
            setPassword("");
            setNewPassword("");
            setEmail("")
            setRole("");
            setErrorMessage("Registration failed. Passwords do not match");
            errRef.current?.focus();
            setIsLoading(false);
        }
    };

    // ------------------ Conditional Rendering ------------------

    // Step 2: Email verification phase
    if (verified === false || user?.verified === false) {
        return (
            <div className="min-h-screen bg-gray-100 text-gray-800 px-4 py-8">
                <div className="w-full max-w-md bg-white shadow-lg rounded-xl p-10 mx-auto mt-16">
                    <h1 className="text-3xl font-semibold text-center mb-6">Verify Your Email</h1>

                    {errorMsg && (
                        <p
                            ref={errRef}
                            className="text-red-700 bg-red-100 border border-red-300 p-3 rounded text-sm mb-4"
                            aria-live="assertive"
                        >
                            {errorMsg}
                        </p>
                    )}

                    <form onSubmit={handleVerificationCode} className="space-y-5">
                        <p>A verification code has been sent to your email. It will expire in {timeLeft} seconds.</p>
                        <div>
                            <label htmlFor="verification_code" className="block text-sm font-medium mb-1">
                                Verification code
                            </label>
                            <input
                                type="text"
                                name="verification_code"
                                id="verification_code"
                                autoComplete="off"
                                required
                                ref={userRef}
                                value={verificationCode}
                                onChange={(e) => {
                                    setVerificationCode(e.target.value);
                                    if (errorMsg) setErrorMessage("");
                                }}
                                className="w-full px-3 py-2 bg-white text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Enter your verification code"
                            />
                        </div>

                        <div className="pt-2 flex flex-col gap-3">
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded transition-transform transform hover:scale-105 cursor-pointer text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? "Verifying..." : "Verify"}
                            </button>

                            <button
                                type="button"
                                disabled={isLoading || timeLeft > 0}
                                onClick={rescendCode}
                                className="w-full py-3 px-6 bg-gray-600 hover:bg-gray-700 text-white font-semibold rounded transition-transform transform hover:scale-105 cursor-pointer text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {timeLeft > 0 ? `Resend available in ${timeLeft}s` : "Resend Verification Code"}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        );
    }

    // Step 1: Registration form
    if (verified === undefined) {
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
                            <label htmlFor="email" className="block text-sm font-medium mb-1">
                                Email
                            </label>
                            <input
                                type="email"
                                name="email"
                                id="email"
                                autoComplete="off"
                                required
                                value={email}
                                onChange={(e) => {
                                    setEmail(e.target.value);
                                    if (errorMsg) setErrorMessage("");
                                }}
                                className="w-full px-3 py-2 bg-white text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Enter your email"
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

                        <div>
                            <label htmlFor="new_password" className="block text-sm font-medium mb-1">
                                Password verification
                            </label>
                            <input
                                type="password"
                                name="password_"
                                id="password_1"
                                autoComplete="off"
                                required
                                value={password_1}
                                onChange={(e) => {
                                    setNewPassword(e.target.value);
                                    if (errorMsg) setErrorMessage("");
                                }}
                                className="w-full px-3 py-2 bg-white text-gray-900 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Verify your password"
                            />
                        </div>

                        <div>
                            <ul style={{ fontSize: '0.9rem', marginTop: '8px' }}>
                                <li>At least 10 characters</li>
                                <li>At least 1 uppercase letter</li>
                                <li>At least 1 lowercase letter</li>
                                <li>At least 1 number</li>
                                <li>At least 1 special character (e.g. !@#$%)</li>
                            </ul>

                        </div>

                        <fieldset className="space-y-2">
                            <legend className="font-medium mb-2">Select your role</legend>

                            <div className="flex items-center">
                                <input
                                    type="radio"
                                    id="user"
                                    name="role"
                                    value="user"
                                    required
                                    checked={role === "user"}
                                    onChange={(e) => setRole(e.target.value)}
                                    className="peer"
                                />
                                <label htmlFor="user" className="ml-2 cursor-pointer peer-checked:font-semibold">
                                    Normal User
                                </label>
                            </div>

                            <div className="flex items-center">
                                <input
                                    type="radio"
                                    id="lawyer"
                                    name="role"
                                    value="lawyer"
                                    required
                                    checked={role === "lawyer"}
                                    onChange={(e) => setRole(e.target.value)}
                                    className="peer"
                                />
                                <label htmlFor="lawyer" className="ml-2 cursor-pointer peer-checked:font-semibold">
                                    Lawyer
                                </label>
                            </div>
                        </fieldset>

                        <div className="pt-2">
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded transition-transform transform hover:scale-105 cursor-pointer text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? "Signing Up..." : "Submit"}
                            </button>
                        </div>
                        <div className="pt-4 text-sm text-center text-gray-600">
                            <span>If you do have an account just</span>{' '}
                            <Link
                                to="/login"
                                className="font-semibold text-blue-600 hover:underline hover:text-blue-800 transition-colors"
                            >
                                Sign In here
                            </Link>
                        </div>
                    </form>
                </div>
            </div>
        )
    }
};

export default Register;
