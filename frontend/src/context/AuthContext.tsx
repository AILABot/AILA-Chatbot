/**
 * @packageDocumentation
 *
 * Auth Context (Global) — Login • Sessions • Conversations • Messages • Feedback
 * ============================================================================
 * A single source of truth for authentication and chat state across your React app.
 * Centralizes user lifecycle (login/register/verify/logout), conversation/message
 * management, and feedback submission—so every component may acknowledge the
 * **Head of the Table** through `useAuth()`.
 *
 * Responsibilities
 * ----------------
 * 1) Authentication lifecycle
 *    - login, logout, register
 *    - verify session on app load (`verifyUser`)
 *    - resend verification codes
 *
 * 2) User state management
 *    - `user`: current profile or `null`
 *    - `conversations`: user's conversations
 *    - `userMessages`: messages of the active conversation
 *
 * 3) Conversation + Message features
 *    - Create/rename conversations
 *    - Create messages
 *    - Fetch conversations and messages
 *    - Submit thumbs feedback
 *    - Create document feedback (links bad answers to source docs)
 *
 * 4) Context Hook
 *    - `useAuth()` for state + actions
 *
 * Integration
 * -----------
 * Wrap your app in `<AuthProvider>` and consume the context anywhere via `useAuth()`.
 *
 * Security & Networking
 * ---------------------
 * - Uses Axios with `withCredentials: true` (JWT cookie).
 * - Backend must set `Access-Control-Allow-Credentials: true` and a specific
 *   `Access-Control-Allow-Origin` (no `*`) to allow cookie-based auth in browsers.
 *
 * Error Handling Contract
 * -----------------------
 * - Many actions resolve to `void | ErrorMessage` or `boolean | ErrorMessage`.
 * - On failures, some actions clear `user` to reflect invalid session.
 * - Callers should handle returned `ErrorMessage` and surface UX to the user.
 *
 * Examples
 * --------
 * ```tsx
 * <AuthProvider>
 *   <App />
 * </AuthProvider>
 *
 * const { user, loginUser, fetchConversations, createMessage } = useAuth()
 * await loginUser('roman', 'acknowledge-me')
 * await fetchConversations(user!.username)
 * await createMessage(convId, 'Hello, court.', 'user', crypto.randomUUID(), null)
 * ```
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import {
    loginAPI,
    getUserMessagesAPI,
    userFeedbackAPI,
    renameConversationAPI,
    resendCodeAPI,
    verifyAPI,
    logoutAPI,
    registerAPI,
    verifyUser,
    createConversationAPI,
    createMessageAPI,
    getConversationsAPI,
    createDocumentFeedbackAPI
} from '../services/AuthService.jsx';
import type {
    LoginAPIOutput,
    UserProfile,
    Message,
    Conversations,
    ErrorMessage,
} from '../models/Types.jsx';


/**
 * Shape of the AuthContext — defines all state + actions.
 *
 * @remarks
 * Consumers get strongly-typed helpers for auth + chat operations.
 * Many actions return either `void`/`boolean` or an {@link ErrorMessage}
 * so components can display precise error toasts/UI.
 */
export interface AuthContextType {
    /** Current logged-in user, or `null` if unauthenticated. */
    user: UserProfile | null;
    /** Messages in the active conversation. */
    userMessages: Message[] | null;
    /** List of all conversations for the user. */
    conversations: Conversations[] | null;
    /** Loading state while verifying session on mount. */
    loading: boolean;

    /** Create a new conversation (supports `conversation_type` e.g., "normal" | "lawsuit"). */
    createConversation: (conversation_name: string, username: string, conversation_type:string) => Promise<Conversations | undefined>;

    /** Create a new message in a conversation (client also appends to local state). */
    createMessage: (conversation_id: string, text: string, role: string, id: string, feedback: string | null) => Promise<void>;

    /** Fetch all messages for a conversation and hydrate `userMessages`. */
    fetchUserMessages: (conversation_id: string) => Promise<void>;

    /** Login a user (sets HttpOnly cookie server-side; hydrates `user`). */
    loginUser(username: string, password: string): Promise<LoginAPIOutput | ErrorMessage | null>;

    /** Logout the current user (clears cookie server-side and local state). */
    logoutUser: () => Promise<void>;

    /** Fetch all conversations for a user and hydrate `conversations`. */
    fetchConversations: (username: string) => Promise<void>;

    /** Register a new user (updates `user` on success with `verified: false`). */
    RegisterUser: (username: string, password: string, email: string, role: string) => Promise<boolean | ErrorMessage>;

    /** Verify a user with a code sent via email. */
    verifyCodeUser: (username: string, code: string) => Promise<boolean | ErrorMessage>;

    /** Resend a verification code to the user's email. */
    resendCode: (username: string, email: string) => Promise<void>;

    /** Submit thumbs feedback on a message (True/False/None as string). */
    userFeedback: (message_id: string, conversation_id: string, feedback: string) => Promise<void>;

    /** Rename a conversation by ID. */
    renameConversation: (conversation_name: string, conversation_id: string) => Promise<void | ErrorMessage>;

    /**
     * Create a document feedback record linking a query and a negatively rated answer
     * to a specific document/context (improves evaluation + reranking).
     */
    createDocumentFeedback: (query_id: string, negative_answer_id: string, doc_name: string, document_text: string, context: string, theme: string) => Promise<void | ErrorMessage>;
}


export const AuthContext = createContext<AuthContextType | undefined>(undefined);

/**
 * AuthProvider — supplies authentication + chat state/actions to child components.
 *
 * @remarks
 * - Wrap the entire app near the root.
 * - Performs session verification on mount (`verifyUser`) to hydrate `user`.
 * - Exposes all actions via {@link useAuth}.
 *
 * @example
 * ```tsx
 * import { AuthProvider } from "./context/AuthContext";
 * root.render(<AuthProvider><App /></AuthProvider>);
 * ```
 */
export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<UserProfile | null>(null);
    const [userMessages, setMessages] = useState<Message[] | null>(null);
    const [conversations, setConversations] = useState<Conversations[]>([]);
    const [loading, setLoading] = useState(true);

    /**
     * Register a new user.
     *
     * @param username - desired username
     * @param password - plaintext password (validated server-side)
     * @param email - user's email
     * @param role - user role (e.g., "user", "admin")
     * @returns `true` on success; otherwise {@link ErrorMessage}
     */
    const RegisterUser = async (username: string, password: string, email: string, role:string): Promise<boolean | ErrorMessage> => {
        try {
            const res = await registerAPI(username, password, email, role);
            if (res && res === true) {
                setUser({ username: username, email: email, verified: false, role: role })
                return true;
            }
            else if (res && typeof res === 'object' && 'error_message' in res) {
                setUser(null);
                return { error_message: res.error_message }
            }
            else {
                setUser(null);
                return { error_message: 'Something went wrong in registration' }
            }
        }
        catch (err) {
            setUser(null);
            return { error_message: String(err) };
        }
    }


    /**
     * Verify user with a code sent to their email.
     *
     * @param username - account username
     * @param code - verification code (string)
     * @returns `true` on success; otherwise {@link ErrorMessage}
     */
    const verifyCodeUser = async (username: string, code: string): Promise<boolean | ErrorMessage> => {
        try {
            const res = await verifyAPI(username, code);
            if (res === true) {
                return true;
            }
            else if (res && typeof res == 'object' && 'error_message' in res) {
                return { error_message: res.error_message }
            }
            else {
                return { error_message: 'Something went wrong in verification' }
            }
        }
        catch (err) {
            console.error("Verification failed", err);
            setUser(null);
            return { error_message: String(err) };
        }
    }

    /**
     * Resend verification code to user's email.
     *
     * @param username - account username
     * @param email - address to receive the code
     * @throws Error if the backend rejects the request
     */
    const resendCode = async (username: string, email: string): Promise<void> => {
        try {
            const res = await resendCodeAPI(username, email);
            if (res) {
                return;
            }
            else {
                throw new Error("Resend Code failed: No response from API");
            }
        }
        catch (err) {
            console.error("Resend Code failed", err);
            setUser(null);
            throw err;
        }
    }


    /**
     * Submit thumbs feedback for a message.
     *
     * @param message_id - message identifier
     * @param conversation_id - parent conversation id
     * @param feedback - "true" | "false" | "null" (string-encoded)
     * @throws Error if backend rejects request
     */
    const userFeedback = async (message_id: string, conversation_id: string, feedback: string | undefined): Promise<void> => {
        try {
            const res = await userFeedbackAPI(message_id, conversation_id, feedback);
            if (res) {
                return;
            }
            else {
                throw new Error("userFeedback failed: No response from API");
            }
        }
        catch (err) {
            console.error("userFeedback failed", err);
            setUser(null);
            throw err;
        }
    }

    /**
     * Login user and hydrate `user` state.
     *
     * @param username - account username
     * @param password - plaintext password
     * @returns On success: {@link LoginAPIOutput}; on failure: {@link ErrorMessage}
     */
    const loginUser = async (username: string, password: string): Promise<LoginAPIOutput | ErrorMessage> => {
        try {
            const res = await loginAPI(username, password);
            if (res && typeof res === 'object' && "user_details" in res) {
                setUser({ username: res.user_details.username, email: res.user_details.email, verified: res.user_details.verified, role: res.user_details.role })
                return res;
            }
            else if (res && typeof res === 'object' && "error_message" in res) {
                setUser(null);
                return { error_message: res.error_message }
            }
            // Ensure a return value in all cases
            setUser(null);
            return { error_message: "Unknown error during login" };
        }
        catch (err) {
            console.error("Login failed", err);
            setUser(null);
            return { error_message: String(err) }
        }
    }

    /**
     * Create a new conversation (e.g., "normal" | "lawsuit").
     *
     * @param conversation_name - title shown in UI
     * @param username - owner username
     * @param conversation_type - server-side flow selector
     * @returns The created conversation or `undefined` on error
     */
    const createConversation = async (conversation_name: string, username: string,conversation_type:string): Promise<Conversations | undefined> => {
        try {
            const res = await createConversationAPI(conversation_name, username, conversation_type);
            if (res) {
                setConversations(prev => [res, ...prev]);
                return res;
            }
            else {
                throw new Error("Conversation was not created properly");
            }
        }
        catch (err) {
            console.error(`Could not create new Conversation with name:${conversation_name}. Error:`, err);
        }
    }

    /**
     * Create a document feedback record (query → negative answer → document).
     *
     * @param query_id - originating query message id
     * @param negative_answer_id - answer message that was downvoted
     * @param doc_name - document title
     * @param document_text - offending passage or text
     * @param context - retrieval context shown with the answer
     * @param theme - category tag (e.g., "GDPR", "Greek Penal Code")
     * @returns void on success; {@link ErrorMessage} on failure
     */
    const createDocumentFeedback = async (query_id: string, negative_answer_id: string, doc_name: string, document_text: string, context: string, theme: string): Promise<void | ErrorMessage> => {
        try {
            const res = await createDocumentFeedbackAPI(query_id, negative_answer_id, doc_name, document_text, context, theme,);
            if (res) {
                return;
            }
            else {
                throw new Error("Conversation was not created properly");
            }
        }
        catch (err) {
            console.error(`Could not create new Document Feedback on Message:${query_id}. Error:`, err);
        }
    }

    /**
     * Fetch all conversations for the given user and hydrate state.
     *
     * @param username - owner username
     */
    const fetchConversations = async (username: string) => {
        try {
            const res = await getConversationsAPI(username);
            if (res) {
                setConversations(res)
            }
        }
        catch (err) {
            console.error(`Could not fetch Conversations from user:${username}. Error:`, err);
        }
    }

    /**
     * Create a new message inside a conversation and append locally.
     *
     * @param conversation_id - parent conversation
     * @param text - message content
     * @param role - "user" | "assistant"
     * @param id - UUID for the message (client-generated)
     * @param feedback - initial feedback marker (string or null)
     */
    const createMessage = async (conversation_id: string, text: string, role: string, id: string, feedback: string | null) => {
        try {
            const res = await createMessageAPI(conversation_id, text, role, id, feedback);
            if (!res) throw new Error("Message was not created properly2");
            console.log(res)
            const newMessage = { conversation_id, message:text, role, id, feedback, timestamp: new Date().toISOString() };
            if (res) {
                setMessages(prev => [...(prev ?? []), newMessage])
            }
            else {
                throw new Error("Message was not created properly");
            }
        }
        catch (err) {
            console.error(`Could not create new Message with id:${conversation_id}, message: ${text}. Error:`, err);
        }
    }

    /**
     * Fetch all messages for a conversation and hydrate `userMessages`.
     *
     * @param conversation_id - parent conversation
     */
    const fetchUserMessages = async (conversation_id: string) => {
        try {
            const messages = await getUserMessagesAPI(conversation_id);
            if (messages) {
                setMessages(messages);
            }
            else {
                setMessages(null);
            }
        }
        catch (err) {
            console.error(`Messages were not fetched from user in conversation ${conversation_id}`, err);
            setMessages(null);
        }
    }

    /**
     * Logout the user.
     *
     * - Clears server cookie via `logoutAPI()`
     * - Clears local `user` and `userMessages`
     */
    const logoutUser = async () => {
        try {
            const res = await logoutAPI();
            if (res) {
                setUser(null);
                setMessages(null);
            }
            else {
                setUser(null);
                setMessages(null);
            }

        }
        catch (err) {
            setUser(null);
            setMessages(null);
            console.error("Logout failed", err);
        }
    }

    /**
     * Rename an existing conversation.
     *
     * @param conversation_id - target id
     * @param conversation_name - new title
     * @returns `void` on success or {@link ErrorMessage} on failure
     */
    const renameConversation = async (conversation_id: string, conversation_name: string): Promise<void | ErrorMessage> => {
        try {
            // console.log("Renaming Conversation", conversation_name, conversation_id);
            const res = await renameConversationAPI(conversation_name, conversation_id);

            if (typeof res === 'object' && 'error_message' in res) {
                return { error_message: res.error_message };
            }

            // Success: do nothing
            return;
        }
        catch (err) {
            console.error("Updating Conversation Name failed", err);
            setUser(null);
            return { error_message: String(err) }
        }
    }

    /**
     * On mount: verify existing session with backend and hydrate `user`.
     *
     * Behavior:
     * - Calls `verifyUser()` (reads server-side cookie).
     * - Sets `user` if valid; clears otherwise.
     * - Always clears `loading` at the end.
     */
    useEffect(() => {
        const initialize = async () => {
            try {
                const res = await verifyUser();
                if (res) {
                    setUser(res);
                }
            }
            catch {
                setUser(null);
            }
            finally {
                setLoading(false);
            }
        };
        initialize();
    }, [])

    return (
        <AuthContext.Provider value={{ user, userMessages, createDocumentFeedback, renameConversation, userFeedback, resendCode, verifyCodeUser, RegisterUser, loginUser, logoutUser, fetchUserMessages, loading, conversations, createConversation, createMessage, fetchConversations }}>
            {children}
        </AuthContext.Provider>
    )

}

/**
 * Hook to access {@link AuthContextType} anywhere in the tree.
 *
 * @remarks
 * Must be used within an {@link AuthProvider}. Throws a descriptive error otherwise.
 *
 * @example
 * ```tsx
 * const { user, loginUser, logoutUser, fetchConversations } = useAuth();
 * ```
 */
export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
};
