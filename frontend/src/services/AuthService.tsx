/**
 * @packageDocumentation
 *
 * AILA Backend Service (Axios)
 * ============================
 * Thin, typed wrappers around the AILA backend HTTP API. Centralizes request
 * shapes, return types, and error handling for:
 *
 * - Auth: login, logout, register, verify, resend-code, session verify
 * - Conversations: create, list, rename
 * - Messages: create, list
 * - Feedback: thumbs on messages, document feedback creation
 * - Chat: non-streaming `/request` helper (UI uses streaming elsewhere)
 *
 * Conventions
 * -----------
 * - All calls use `withCredentials: true` (HttpOnly cookie for JWT).
 * - On Axios errors caught here, functions return `{ error_message: string }`
 *   when applicable, or `undefined` for fire-and-forget helpers.
 * - Functions that must return data bubble it through `.data` using proper types.
 * - Some functions intentionally rethrow Axios errors (documented below) so the
 *   UI can display rich error states.
 *
 * Usage Example
 * -------------
 * ```ts
 * import { loginAPI } from '../services/AuthService';
 *
 * const res = await loginAPI('alice', 's3cret');
 * if ('user_details' in res) {
 *   console.log(res.user_details.username);
 * } else {
 *   console.error(res.error_message);
 * }
 * ```
 */

import axios from 'axios';
import api from '../api/axios.jsx';
import type {
  LoginAPIOutput,
  UserProfile,
  Message,
  Conversations,
  ErrorMessage,
} from '../models/Types.jsx';

/**
 * POST /login — Authenticate with username/password.
 *
 * Body:
 * - `{ username, password }`
 *
 * Returns:
 * - **Success**: {@link LoginAPIOutput} `{ user_details: UserProfile }`
 * - **Failure**: {@link ErrorMessage} `{ error_message }`
 *
 * Notes:
 * - Sets an HttpOnly cookie on the backend; requires `withCredentials: true`.
 */
const loginAPI = async (username: string, password: string): Promise<LoginAPIOutput | ErrorMessage> => {
  try {
    const response = await api.post('/login', { username, password }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      return { error_message: err.response?.data.detail };
    }
    console.error("Non-Axios error:", err);
    return { error_message: String(err) };
  }
};

/**
 * POST /update_conversation — Rename a conversation.
 *
 * Body:
 * - `{ conversation_name, conversation_id }`
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: {@link ErrorMessage}
 */
const renameConversationAPI = async (conversation_name: string, conversation_id: string): Promise<boolean | ErrorMessage> => {
  try {
    const response = await api.post('/update_conversation', { conversation_name, conversation_id }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      return { error_message: err.response?.data.detail };
    }
    console.error("Non-Axios error:", err);
    return { error_message: String(err) };
  }
};

/**
 * POST /register — Create a new user.
 *
 * Body:
 * - `{ username, password, email, role }`
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: {@link ErrorMessage}
 */
const registerAPI = async (username: string, password: string, email: string, role: string): Promise<boolean | ErrorMessage> => {
  try {
    const response = await api.post('/register', { username, password, email, role }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      return { error_message: err.response?.data.detail };
    }
    console.error("Non-Axios error:", err);
    return { error_message: String(err) };
  }
};

/**
 * POST /verify — Verify a user via emailed code.
 *
 * Body:
 * - `{ username, code }`
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: {@link ErrorMessage}
 */
const verifyAPI = async (username: string, code: string): Promise<boolean | ErrorMessage> => {
  try {
    const response = await api.post('/verify', { username, code }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      return { error_message: err.response?.data.detail };
    }
    console.error("Non-Axios error:", err);
    return { error_message: String(err) };
  }
};

/**
 * POST /resend-code — Resend email verification code.
 *
 * Body:
 * - `{ username, email }`
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: `undefined` (error is logged)
 */
const resendCodeAPI = async (username: string, email: string): Promise<boolean | undefined> => {
  try {
    const response = await api.post('/resend-code', { username, email }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * POST /user_feedback — Submit thumbs feedback for a message.
 *
 * Body:
 * - `{ message_id, conversation_id, feedback }`
 *   - `feedback`: "true" | "false" | "null" (string)
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: `undefined` (error is logged)
 */
const userFeedbackAPI = async (message_id: string, conversation_id: string, feedback: string | undefined): Promise<boolean | undefined> => {
  try {
    await api.post('/user_feedback', { message_id, conversation_id, feedback }, { withCredentials: true });
    return true;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * POST /new_conversation — Create a conversation.
 *
 * Body:
 * - `{ conversation_name, username, conversation_type }`
 *   - `conversation_type`: "normal" | "lawsuit"
 *
 * Returns:
 * - **Success**: {@link Conversations}
 * - **Failure**: `undefined` (error is logged)
 */
const createConversationAPI = async (conversation_name: string, username: string, conversation_type: string): Promise<Conversations | undefined> => {
  try {
    const response = await api.post('/new_conversation', { conversation_name, username, conversation_type }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * POST /new_document_feedback — Create a document-feedback record.
 *
 * Purpose:
 * - Links a query message to a negatively rated answer and supplies the
 *   source document text + context + theme for evaluation/reranking.
 *
 * Body:
 * - `{ query_id, negative_answer_id, doc_name, doc_text, context, theme }`
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: `undefined` (error is logged)
 */
const createDocumentFeedbackAPI = async (query_id: string, negative_answer_id: string, doc_name: string, document_text: string, context: string, theme: string): Promise<boolean | undefined> => {
  try {
    const response = await api.post('/new_document_feedback', { query_id, negative_answer_id, doc_name, doc_text: document_text, context, theme }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * POST /new_message — Create a message inside a conversation.
 *
 * Body:
 * - `{ conversation_id, text, role, id, feedback }`
 *   - `role`: "user" | "assistant"
 *   - `id`: client-generated UUID for dedupe
 *
 * Returns:
 * - **Success**: {@link Message}
 * - **Failure**: `undefined` (error is logged)
 */
const createMessageAPI = async (conversation_id: string, text: string, role: string, id: string, feedback: string | null): Promise<Message | undefined> => {
  try {
    const response = await api.post('/new_message', { conversation_id, text, role, id, feedback }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * GET /user_conversations — List conversations for a user.
 *
 * Query:
 * - `?username=<string>`
 *
 * Returns:
 * - **Success**: {@link Conversations}[]
 * - **Failure**: `undefined` (error is logged)
 */
const getConversationsAPI = async (username: string): Promise<Conversations[] | undefined> => {
  try {
    const response = await api.get('/user_conversations', {
      params: { username },
      withCredentials: true
    });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * GET /messages — Fetch messages for a conversation.
 *
 * Query:
 * - `?conversation_id=<uuid>`
 *
 * Returns:
 * - **Success**: {@link Message}[]
 * - **Failure**: throws Axios error (caller must handle)
 *
 * Rationale:
 * - This function deliberately lets Axios errors bubble so the UI can
 *   differentiate between empty state vs. network/auth errors.
 */
const getUserMessagesAPI = async (conversation_id: string): Promise<Message[] | undefined> => {
  const response = await api.get('/messages', {
    params: { conversation_id },
    withCredentials: true
  });
  return response.data;
};

/**
 * GET /get_user — Verify session via cookie and return profile.
 *
 * Returns:
 * - **Success**: {@link UserProfile}
 * - **Failure**: throws Axios error (caller must handle)
 */
const verifyUser = async (): Promise<UserProfile | undefined> => {
  const response = await api.get('/get_user', { withCredentials: true });
  return response.data;
};

/**
 * POST /request — Direct (non-streaming) chat call.
 *
 * Body:
 * - `{ message }`
 *
 * Returns:
 * - **Success**: backend JSON response
 * - **Failure**: `void` (error is logged)
 *
 * Note:
 * - The UI typically uses the **streaming** fetch variant instead of this helper.
 */
const requestAPI = async (userQuery: string): Promise<boolean | void> => {
  try {
    const response = await api.post('/request', { message: userQuery }, { withCredentials: true });
    return response.data;
  } catch (err) {
    if (err instanceof Error) console.error(err.message);
    else console.error(err);
  }
};

/**
 * POST /logout — Clear server-side session (HttpOnly cookie).
 *
 * Returns:
 * - **Success**: `true`
 * - **Failure**: `false` (and logs the error)
 */
const logoutAPI = async (): Promise<boolean | undefined> => {
  try {
    const response = await api.post('/logout');
    if (response) return true;
  } catch (err) {
    console.error("Logout failed:", err);
    return false;
  }
};

export {
  loginAPI,
  getUserMessagesAPI,
  createDocumentFeedbackAPI,
  userFeedbackAPI,
  resendCodeAPI,
  verifyAPI,
  renameConversationAPI,
  logoutAPI,
  registerAPI,
  requestAPI,
  verifyUser,
  createConversationAPI,
  createMessageAPI,
  getConversationsAPI
};
