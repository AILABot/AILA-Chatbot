/**
* @packageDocumentation
*
* @remarks
 * These types define the data contracts used across the frontend.
 * They ensure strong typing when interacting with the backend API.
*
 * Each type corresponds to a resource or response structure
 * exchanged between frontend and backend.
*/

/**
 * Represents a registered user in the system.
 *
*
* @property username - Unique identifier chosen by the user.
* @property email - User’s email address (must be unique).
* @property verified - Whether the email is verified.
* - `true` → user’s email is verified
* - `false` → verification pending or failed
* - `null` → verification status unknown (e.g., session expired)
*/
export type UserProfile = {
  username: string;
  email: string;
  verified: boolean | null;
  role: string | null;
};

/**
* Represents a chat message inside a conversation.
*
* @property feedback - User’s feedback on assistant’s reply.
* - `true` → positive feedback
* - `false` → negative feedback
* - `null` → no feedback given
* @property id - Unique identifier (UUID).
* @property message - Message text.
* @property timestamp - ISO-8601 formatted timestamp.
* @property role - Sender role ("user" | "assistant").
*/
export type Message = {
  feedback: string | null;
  id: string;
  message: string;
  timestamp: string;
  role: string;
};

/**
* Returned by the login API when authentication succeeds.
*
* @property user_details - The authenticated user’s profile info.
*/
export type LoginAPIOutput = {
  user_details: UserProfile;
};

/**
* Represents a conversation/chat thread.
*
* @property conversation_name - Display name/title of the conversation.
* @property conversation_id - Unique identifier (UUID).
*/
export type Conversations = {
  conversation_name: string;
  conversation_id: string;
  conversation_type: string;
};

/**
* Represents an error object returned by API calls.
*
* @property error_message - Descriptive error message from backend.
*/
export type ErrorMessage = {
  error_message: string;
};
