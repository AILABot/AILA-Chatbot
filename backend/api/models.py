"""
Pydantic models used for request/response validation and API data contracts.

Each class defines the structure of data expected in API endpoints, ensuring
validation and automatic OpenAPI schema generation.
"""

from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field, HttpUrl
from fastapi import Form

class FileRec(BaseModel):
    """Metadata for an uploaded/stored file."""
    original: str = Field(..., description="Original filename as provided by the client.", example="report.pdf")
    path: str = Field(..., description="Server-side storage path or key.", example="/uploads/2025/09/report-1234.pdf")
    mime: str = Field(..., description="MIME type of the file.", example="application/pdf")
    public_url: Optional[HttpUrl] = Field(
        None,
        description="Publicly accessible URL if served via CDN/static hosting.",
        example="https://cdn.example.com/uploads/report-1234.pdf"
    )


class DocumentFeedbackDetails(BaseModel):
    """
    Details captured when a user marks a retrieved document/answer as unhelpful or incorrect.
    Useful for offline evaluation, retraining, and retrieval quality dashboards.
    """
    query_id: str = Field(..., description="Identifier of the originating user query.", example="q_7f1b6d3a")
    negative_answer_id: str = Field(..., description="Identifier of the answer the user disliked.", example="ans_c0ffee42")
    doc_name: str = Field(..., description="Document title or filename.", example="GDPR_Article_6")
    doc_text: str = Field(..., description="Raw text (or excerpt) of the document passage shown to the user.")
    context: str = Field(..., description="Retrieved context chunk shown alongside the answer.")
    theme: str = Field(..., description="Optional tag/category (e.g., 'GDPR', 'Phishing', 'Case Law').", example="GDPR")


class UserCredentials(BaseModel):
    """
    Represents login credentials for a user.
    """
    username: str
    """The username of the user"""
    password: str
    """The plaintext password provided for authentication."""

class ConversationCreationDetails(BaseModel):
    """
    Represents details needed to create a new conversation.
    """
    username: str
    """The username of the conversation owner."""
    conversation_name: str
    """A human-readable title for the conversation."""
    conversation_type:str
    """The type of the conversation (normal or other depending on the tool used)"""

class UpdateConversationDetails(BaseModel):
    """
    Represents details required to update an existing conversation.
    """
    conversation_name: str
    """New name/title for the conversation."""
    conversation_id: str
    """Unique identifier of the conversation to update."""

class NewMessage(BaseModel):
    """
    Represents a new message to be created in a conversation.
    """
    feedback: str | None
    """Optional feedback flag (True/False, None if unset)."""
    id: str
    """Unique identifier of the message."""
    conversation_id: str
    """The ID of the conversation the message belongs to."""
    text: str
    """The text content of the message."""
    role: str
    """The role of the sender (e.g., 'user', 'assistant', 'system')."""


class UserOpenData(BaseModel):
    """
    Publicly shareable user data (non-sensitive).
    """
    email: str
    """User's email address."""
    username: str
    """User's username."""


class Message(BaseModel):
    """
    Represents a message sent in an API request (e.g., chat interaction).
    """
    message: str
    """The current message being sent."""
    conversation_history: List[dict]
    """History of previous messages in the conversation."""
    web_search_tool: bool
    """A flag that defines whether the online mode should be enabled or not"""
    conversation_type: str
    """The type of the conversation (normal or other depending on the tool used)"""
    conversation_id: str = Form(...)
    """The ID of the conversation the message belongs to."""


class UserAuthentication(BaseModel):
    """
    Authentication response returned after login attempts.
    """
    authenticated: bool
    """Whether the authentication was successful."""
    detail: str
    """Additional information or error message."""
    user_details: UserCredentials | None
    """User details if authenticated, otherwise None."""


class UserData(BaseModel):
    """
    Represents data required to register a new user.
    """
    username: str
    """Desired username."""
    password: str
    """Password chosen by the user."""
    email: str
    """Email address of the user."""
    role: str
    """The role of the specified user (can be normal user or lawyer)"""


class VerifCode(BaseModel):
    """
    Represents a request to verify a user's email.
    """
    username: str
    """Username associated with the verification code."""
    code: str
    """Verification code provided by the user."""


class UserFeedback(BaseModel):
    """
    Represents feedback on a user message.
    """
    message_id: str
    """The ID of the message being reviewed."""
    conversation_id: str
    """The conversation to which the message belongs."""
    feedback: str | None
    """Feedback value (True=positive, False=negative, None=unset)."""
