"""
DAOs Package — Data Access Layer (SQLAlchemy 2.0)
=================================================

The `daos` package provides the Data Access Layer for the application.
It encapsulates all interactions with SQLAlchemy ORM entities, providing
clean CRUD APIs for the service layer while hiding direct query details.

Conventions
-----------
- SQLAlchemy 2.0 typed mappings (Mapped[...] / mapped_column)
- Session lifecycle (open/commit/rollback) is handled by callers
- DAOs surface exceptions so upper layers decide error policy

Contents
--------
- UserDao
    Handles user persistence:
    * Creates users with password hashing
    * Fetches users by username or email
    * Updates verification status and codes
    * Manages authentication/session tokens

- ConversationDao
    Manages conversation records:
    * Creates new conversations
    * Fetches conversations by user ID or name
    * Updates conversation name, `conversation_type`, and last-updated timestamp

- UserMessagesDao
    Manages user message records:
    * Creates messages within a conversation
    * Fetches messages by conversation (chronological order)
    * Updates message feedback status (True/False/None)

- DocumentFeedbackDao   ← NEW
    Persists and reads feedback linking a query and a negatively rated answer
    to the associated document/context:
    * createDocument(session, DocumentFeedback) — stages a feedback record
    * fetchDocs(session) — returns all `DocumentFeedback` rows for analytics/evaluation

Notes
-----
- Entities live under `backend.database.entities.*` (PostgreSQL + UUID + UTC).
- Keep migrations in sync with entity schema (e.g., `conversation_type`).
- Service layer composes DAOs to implement business workflows.
"""
