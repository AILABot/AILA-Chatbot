"""
Entities Package — SQLAlchemy 2.0 ORM Models (PostgreSQL + UUID + UTC)
======================================================================

The `entities` package defines the ORM models of the application, mapping
database tables to Python classes using SQLAlchemy 2.0-typed mappings.
These classes form the persistence backbone and are consumed by DAOs
(`daos` package) to perform CRUD and transactional operations.

Tech Stack & Conventions
------------------------
- PostgreSQL with native UUID columns
- Timezone-aware timestamps (UTC)
- SQLAlchemy 2.0 style `Mapped[...]` + `mapped_column(...)`
- Clear foreign keys for relational integrity

Contents
--------
- User
    Represents a registered user in the system.
    * Stores credentials (hashed password) and session token
    * Holds role and verification status
    * Tracks email verification code and timestamp

- Conversation
    Represents a conversation belonging to a user.
    * Fields: `id` (UUID PK), `conversation_name`, `user_id` (FK → app_user.id)
    * Tracks `last_updated` (UTC, tz-aware)
    * NEW: `conversation_type` (e.g., "normal", "lawsuit") used by the API layer
      to choose between standard Q&A vs. legal complaint drafting flows

- Message (UserMessage)
    Represents a single message within a conversation.
    * Fields: `id` (UUID PK), `conversation_id` (FK → conversation.id)
    * Stores `message_text`, `role` ("user" | "assistant" | "system")
    * Records `date_created_on` (UTC, tz-aware)
    * Optional `feedback` flag (True/False/None)

- DocumentFeedback
    Captures feedback tying a **query** to a **negatively rated answer** and its document context.
    * Table: `document_feedback`
    * Fields:
        - `id` (UUID PK)
        - `query_id` (FK → message.id)            # the originating user query
        - `negative_answer_id` (FK → message.id)  # the answer marked negative
        - `doc_name`, `doc_text`, `context`, `theme`
        - `date_created` (UTC, tz-aware)
    * Used for evaluation, analytics, and improving retrieval/reranking

Usage
-----
- DAOs import these models to implement repository operations.
- API routers and services reference entities via DAOs for isolation and testability.
- Migrations should keep schema in sync with model attributes (incl. `conversation_type`).

"""
