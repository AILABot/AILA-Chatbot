"""
Conversation ORM Model
=======================

The ``Conversation`` ORM model represents a user-owned conversation record stored
in the ``conversation`` PostgreSQL table. It is implemented with SQLAlchemy
2.0-style typing and PostgreSQL UUID columns.

Key features
~~~~~~~~~~~~
- PostgreSQL-native UUID primary key (``id``)
- Human-readable name (``conversation_name``)
- Foreign key to the owning user (``user_id`` → ``app_user.id``)
- Timezone-aware ``last_updated`` timestamp (UTC)
- **Conversation type** (``conversation_type``) used by the API layer to steer flows
  such as standard Q&A (``"normal"``) or legal complaint drafting (``"lawsuit"``)

Integration notes
~~~~~~~~~~~~~~~~~
- The FastAPI router reads ``conversation_type`` to route behavior in ``/request``:
  - ``"normal"`` → LLM pipeline (web-search or RAG) and streamed legal answer.
  - ``"lawsuit"`` → intake gatekeeper, follow-up question generation, and final Greek
    criminal complaint (Μήνυση) draft (with optional DOCX + S3 delivery).
- This model is persisted via SQLAlchemy; request/response contracts are defined in
  ``backend.api.models``; pipeline orchestration lives in ``llm_pipeline``; evidence
  handling & DOCX generation live in ``prompt_utilities``; S3 I/O in ``aws_bucket_funcs``.
"""

from backend.database.config.connection_engine import declarativeBase
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import Column, ForeignKey, DateTime, TEXT
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID
from datetime import datetime, timezone

class Conversation(declarativeBase):
    """
    ORM model for the `conversation` table.
    Represents a conversation belonging to a specific user.

    Attributes
    ----------
    id : UUID
        Primary key. Unique identifier for the conversation.
    conversation_name : str
        Human-readable name/title of the conversation.
    user_id : UUID
        Foreign key reference to the `app_user` table (the owner of the conversation).
    last_updated : datetime
        Timestamp of the last update to the conversation (timezone-aware, UTC).
        Defaults to the current UTC time on insert.
    conversation_type : str | None
        Optional tag used by the API to determine the handling flow (e.g., "normal", "lawsuit").
    """

    __tablename__ = 'conversation'

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True
    )
    """Primary key. UUID of the conversation."""

    conversation_name: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Name of the conversation (cannot be null)."""

    user_id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey('app_user.id'), nullable=False
    )
    """Foreign key reference to the `app_user` table (owner)."""

    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc)
    )
    """Timestamp when the conversation was last updated (UTC, timezone-aware)."""

    conversation_type: Mapped[str] = mapped_column(
            TEXT, nullable=True
    )
    """Optional flow selector (e.g., "normal", "lawsuit"). Used by the router and pipeline."""

    def __init__(self, conversation_id: UUID, conversation_name: str, user_id: UUID, last_updated, conversation_type:str):
        """
        Initialize a new Conversation object.

        Parameters
        ----------
        conversation_id : UUID
            Unique identifier for the conversation.
        conversation_name : str
            Name/title of the conversation.
        user_id : UUID
            The ID of the user who owns this conversation.
        last_updated : datetime | str
            Last updated timestamp. Accepts datetime or ISO8601 string.
        conversation_type : str
            Optional flow selector (e.g., "normal", "lawsuit") for API behavior.
        """
        self.id = conversation_id
        self.conversation_name = conversation_name
        self.user_id = user_id
        self.conversation_type = conversation_type
        if isinstance(last_updated, str):
            self.last_updated = datetime.fromisoformat(last_updated)
        else:
            self.last_updated = last_updated

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Conversation.

        Returns
        -------
        str
            A formatted string containing user ID, conversation name, and last updated timestamp.
        """
        return (
            f"User: id:{self.user_id}, conversation: {self.conversation_name}, time_created: {self.last_updated}"
        )
