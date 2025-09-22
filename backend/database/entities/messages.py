"""
UserMessage ORM Model
=====================

The ``UserMessage`` ORM model represents a single message record within a
conversation. Each message is tied to a ``Conversation`` entity via a
foreign key and may optionally carry user feedback.

Key features
~~~~~~~~~~~~
- PostgreSQL-native UUID primary key (``id``)
- Foreign key reference to ``conversation.id`` (``conversation_id``)
- Timezone-aware ``date_created_on`` timestamp (UTC)
- Message text content (``message_text``)
- Sender role indicator (``role``)
- Optional feedback flag (``feedback``)

"""

from backend.database.config.connection_engine import declarativeBase
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import ForeignKey, DateTime, Boolean, TEXT
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID
from datetime import datetime, timezone

class UserMessage(declarativeBase):
    """
    ORM model for the `message` table.
    Represents a single message within a conversation.

    Attributes
    ----------
    id : UUID
        Primary key. Unique identifier for the message.
    conversation_id : UUID
        Foreign key reference to the `conversation` table.
    date_created_on : datetime
        Timestamp when the message was created.
    message_text : str
        Content of the message.
    role : str
        Role of the sender (e.g., "user", "assistant", "system").
    feedback : bool | None
        Optional feedback flag for the message (True/False). Default is None.
    """

    __tablename__ = 'message'

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True
    )
    """Primary key. UUID of the message."""

    conversation_id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey('conversation.id'), nullable=False
    )
    """Foreign key to the conversation this message belongs to."""

    date_created_on: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc)
    )
    """Timestamp when the message was created. Defaults to current UTC time."""

    message_text: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Text content of the message (cannot be null)."""

    role: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Role of the message sender (e.g., user, assistant, system)."""

    feedback: Mapped[str] = mapped_column(
        TEXT, nullable=True
    )
    """Optional feedback flag (string)."""

    def __init__(
        self,
        message_id: UUID,
        conversation_id: UUID,
        message: str,
        date_created_on,
        role: str,
        feedback: str | None = None,
    ):
        """
        Initialize a new UserMessage object.

        Parameters
        ----------
        message_id : UUID
            Unique identifier of the message.
        conversation_id : UUID
            ID of the conversation this message belongs to.
        message : str
            The content of the message.
        date_created_on : datetime | str
            Timestamp when the message was created. Accepts datetime or ISO8601 string.
        role : str
            The role of the sender (user/assistant/system).
        feedback : string | None, optional
            Feedback flag for the message (default is None).
        """
        self.id = message_id
        self.conversation_id = conversation_id
        self.message_text = message
        self.role = role
        self.feedback = feedback
        if isinstance(date_created_on, str):
            self.date_created_on = datetime.fromisoformat(date_created_on)
        else:
            self.date_created_on = date_created_on

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the message.

        Returns
        -------
        str
            A formatted string containing conversation ID, message text, role, and creation timestamp.
        """
        return (
            f"Conversation: id:{self.conversation_id}, "
            f"role: {self.role}, "
            f"message: {self.message_text}, "
            f"time_created: {self.date_created_on}"
        )
