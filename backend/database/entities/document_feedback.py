"""
DocumentFeedback ORM Model
==========================

The ``DocumentFeedback`` ORM model represents a feedback event captured during a
Q&A interaction. It links a *query message* to a *negatively rated answer* and
stores the offending document passage, its context, and a theme tag for analysis.

Table
-----
- PostgreSQL table: ``document_feedback`` (SQLAlchemy 2.0 typed mappings)
- PostgreSQL-native UUID columns for identifiers

Key Features
~~~~~~~~~~~~
- UUID primary key (``id``)
- Foreign keys to messages (``query_id`` and ``negative_answer_id`` → ``message.id``)
- Text fields capturing document title (``doc_name``), raw text (``doc_text``), and context (``context``)
- Theme/category tag (``theme``) for analytics
- Timezone-aware creation timestamp (UTC)

Integration Notes
~~~~~~~~~~~~~~~~~
- Created via the FastAPI endpoint ``/new_document_feedback`` (see `backend.api.fast_api`)
- Complements the pipeline’s evaluation loop by logging unhelpful/incorrect answers
"""

from backend.database.config.connection_engine import declarativeBase
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import ForeignKey, DateTime, Boolean, TEXT
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID
from datetime import datetime, timezone

class DocumentFeedback(declarativeBase):
    """
    ORM model for the `document_feedback` table.
    Stores feedback linking a user query to a negatively rated answer and its document.

    Attributes
    ----------
    id : UUID
        Primary key for this feedback record.
    query_id : UUID
        Foreign key to the original **query** message (`message.id`).
    negative_answer_id : UUID
        Foreign key to the **answer** message that was marked negative (`message.id`).
    doc_name : str
        Human-readable name/title of the source document.
    doc_text : str
        Raw text or excerpt of the document passage presented to the user.
    context : str
        Retrieval context chunk shown with the answer.
    theme : str
        Tag/category for analytics (e.g., "GDPR", "Greek Penal Code", "Phishing").
    date_created : datetime
        Time when the feedback was recorded (UTC, timezone-aware).
    """

    __tablename__ = 'document_feedback'

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True
    )
    """Primary key (UUID) for this feedback record."""

    query_id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey('message.id'), nullable=False
    )
    """Foreign key to the originating query message (`message.id`)."""

    negative_answer_id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey('message.id'), nullable=False
    )
    """Foreign key to the negatively rated answer message (`message.id`)."""

    doc_name: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Document title or identifier."""

    doc_text: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Document text or extracted passage displayed to the user."""

    context: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Retrieved context shown alongside the answer (for auditing/debugging)."""

    theme: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Categorical tag for analytics (e.g., 'GDPR', 'GPC', 'Phishing')."""

    date_created: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc)
    )
    """Creation timestamp (UTC, timezone-aware)."""

    def __init__(
        self,
        doc_id: UUID,
        query_id: UUID,
        negative_answer_id: UUID,
        doc_name: str,
        doc_text: str,
        context: str,
        theme: str,
        date_created,
    ):
        """
        Initialize a new DocumentFeedback object.

        Parameters
        ----------
        doc_id : UUID
            Unique identifier for this feedback record.
        query_id : UUID
            ID of the original query message (`message.id`).
        negative_answer_id : UUID
            ID of the negatively rated answer message (`message.id`).
        doc_name : str
            Title or name of the associated document.
        doc_text : str
            Document passage or full text shown to the user.
        context : str
            Retrieval context segment provided with the answer.
        theme : str
            Category label for analytics (e.g., "GDPR", "Greek Penal Code").
        date_created : datetime | str
            Creation timestamp; accepts a `datetime` or ISO8601 string.
        """
        self.id = doc_id
        self.query_id = query_id
        self.negative_answer_id = negative_answer_id
        self.doc_name = doc_name
        self.doc_text = doc_text
        self.context = context
        self.theme = theme
        self.date_created = date_created
        if isinstance(date_created, str):
            self.date_created = datetime.fromisoformat(date_created)
        else:
            self.date_created = date_created

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the feedback record.

        Returns
        -------
        str
            A formatted string summarizing linkage and document metadata.
        """
        return (
            f"Conversation: id:{self.id}, "
            f"query_id: {self.query_id}, "
            f"negative_answer_id: {self.negative_answer_id}, "
            f"doc_name: {self.doc_name}"
        )
