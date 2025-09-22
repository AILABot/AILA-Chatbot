"""
User Messages DAO

Purpose
-------
Data-access layer for the `UserMessage` ORM entity. Provides:
- Message creation
- Retrieval by conversation (chronological)
- Feedback updates (thumbs up/down style boolean)

Design
------
- Requires an active SQLAlchemy `Session` provided by the caller.
- Keeps business rules (auth, validation, rate limits) in higher layers.
- Retrieval uses a subquery for "latest-first then re-order ascending" semantics;
  see "Performance & Alternatives" for simpler approaches.

Entity (expected columns)
-------------------------
UserMessage:
- id: UUID / primary key
- conversation_id: UUID (FK to conversations)
- role: str (e.g., "user", "assistant", "system")
- message_text: str
- feedback: bool | None
- date_created_on: datetime (creation timestamp)

Usage
-----
.. code-block:: python

    from sqlalchemy.orm import Session
    from backend.database.connection_engine import connection_engine
    from backend.database.entities.messages import UserMessage
    from backend.database.daos.user_messages_dao import UserMessagesDao

    dao = UserMessagesDao()
    with Session(connection_engine) as session:
        # Create a message (caller may choose to commit here or after a batch)
        msg = UserMessage(
            id=..., conversation_id=..., role="user", text="Acknowledge the Tribal Chief!",
        )
        dao.createMessage(session, msg)
        session.commit()

        # Fetch messages for a conversation (chronological)
        messages = dao.fetchMessagesByConversationId(session, conversation_id=msg.conversation_id)

        # Update feedback on a specific message
        dao.updateMessageFeedback(
            session,
            conversation_id=msg.conversation_id,
            message_id=msg.id,
            feedback=True,
        )

Error Handling
--------------
- Methods catch generic `Exception`, print a message, and re-raise.
  Prefer structured logging (e.g., `logger.exception(...)`) over `print(...)`.
- `updateMessageFeedback` uses `.one()` which raises:
  - `NoResultFound` if the message doesn’t exist or doesn’t belong to the conversation.
  - `MultipleResultsFound` if data integrity is violated (should not happen with proper PK/FK).
"""

from sqlalchemy.orm import Session, aliased
from backend.database.entities.messages import UserMessage
from uuid import UUID
from sqlalchemy import desc, asc
from typing import List

class UserMessagesDao:
    """
    Data Access Object (DAO) for managing User Messages.
    Provides methods to create, fetch, and update messages within conversations.
    """

    def createMessage(self, session: Session, userMessage: UserMessage) -> UserMessage:
        """
        Create a new user message record.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        userMessage : UserMessage
            Message entity instance to be added.

        Returns
        -------
        UserMessage
            The message object that was added.

        Raises
        ------
        Exception
            If the insert operation fails.
        """
        try:
            session.add(userMessage)
            return userMessage
        except Exception as e:
            print(f"Error in UserMessagesDao.createMessage. Error Message: {e}")
            raise e
        
    def fetchMessageById(self, session: Session, id: UUID) -> List[UserMessage]:
        """Return all `UserMessage` rows matching a given UUID.

        This mirrors the existing behavior used elsewhere in your codebase
        (returns a list that may be empty).

        Args:
            session: Active SQLAlchemy session.
            id: UUID of the message to fetch.

        Returns:
            list[UserMessage]: Zero or more matches (usually 0 or 1).

        Raises:
            Exception: Propagates unexpected database errors.
        """
        try:
            return (
                session.query(UserMessage)
                .filter(UserMessage.id == id)
                .all()
            )
        except Exception as e:
            print("Error in UserMessagesDao.fetchMessageById (id=%s)", id)
            raise e

    def fetchMessagesByConversationId(self, session: Session, conversation_id: UUID):
        """
        Fetch all messages in a conversation, ordered by creation time (ascending).
        Internally, retrieves the latest messages first via a subquery,
        then re-orders them chronologically.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation_id : UUID
            Unique identifier of the conversation.

        Returns
        -------
        list[UserMessage]
            List of messages belonging to the specified conversation.

        Raises
        ------
        Exception
            If query fails.
        """
        try:
            subq = (
                session.query(UserMessage)
                .filter(UserMessage.conversation_id == conversation_id)
                .order_by(desc(UserMessage.date_created_on))
                # Optional: uncomment limit for recent N messages
                # .limit(5)
            ).subquery()

            recentMessages = aliased(UserMessage, subq)

            messages = (
                session.query(recentMessages)
                .order_by(asc(recentMessages.date_created_on))
                .all()
            )
            return messages
        except Exception as e:
            print(f"Error in UserMessagesDao.fetchMessagesByConversationId. Error Message: {e}")
            raise e

    def updateMessageFeedback(self, session: Session, conversation_id: UUID, message_id: UUID, feedback: str):
        """
        Update feedback status of a specific message within a conversation.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation_id : UUID
            Unique identifier of the conversation.
        message_id : UUID
            Unique identifier of the message.
        feedback : bool
            New feedback value (string).

        Raises
        ------
        Exception
            If update fails or message is not found.
        """
        try:
            message = (
                session.query(UserMessage)
                .filter(
                    UserMessage.id == message_id,
                    UserMessage.conversation_id == conversation_id,
                )
                .one()
            )
            message.feedback = feedback
            session.commit()
        except Exception as e:
            print(f"Error in UserMessagesDao.updateMessageFeedback. Error Message: {e}")
            raise e
