"""
Conversation DAO

Purpose
-------
Provides a thin data-access layer for the `Conversation` ORM entity:
- Create conversations
- Query by user, by name, or both
- Update `last_updated` and `conversation_name`

Design
------
- Requires an active SQLAlchemy `Session` supplied by the caller (no session
  creation inside the DAO). This keeps transaction boundaries in the service
  layer where they belong.
- Uses straightforward ORM queries (`session.query(...).filter(...).all()`).
- Update operations fetch the target row and mutate attributes, then `commit()`.

Entity
------
Conversation (imported from `backend.database.entities.conversations`)
Expected columns used here:
- id (UUID/string)
- user_id (UUID)
- conversation_name (str)
- last_updated (timestamp/datetime or str)

Usage
-----
.. code-block:: python

    from sqlalchemy.orm import Session
    from backend.database.connection_engine import connection_engine
    from backend.database.entities.conversations import Conversation
    from backend.database.daos.conversation_dao import ConversationDao  # wherever this file lives

    dao = ConversationDao()
    with Session(connection_engine) as session:
        # Create
        conv = Conversation(
            id=..., user_id=..., conversation_name="My Case", last_updated="2025-08-27T12:00:00Z"
        )
        dao.createConversation(session, conv)
        session.commit()  # Commit is the caller's responsibility after create

        # Read by user
        items = dao.fetchConversationByUserId(session, user_id=conv.user_id)

        # Read by name (global)
        items_by_name = dao.fetchConversationByConverastionName(session, "My Case")

        # Read by user + name
        items_by_user_and_name = dao.fetchConversationByUserIdAndConverastionName(
            session, conv.user_id, "My Case"
        )

        # Update timestamp
        dao.updateConversationByDate(session, conversation_id=str(conv.id), timestamp="2025-08-27T12:30:00Z")

        # Update name
        dao.updateConversationByName(session, conversation_id=str(conv.id), conversation_name="Renamed Case")

Error Handling
--------------
- Methods catch generic `Exception`, log/print the error message, and re-raise.
- `update*` methods use `.one()`, which raises `NoResultFound` or `MultipleResultsFound`
  if the target row count is not exactly one. Callers should be ready to handle these.

"""

from sqlalchemy.orm import Session
from backend.database.entities.conversations import Conversation
from uuid import UUID
from sqlalchemy import desc

class ConversationDao:
    """
    Data Access Object (DAO) for managing Conversation entities.
    Provides CRUD operations on the `Conversation` table.
    """

    def createConversation(self, session: Session, conversation: Conversation):
        """
        Create a new conversation record.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation : Conversation
            Conversation entity instance to be added.

        Raises
        ------
        Exception
            If the conversation cannot be created.
        """
        try:
            session.add(conversation)
        except Exception as e:
            print(f"Error in ConversationDao.createConversation. Error: {e}")
            raise e

    def fetchConversationByUserId(self, session: Session, user_id: UUID):
        """
        Fetch all conversations belonging to a specific user,
        ordered by most recently updated.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        user_id : UUID
            Unique identifier of the user.

        Returns
        -------
        list[Conversation]
            List of conversations for the given user.

        Raises
        ------
        Exception
            If the query fails.
        """
        try:
            conversations = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(desc(Conversation.last_updated))
                .all()
            )
            return conversations
        except Exception as e:
            print(f"Error in ConversationDao.fetchConversationByUserId. Error: {e}")
            raise e

    def fetchConversationByConverastionName(self, session: Session, conversation_name: str):
        """
        Fetch all conversations with a given conversation name.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation_name : str
            Name of the conversation.

        Returns
        -------
        list[Conversation]
            List of matching conversations.

        Raises
        ------
        Exception
            If the query fails.
        """
        try:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_name == conversation_name)
                .all()
            )
            return conversation
        except Exception as e:
            print(f"Error in ConversationDao.fetchConversationByConverastionName. Error: {e}")
            raise e

    def fetchConversationByUserIdAndConverastionName(
        self, session: Session, user_id: UUID, conversation_name: str
    ):
        """
        Fetch conversations for a user by both user ID and conversation name.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        user_id : UUID
            Unique identifier of the user.
        conversation_name : str
            Name of the conversation.

        Returns
        -------
        list[Conversation]
            List of matching conversations.

        Raises
        ------
        Exception
            If the query fails.
        """
        try:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .filter(Conversation.conversation_name == conversation_name)
                .all()
            )
            return conversation
        except Exception as e:
            print(f"Error in ConversationDao.fetchConversationByUserIdAndConverastionName. Error: {e}")
            raise e

    def updateConversationByDate(self, session: Session, conversation_id: str, timestamp: str):
        """
        Update the last updated timestamp of a conversation.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation_id : str
            Unique identifier of the conversation.
        timestamp : str
            New timestamp string to set (ISO format recommended).

        Raises
        ------
        Exception
            If the update fails.
        """
        try:
            conversation = (
                session.query(Conversation).filter(Conversation.id == conversation_id).one()
            )
            conversation.last_updated = timestamp
            session.commit()
        except Exception as e:
            print(f"Error in ConversationDao.updateConversationByDate. Error: {e}")
            raise e

    def updateConversationByName(self, session: Session, conversation_id: str, conversation_name: str):
        """
        Update the name of a conversation.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        conversation_id : str
            Unique identifier of the conversation.
        conversation_name : str
            New conversation name to assign.

        Raises
        ------
        Exception
            If the update fails.
        """
        try:
            conversation = (
                session.query(Conversation).filter(Conversation.id == conversation_id).one()
            )
            conversation.conversation_name = conversation_name
            session.commit()
        except Exception as e:
            print(f"Error in ConversationDao.updateConversationByName. Error: {e}")
            raise e
        
    def fetchConversationByConversationType(
        self, session: Session, user_id: UUID, conversation_type:str
    ):
        """
        Fetch conversations for a user by both user ID and conversation type.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        user_id : UUID
            Unique identifier of the user.
        conversation_type : str
            Type of the conversation.

        Returns
        -------
        list[Conversation]
            List of matching conversations.

        Raises
        ------
        Exception
            If the query fails.
        """
        try:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .filter(Conversation.conversation_type == conversation_type)
                .all()
            )
            return conversation
        except Exception as e:
            print(f"Error in ConversationDao.fetchConversationByConversationType. Error: {e}")
            raise e
