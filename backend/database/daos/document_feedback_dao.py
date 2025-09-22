"""
DocumentFeedback DAO — Create & Fetch
=====================================

Purpose
-------
Thin data-access layer for the DocumentFeedback entity:
- Persist a new feedback record (links a query message to a negatively rated answer,
  with captured doc text/context/theme).
- Fetch all feedback records.

Transaction Model
-----------------
- This DAO **adds** objects to the SQLAlchemy session but does **not** call `commit()`.
  The caller controls transactions (commit/rollback) and session lifecycle.

Error Handling
--------------
- Operational errors are logged and re-raised for the caller to handle.
"""

from sqlalchemy.orm import Session
from backend.database.entities.document_feedback import DocumentFeedback
import uuid

class DocumentFeedbackDao:
    """
    Data Access Object for `DocumentFeedback`.

    Responsibilities:
        - Add a new `DocumentFeedback` instance to the active session.
        - Retrieve all `DocumentFeedback` rows.

    Notes:
        - Session management (commit/rollback/close) is delegated to the caller.
        - Consider adding filtered queries (by `query_id`, `negative_answer_id`, date range)
          and pagination in future iterations.
    """

    def createDocument(self, session: Session, document_feedback: DocumentFeedback) -> bool:
        """
        Add a new `DocumentFeedback` record to the session.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session (transaction boundary controlled by the caller).
        document_feedback : DocumentFeedback
            The `DocumentFeedback` entity to persist.

        Returns
        -------
        bool
            True if the object was added to the session successfully.
            (A subsequent `session.commit()` is required to persist it.)

        Raises
        ------
        Exception
            Propagates any SQLAlchemy or application error encountered during `session.add`.
        """
        try:
            session.add(document_feedback)
            return True
        except Exception as e:
            print(f"Error in DocumentFeedbackDao.createDocument. Error Message: {e}")
            raise e

    def fetchDocs(self, session: Session):
        """
        Fetch all `DocumentFeedback` records.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.

        Returns
        -------
        list[DocumentFeedback]
            A list containing every feedback row in `document_feedback`.

        Raises
        ------
        Exception
            Propagates any SQLAlchemy or application error encountered during the query.
        """
        try:
            all_docs = session.query(DocumentFeedback).all()
            return all_docs
        except Exception as e:
            print(f"Error in DocumentFeedbackDao.fetchDocs. Error Message: {e}")
            raise e
