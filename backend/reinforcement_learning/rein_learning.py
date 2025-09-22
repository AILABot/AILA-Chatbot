"""
RL pipeline orchestrator for:
- Fetching curated feedback samples from the database
- Formatting them for model fine-tuning and indexing
- Fine-tuning embeddings and cross-encoders
- Updating vector indexes (phishing, GDPR, Greek Penal Code, and law cases)

Sphinx:
    Docstrings follow **Google style** for `sphinx.ext.napoleon`.
"""

from chunk_docs import Document_Chunker
from backend.database.daos.document_feedback_dao import DocumentFeedbackDao
from backend.database.entities.document_feedback import DocumentFeedback
from backend.database.daos.user_message_dao import UserMessagesDao
from backend.database.helpers.transactionManagement import transactional
from backend.reinforcement_learning.embedding_training import EmbeddingFinetuning
from backend.reinforcement_learning.cross_encoder_finetuning import CrossEncoderFinetuning
from backend.reinforcement_learning.index_creation_update import index_creation, index_update
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import os


def get_docs(session: Session) -> List[DocumentFeedback]:
    """Fetch all `DocumentFeedback` rows to serve as training/indexing sources.

    Args:
        session: SQLAlchemy session.

    Returns:
        List of `DocumentFeedback` entities.
    """
    doc_feedback_dao = DocumentFeedbackDao()
    return doc_feedback_dao.fetchDocs(session)


def get_message(session: Session, id: str) -> Optional[str]:
    """Resolve a stored message's text by its ID.

    Args:
        session: SQLAlchemy session.
        id: Message row identifier.

    Returns:
        The message text if found; otherwise `None`.
    """
    user_message_dao = UserMessagesDao()
    message = user_message_dao.fetchMessageById(session, id)
    if len(message) > 0:
        return message[0].message_text
    return None


def format_docs(session: Session, docs: List[DocumentFeedback]) -> List[Dict[str, Any]]:
    """Transform DB feedback rows into the normalized training/index format.

    Produces items with keys:
        - thematology (theme)
        - title (document title)
        - doc (raw text)
        - example: { query, answer: [str], negative_answer: [str] }

    Args:
        session: SQLAlchemy session for resolving message texts.
        docs: Feedback rows.

    Returns:
        List of normalized dictionaries suitable for training & indexing.
    """
    new_documents: List[Dict[str, Any]] = []
    for doc in docs:
        query = get_message(session, doc.query_id)
        answer = doc.context
        negative = get_message(session, doc.negative_answer_id)

        # Ensure lists for downstream consumers expecting list[str]
        answers_list = [a for a in [answer] if a is not None]
        negatives_list = [negative] if isinstance(negative, str) else ([] if negative is None else list(negative))

        example = {
            "query": query,
            "answer": answers_list,
            "negative_answer": negatives_list,
        }

        new_documents.append(
            {
                "thematology": doc.theme,
                "title": doc.doc_name,
                "doc": doc.doc_text,
                "example": example,
            }
        )
    return new_documents


def unique_dicts(list_of_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """De-duplicate a list of (JSON-serializable) dictionaries.

    Args:
        list_of_dicts: Input list (must be JSON serializable).

    Returns:
        New list with duplicates removed, preserving first occurrence.
    """
    seen = set()
    unique: List[Dict[str, Any]] = []
    for d in list_of_dicts:
        j = json.dumps(d, sort_keys=True)
        if j not in seen:
            seen.add(j)
            unique.append(d)
    return unique


if __name__ == "__main__":

    @transactional
    def test(session: Session) -> None:
        """End-to-end pipeline:
            1) Load feedback rows
            2) Format + dedupe
            3) Fine-tune embeddings (Matryoshka) on all configured models
            4) Fine-tune cross-encoder (reranker)
            5) Chunk new docs and update indexes by theme

        Args:
            session: Injected SQLAlchemy session via `@transactional`.
        """
        input_paths = {
            "phishing": "backend/files/en/Phishing Scenarios",
            "gdpr": "backend/files/en/General Data Protection Regulation",
            "cybercrime": "backend/files/en/Greek Cybercrime Legislation",
            "cases": "backend/files/en/Law Cases",
        }

        # 1) Fetch
        docs = get_docs(session)

        # 2) Normalize + dedupe
        new_docs = format_docs(session, docs)
        unique_documents = unique_dicts(new_docs)

        # 3) Finetune embeddings (Matryoshka)
        embedding_training = EmbeddingFinetuning()
        models = embedding_training.models
        embedding_training.model_finetuning(unique_documents, models)

        # 4) Finetune cross-encoder (reranker)
        cross_encoder_finetuning = CrossEncoderFinetuning()
        cross_encoder_finetuning.cross_encoder_tuning(unique_documents)

        # 5) Index insertion/update only for files not present on disk
        docs_dict = {theme: [] for theme in input_paths.keys()}
        for doc in unique_documents:
            theme = doc["thematology"]
            docs_dict[theme].append(doc)

        chunk_docs = Document_Chunker()
        for theme in input_paths:
            chunks = []
            files = []
            for doc in docs_dict[theme]:
                # Only index docs not already present by title filename
                if doc["title"] not in os.listdir(input_paths[theme]):
                    files.append(doc)
                else:
                    print(f"Document '{doc['title']}' already present; skipping.")
            print(theme, len(files))
            chunks += chunk_docs.chunk_creation(theme, files)
            if len(chunks) != 0:
                index_update(theme, chunks)

    test()
