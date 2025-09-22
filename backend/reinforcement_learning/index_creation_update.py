"""
Vector index creation & update utilities for phishing scenarios, Greek Penal Code,
GDPR, and Greek law cases using LlamaIndex + HuggingFace embeddings.

This module is Sphinx-friendly: all public functions include Google-style
docstrings (for `sphinx.ext.napoleon`) so your API docs are generated
directly from source.

Conventions
-----------
- Each `chunk` is a dict:
    {
        "id": str,
        "content": str,
        "metadata": dict
    }
- Indexes are persisted under `backend/vector_indexes/*`.
- Embeddings use HuggingFace models via LangChain's `HuggingFaceEmbeddings`.
"""

from typing import List
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from langchain.embeddings import HuggingFaceEmbeddings


def phishing_index_creation(chunks: List[dict]) -> None:
    """Create (or overwrite) the phishing scenarios index.

    Args:
        chunks: List of chunk dicts with keys ``id``, ``content``, ``metadata``.

    Side Effects:
        Persists the index to
        ``backend/vector_indexes/phishing_index_documents_trained_embedding``.
    """
    phishing_documents = []
    for chunk in chunks:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        phishing_documents.append(Document(text=text, metadata=metadata))

    phishing_index = VectorStoreIndex.from_documents(
        documents=phishing_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    phishing_index.storage_context.persist(
        "backend/vector_indexes/phishing_index_documents_trained_embedding"
    )


def phishing_index_update(chunks: List[dict]) -> None:
    """Append new phishing chunks to an existing index (rebuilds & persists).

    Args:
        chunks: New phishing chunks to add.

    Behavior:
        - Loads existing index & docstore.
        - Merges with new documents.
        - Rebuilds a fresh `VectorStoreIndex` and persists.

    No-op if ``chunks`` is empty.
    """
    storage_context = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/phishing_index_documents_trained_embedding"
    )

    phishing_index = load_index_from_storage(
        storage_context,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in phishing_index.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    new_phishing_documents = []
    for chunk in chunks:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        new_phishing_documents.append(Document(text=chunk["content"], metadata=metadata))

    if len(new_phishing_documents) == 0:
        return

    all_docs = existing_docs + new_phishing_documents

    phishing_index = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    phishing_index.storage_context.persist(
        "backend/vector_indexes/phishing_index_documents_trained_embedding"
    )


def greek_penal_code_index_creation(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Create indexes for Greek Penal Code (recall+precision variants).

    Args:
        chunks_recall: Larger/softer chunks to favor recall.
        chunks_precision: Smaller/tighter chunks to favor precision.

    Side Effects:
        Persists to:
            - ``gpc_recall_index_documents_recall_trained_embedding``  (recall)
            - ``gpc_recall_index_documents_precision_trained_embedding`` (precision)
    """
    # Recall
    gpc_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gpc_recall_documents.append(Document(text=text, metadata=metadata))

    gpc_index_recall = VectorStoreIndex.from_documents(
        documents=gpc_recall_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/legal-bert-base-uncased-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gpc_index_recall.storage_context.persist(
        "backend/vector_indexes/gpc_recall_index_documents_recall_trained_embedding"
    )

    # Precision
    gpc_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gpc_precision_documents.append(Document(text=text, metadata=metadata))

    gpc_index_precision = VectorStoreIndex.from_documents(
        documents=gpc_precision_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gpc_index_precision.storage_context.persist(
        "backend/vector_indexes/gpc_recall_index_documents_precision_trained_embedding"
    )


def greek_penal_code_index_update(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Update (append) Greek Penal Code indexes (recall+precision).

    Args:
        chunks_recall: New recall-oriented chunks.
        chunks_precision: New precision-oriented chunks.

    Notes:
        - Embedding models match those used at creation.
    """
    # ---- Recall
    storage_context_recall = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/gpc_recall_index_documents_recall_trained_embedding"
    )

    gpc_index_recall_retriever = load_index_from_storage(
        storage_context_recall,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/legal-bert-base-uncased-legal-matryoshka",  # match creation
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in gpc_index_recall_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    gpc_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gpc_recall_documents.append(Document(text=text, metadata=metadata))

    if len(gpc_recall_documents) == 0:
        pass
    else:
        all_docs = existing_docs + gpc_recall_documents
        gpc_index_recall_retriever = VectorStoreIndex.from_documents(
            all_docs,
            embed_model=HuggingFaceEmbeddings(
                model_name="IoannisKat1/legal-bert-base-uncased-legal-matryoshka",
                model_kwargs={"trust_remote_code": True},
            ),
        )
        gpc_index_recall_retriever.storage_context.persist(
            "backend/vector_indexes/gpc_recall_index_documents_recall_trained_embedding"
        )

    # ---- Precision
    storage_context_precision = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/gpc_recall_index_documents_precision_trained_embedding"
    )

    gpc_index_precision_retriever = load_index_from_storage(
        storage_context_precision,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",  # match creation
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in gpc_index_precision_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    gpc_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gpc_precision_documents.append(Document(text=text, metadata=metadata))

    if len(gpc_precision_documents) == 0:
        return

    all_docs = existing_docs + gpc_precision_documents

    gpc_index_precision_retriever = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gpc_index_precision_retriever.storage_context.persist(
        "backend/vector_indexes/gpc_recall_index_documents_precision_trained_embedding"
    )


def gdpr_index_creation(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Create indexes for GDPR (recall+precision variants).

    Args:
        chunks_recall: Recall-oriented GDPR chunks.
        chunks_precision: Precision-oriented GDPR chunks.

    Side Effects:
        Persists to:
            - ``gdpr_recall_index_documents_recall_trained_embedding`` (recall)
            - ``gdpr_precision_index_documents_precision_trained_embedding`` (precision)
    """
    gdpr_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gdpr_recall_documents.append(Document(text=text, metadata=metadata))

    gdpr_index_recall = VectorStoreIndex.from_documents(
        documents=gdpr_recall_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gdpr_index_recall.storage_context.persist(
        "backend/vector_indexes/gdpr_recall_index_documents_recall_trained_embedding"
    )

    gdpr_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gdpr_precision_documents.append(Document(text=text, metadata=metadata))

    gdpr_index_precision = VectorStoreIndex.from_documents(
        documents=gdpr_precision_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gdpr_index_precision.storage_context.persist(
        "backend/vector_indexes/gdpr_precision_index_documents_precision_trained_embedding"
    )


def gdpr_index_update(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Update (append) GDPR indexes (recall+precision)."""
    # Recall
    storage_context_recall = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/gdpr_recall_index_documents_recall_trained_embedding"
    )

    gdpr_index_recall_retriever = load_index_from_storage(
        storage_context_recall,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in gdpr_index_recall_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    gdpr_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gdpr_recall_documents.append(Document(text=text, metadata=metadata))

    if len(gdpr_recall_documents) > 0:
        all_docs = existing_docs + gdpr_recall_documents
        gdpr_index_recall_retriever = VectorStoreIndex.from_documents(
            all_docs,
            embed_model=HuggingFaceEmbeddings(
                model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
                model_kwargs={"trust_remote_code": True},
            ),
        )
        gdpr_index_recall_retriever.storage_context.persist(
            "backend/vector_indexes/gdpr_recall_index_documents_recall_trained_embedding"
        )

    # Precision
    storage_context_precision = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/gdpr_precision_index_documents_precision_trained_embedding"
    )

    gdpr_index_precision_retriever = load_index_from_storage(
        storage_context_precision,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in gdpr_index_precision_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    gdpr_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            gdpr_precision_documents.append(Document(text=text, metadata=metadata))

    if len(gdpr_precision_documents) == 0:
        return

    all_docs = existing_docs + gdpr_precision_documents

    gdpr_index_precision_retriever = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/multilingual-e5-large-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    gdpr_index_precision_retriever.storage_context.persist(
        "backend/vector_indexes/gdpr_precision_index_documents_precision_trained_embedding"
    )


def law_cases_index_creation(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Create indexes for Greek law cases (recall+precision variants).

    Args:
        chunks_recall: Recall-oriented case-law chunks.
        chunks_precision: Precision-oriented case-law chunks.

    Side Effects:
        Persists to:
            - ``law_cases_recall_index_documents_recall_trained_embedding`` (recall)
            - ``law_cases_recall_index_documents_precision_trained_embedding`` (precision)
    """
    law_cases_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        law_cases_recall_documents.append(Document(text=text, metadata=metadata))

    law_cases_index_recall = VectorStoreIndex.from_documents(
        documents=law_cases_recall_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    law_cases_index_recall.storage_context.persist(
        "backend/vector_indexes/law_cases_recall_index_documents_recall_trained_embedding"
    )

    law_cases_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            law_cases_precision_documents.append(Document(text=text, metadata=metadata))

    law_cases_index_precision = VectorStoreIndex.from_documents(
        documents=law_cases_precision_documents,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/bge-m3-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    law_cases_index_precision.storage_context.persist(
        "backend/vector_indexes/law_cases_recall_index_documents_precision_trained_embedding"
    )


def law_cases_index_update(chunks_recall: List[dict], chunks_precision: List[dict]) -> None:
    """Update (append) law cases indexes (recall+precision)."""
    # Recall
    storage_context_recall = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/law_cases_recall_index_documents_recall_trained_embedding"
    )

    law_index_recall_retriever = load_index_from_storage(
        storage_context_recall,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in law_index_recall_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    law_cases_recall_documents = []
    for chunk in chunks_recall:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        law_cases_recall_documents.append(Document(text=text, metadata=metadata))

    if len(law_cases_recall_documents) > 0:
        all_docs = existing_docs + law_cases_recall_documents
        law_index_recall_retriever = VectorStoreIndex.from_documents(
            all_docs,
            embed_model=HuggingFaceEmbeddings(
                model_name="IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
                model_kwargs={"trust_remote_code": True},
            ),
        )
        law_index_recall_retriever.storage_context.persist(
            "backend/vector_indexes/law_cases_recall_index_documents_recall_trained_embedding"
        )

    # Precision
    storage_context_precision = StorageContext.from_defaults(
        persist_dir="backend/vector_indexes/law_cases_recall_index_documents_precision_trained_embedding"
    )

    law_index_precision_retriever = load_index_from_storage(
        storage_context_precision,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/bge-m3-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )

    existing_docs = []
    for node in law_index_precision_retriever.docstore.docs.values():
        text = node.text
        metadata = node.metadata or {}
        existing_docs.append(Document(text=text, metadata=metadata))

    law_cases_precision_documents = []
    for chunk in chunks_precision:
        metadata = {"id": chunk["id"], **chunk["metadata"]}
        text = chunk["content"]
        if len(text) > 5:
            law_cases_precision_documents.append(Document(text=text, metadata=metadata))

    if len(law_cases_precision_documents) == 0:
        return

    all_docs = existing_docs + law_cases_precision_documents

    law_index_precision_retriever = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=HuggingFaceEmbeddings(
            model_name="IoannisKat1/bge-m3-legal-matryoshka",
            model_kwargs={"trust_remote_code": True},
        ),
    )
    law_index_precision_retriever.storage_context.persist(
        "backend/vector_indexes/law_cases_recall_index_documents_precision_trained_embedding"
    )


def index_creation(theme: str, chunks) -> None:
    """Dispatch index creation by theme.

    Args:
        theme: One of ``{'phishing','cybercrime','gdpr','cases'}``.
        chunks: For ``'phishing'`` a single list of chunks.
                For others, a pair ``[chunks_recall, chunks_precision]``.
    """
    if theme == "phishing":
        phishing_index_creation(chunks)
    if theme == "cybercrime":
        greek_penal_code_index_creation(chunks[0], chunks[1])
    if theme == "gdpr":
        gdpr_index_creation(chunks[0], chunks[1])
    if theme == "cases":
        law_cases_index_creation(chunks[0], chunks[1])


def index_update(theme: str, chunks) -> None:
    """Dispatch index update by theme.

    Args:
        theme: One of ``{'phishing','cybercrime','gdpr','cases'}``.
        chunks: For ``'phishing'`` a single list of chunks.
                For others, a pair ``[chunks_recall, chunks_precision]``.
    """
    if theme == "phishing":
        phishing_index_update(chunks)
    if theme == "cybercrime":
        greek_penal_code_index_update(chunks[0], chunks[1])
    if theme == "gdpr":
        gdpr_index_update(chunks[0], chunks[1])
    if theme == "cases":
        law_cases_index_update(chunks[0], chunks[1])
