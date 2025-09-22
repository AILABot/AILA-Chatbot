"""
Reinforcement Learning (RL) pipeline — fine-tuning, chunking, and indexing.

This package bundles:
- Embedding fine-tuning with Matryoshka evaluation.
- Cross-encoder (re-ranker) fine-tuning with hard negatives.
- Document chunking utilities (sentence/char/token/semantic/LLM).
- Vector index creation & incremental updates (phishing, GDPR, Greek Penal Code, law cases).

Modules in this package
-----------------------
- ``chunk_doc_tool.py``       : Chunker implementations (Sentence/Character/Token/Recursive/Kamradt/Cluster/LLM).
- ``chunk_docs.py``           : High-level :class:`Document_Chunker` to build themed chunks.
- ``embedding_training.py``   : :class:`EmbeddingFinetuning` — train & eval embedding models.
- ``cross_encoder_finetuning.py`` : :class:`CrossEncoderFinetuning` — train re-ranker with hard negatives.
- ``index_creation_update.py``: Index builders/updaters for phishing/GDPR/GPC/cases.
- ``rein_learning.py``        : Orchestration script (entrypoint) for the full pipeline.
- ``new_queries.csv``         : Base QA pairs used for training/evaluation.

Quick start
-----------
You can import the core API right from the package:

    >>> from backend.reinforcement_learning import (
    ...     EmbeddingFinetuning, CrossEncoderFinetuning, Document_Chunker,
    ...     SentenceChunker, CharacterChunker, TokenChunker,
    ...     RecursiveCharacterChunker, ResTokenChunker, KamradtChunker, ClusterChunker, LLMChunker,
    ...     index_creation, index_update,
    ...     phishing_index_creation, phishing_index_update,
    ...     greek_penal_code_index_creation, greek_penal_code_index_update,
    ...     gdpr_index_creation, gdpr_index_update,
    ...     law_cases_index_creation, law_cases_index_update,
    ... )

Sphinx notes
------------
- Docstrings are Google-style (compatible with ``sphinx.ext.napoleon``).
- ``__all__`` exposes the public API for tidy ``autosummary`` pages.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   EmbeddingFinetuning
   CrossEncoderFinetuning
   Document_Chunker
   SentenceChunker
   CharacterChunker
   TokenChunker
   RecursiveCharacterChunker
   ResTokenChunker
   KamradtChunker
   ClusterChunker
   LLMChunker
   index_creation
   index_update
   phishing_index_creation
   phishing_index_update
   greek_penal_code_index_creation
   greek_penal_code_index_update
   gdpr_index_creation
   gdpr_index_update
   law_cases_index_creation
   law_cases_index_update
"""