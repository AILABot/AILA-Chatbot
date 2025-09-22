"""
Document chunking utilities for phishing scenarios, GDPR, Greek court decisions,
and Greek cybercrime legislation.

This module is Sphinx-friendly: all public objects include Google-style
docstrings (for `sphinx.ext.napoleon`) so your API reference can be generated
directly from the source.

Overview
--------
- ``Document_Chunker`` orchestrates per-theme parsing using a provided
  ``BaseChunker`` (sentence/token/character-based variants).
- Each parser returns normalized chunk dicts:
  ``{"id": str, "content": str, "metadata": {...}}``.

Expected input
--------------
``files`` is a list of dicts with at least:
    - ``"title"``: str — document title/identifier
    - ``"doc"``:   str — full text content to be chunked
"""

import os
from chunking_evaluation import BaseChunker
import re
from typing import List, Dict, Any, Union
import chunk_doc_tool


class Document_Chunker:
    """Chunk builder for multiple legal/infosec document themes.

    This class provides specialized parsers that:
      1) split raw text into smaller units using a supplied ``BaseChunker`` and
      2) attach consistent metadata for downstream retrieval/evaluation.

    Notes:
        - All parsers expect each entry of ``files`` to contain keys
          ``\"title\"`` and ``\"doc\"``.
        - Returned chunk dicts are shaped for vector DB ingestion and evaluation.
    """

    def __init__(self) -> None:
        """Create a new ``Document_Chunker``."""
        pass

    def parse_phishing(self, files: List[Dict[str, Any]], chunker: BaseChunker) -> List[Dict[str, Any]]:
        """Parse and chunk **phishing scenario** explainers.

        Args:
            files: List of items with keys:
                - ``title``: scenario name (e.g., ``"Smishing"``, ``"Vishing"``).
                - ``doc``:   scenario explainer text.
            chunker: A chunker implementing ``split_text(text: str) -> List[str]``.

        Returns:
            List of chunk dicts. Example item:
            ``{
                "id": "phishing_0",
                "content": "...",
                "metadata": {
                    "source": "Phishing Scenarios",
                    "doc_type": "explainer",
                    "title": "<attack_type>",
                    "lang": "en"
                }
            }``
        """
        chunks: List[Dict[str, Any]] = []
        counter = 0
        for file in files:
            attack_type = file["title"]
            text_chunks = chunker.split_text(file["doc"])
            for chunk in text_chunks:
                chunks.append(
                    {
                        "id": f"phishing_{counter}",
                        "content": chunk,
                        "metadata": {
                            "source": "Phishing Scenarios",
                            "doc_type": "explainer",
                            "title": attack_type,
                            "lang": "en",
                        },
                    }
                )
                counter += 1
        return chunks

    def parse_gdpr(self, files: List[Dict[str, Any]], chunker: BaseChunker) -> List[Dict[str, Any]]:
        """Parse and chunk **GDPR** regulatory text.

        Args:
            files: List with keys:
                - ``title``: GDPR section/part title.
                - ``doc``:   raw text content.
            chunker: Text splitter.

        Returns:
            List of chunk dicts with ``source="GDPR"`` and ``doc_type="regulation"``.
        """
        chunks: List[Dict[str, Any]] = []
        counter = 0
        for file in files:
            title = file["title"]
            text_chunks = chunker.split_text(file["doc"])
            for chunk in text_chunks:
                chunks.append(
                    {
                        "id": f"gdpr_{counter}",
                        "content": chunk,
                        "metadata": {
                            "source": "GDPR",
                            "doc_type": "regulation",
                            "title": title,
                            "lang": "en",
                        },
                    }
                )
                counter += 1
        return chunks

    def parse_law_cases(self, files: List[Dict[str, Any]], chunker: BaseChunker) -> List[Dict[str, Any]]:
        """Parse and chunk **Greek court decisions** (English summaries).

        The function extracts lightweight metadata via regex from the case text:
            - ``Decision number: <value>``
            - ``Court (Civil/Criminal): <value>``
            - ``Outcome (innocent, guilty): <value>``
            - ``Law <n>/<yyyy>`` or ``Article <n>[A-Z]? (of Law <n>/<yyyy>)?`` occurrences

        Args:
            files: List with keys:
                - ``title``: filename/title label for the decision.
                - ``doc``:   full decision text.
            chunker: Text splitter.

        Returns:
            List of chunk dicts with metadata:
            ``{
              "source": "Greek Court Decisions",
              "doc_type": "case_law",
              "jurisdiction": "GR",
              "case_id": str,
              "civil_or_criminal": "civil" | "criminal" | "unknown",
              "outcome": "innocent" | "guilty" | "unknown",
              "relevant_laws": [ ... ],
              "title": str,
              "lang": "en"
            }``
        """
        chunks: List[Dict[str, Any]] = []
        counter = 0
        case_id_ = 0
        for file in files:
            case = file["doc"]
            match = re.search(r"Decision number:\s*(.*?)\n", case)
            case_id = match.group(1).strip() if match else f"case_{case_id_}"
            court = re.search(r"Court \(Civil/Criminal\):\s*(.*?)\n", case)
            court_type = court.group(1).strip().lower() if court else "unknown"
            outcome = re.search(r"Outcome \(innocent, guilty\):\s*(.*?)\n", case)
            laws = re.findall(r"Law\s+\d+/\d+|Article\s+\d+[A-Z]?(\s+of\s+Law\s+\d+/\d+)?", case)

            # Prefer provided title; fall back gracefully.
            title = file.get("title", f"case_{case_id_}")

            text_chunks = chunker.split_text(case)
            for chunk in text_chunks:
                chunks.append(
                    {
                        "id": f"case_{counter}",
                        "content": chunk,
                        "metadata": {
                            "title": title,
                            "source": "Greek Court Decisions",
                            "doc_type": "case_law",
                            "jurisdiction": "GR",
                            "case_id": case_id,
                            "civil_or_criminal": court_type,
                            "outcome": outcome.group(1).strip() if outcome else "unknown",
                            "relevant_laws": list(set(laws)),
                            "lang": "en",
                        },
                    }
                )
                counter += 1
            case_id_ += 1
        return chunks

    def parse_cybercrime(self, files: List[Dict[str, Any]], chunker: BaseChunker) -> List[Dict[str, Any]]:
        """Parse and chunk **Greek cybercrime law** excerpts.

        The function derives metadata from the document title using regex:
            - ``Article <number>[A-Z]?`` → ``article_number``
            - ``Ν/Π/Κ`` codes like ``Ν. 4411/2016`` → ``law``

        Args:
            files: List with keys:
                - ``title``: statute/article heading containing identifiers.
                - ``doc``:   statute text.
            chunker: Text splitter.

        Returns:
            List of chunk dicts with ``source="Greek Cybercrime Law"`` and
            ``doc_type="criminal_statute"``.
        """
        chunks: List[Dict[str, Any]] = []
        counter = 0
        for file in files:
            title = file["title"]
            article_id = re.findall(r"Article\s+(\d+[A-Z]?)", title)
            law_id = re.findall(r"[ΝΠΚ]\.?\s?\d+/?\d*", title)
            text_chunks = chunker.split_text(file["doc"])
            for chunk in text_chunks:
                chunks.append(
                    {
                        "id": f"cybercrime_{counter}",
                        "content": chunk,
                        "metadata": {
                            "title": title,
                            "source": "Greek Cybercrime Law",
                            "doc_type": "criminal_statute",
                            "law": law_id[0] if law_id else "unknown",
                            "article_number": article_id[0] if article_id else str(counter),
                            "lang": "en",
                            "jurisdiction": "GR",
                        },
                    }
                )
                counter += 1
        return chunks

    def chunk_creation(self, theme: str, files: List[Dict[str, Any]]) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Create chunks for a given **theme** using preset chunkers.

        Themes & outputs
        ----------------
        - ``"phishing"`` → **single** list from :meth:`parse_phishing`.
        - ``"cybercrime"`` → **pair** ``[recall_chunks, precision_chunks]`` using:
            - recall:   ``SentenceChunker(sentences_per_chunk=20)``
            - precision:``ResTokenChunker(tokens_per_chunk=200, overlap=20)``
        - ``"gdpr"`` → **pair** ``[recall_chunks, precision_chunks]`` using:
            - recall:   ``SentenceChunker(sentences_per_chunk=20)``
            - precision:``ResTokenChunker(tokens_per_chunk=200, overlap=20)``
        - ``"cases"`` → **pair** ``[recall_chunks, precision_chunks]`` using:
            - recall:   ``TokenChunker(tokens_per_chunk=1000, overlap=100)``
            - precision:``RecursiveCharacterChunker(characters_per_chunk=100, overlap=0)``

        Args:
            theme: One of ``{"phishing","cybercrime","gdpr","cases"}``.
            files: List of input records with ``title`` and ``doc``.

        Returns:
            Either:
                - a **list** of chunks (for ``phishing``), or
                - a **list of two lists** ``[recall_chunks, precision_chunks]`` (for others).

        Notes:
            - This method wires concrete chunker configs; swap them here to
              tune recall/precision trade-offs globally.
        """
        chunks: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]] = []
        if theme == "phishing":
            chunks = self.parse_phishing(files, chunk_doc_tool.SentenceChunker(sentences_per_chunk=1))
        if theme == "cybercrime":
            gpc_chunks_recall = self.parse_cybercrime(files, chunk_doc_tool.SentenceChunker(sentences_per_chunk=20))
            gpc_chunks_precision = self.parse_cybercrime(files, chunk_doc_tool.ResTokenChunker(tokens_per_chunk=200, overlap=20))
            chunks = [gpc_chunks_recall, gpc_chunks_precision]
        if theme == "gdpr":
            gdpr_chunks_recall = self.parse_gdpr(files, chunk_doc_tool.SentenceChunker(sentences_per_chunk=20))
            gdpr_chunks_precision = self.parse_gdpr(files, chunk_doc_tool.ResTokenChunker(tokens_per_chunk=200, overlap=20))
            chunks = [gdpr_chunks_recall, gdpr_chunks_precision]
        if theme == "cases":
            law_cases_chunks_recall = self.parse_law_cases(files, chunk_doc_tool.TokenChunker(tokens_per_chunk=1000, overlap=100))
            law_cases_chunks_precision = self.parse_law_cases(files, chunk_doc_tool.RecursiveCharacterChunker(characters_per_chunk=100, overlap=0))
            chunks = [law_cases_chunks_recall, law_cases_chunks_precision]
        return chunks if len(chunks) != 0 else []
