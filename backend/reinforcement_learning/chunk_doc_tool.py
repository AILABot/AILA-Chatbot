"""
Chunker implementations for splitting texts into semantically meaningful pieces.

This module defines a set of `BaseChunker` implementations used to slice text
by sentences, characters, tokens, recursive strategies, clustering, Kamradt's
semantic approach, and an LLM-driven semantic chunker.

It is written with Sphinx in mind:
- Docstrings follow **Google style**, compatible with `sphinx.ext.napoleon`.
- Public classes and methods document parameters, returns, and notes.

Example:
    >>> from chunk_doc_tool import SentenceChunker
    >>> chunker = SentenceChunker(sentences_per_chunk=2)
    >>> chunker.split_text("Hello world. This is a test. Another sentence.")
    ['Hello world. This is a test.', 'Another sentence.']

Environment:
    - Uses OpenAI models via API key configured at
      ``backend.database.config.config.settings.API_KEY``.
"""

import tiktoken
import re
from chunking_evaluation import BaseChunker
from typing import List, Callable
from chunking_evaluation.chunking import (
    ClusterSemanticChunker,
    LLMSemanticChunker,
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker,
)
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from backend.database.config.config import settings  # noqa: E402

encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    """Count tokens using the `cl100k_base` encoding.

    Args:
        text: Input string.

    Returns:
        Number of tokens in ``text`` under the `cl100k_base` encoding.
    """
    return len(encoding.encode(text))


class SentenceChunker(BaseChunker):
    """Split text into chunks of N sentences.

    Sentences are detected with a simple regex boundary on ``. ! ?`` followed by
    whitespace. Each chunk concatenates ``sentences_per_chunk`` sentences.

    Args:
        sentences_per_chunk: Number of sentences per returned chunk (default: 3).

    Notes:
        - Validates that each chunk is non-empty.
        - Raises if any chunk would exceed ~8k tokens (safety for typical model limits).
    """

    def __init__(self, sentences_per_chunk: int = 3) -> None:
        self.sentences_per_chunk = sentences_per_chunk

    def split_text(self, text: str) -> List[str]:
        """Split input text into fixed-size sentence groups.

        Args:
            text: The full text to split.

        Returns:
            List of sentence-grouped chunks.

        Raises:
            ValueError: If no valid string chunks can be produced or a chunk
                exceeds the max token limit.
        """
        # Handle the case where the input text is empty
        if not text:
            return []

        # Regex looks for whitespace following ., ! or ? and makes a split
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[str] = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.sentences_per_chunk]).strip()
            chunks.append(chunk)

        valid_chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
        if not valid_chunks:
            raise ValueError("No valid string chunks to embed.")

        MAX_TOKENS = 8191
        for c in valid_chunks:
            if num_tokens(c) > MAX_TOKENS:
                raise ValueError("Bigger than max tokens.")

        return valid_chunks


class CharacterChunker(BaseChunker):
    """Split text into fixed-size character windows (with optional overlap).

    Args:
        characters_per_chunk: Target characters per chunk (default: 1000).
        overlap: Characters to overlap between consecutive chunks (default: 0).

    Notes:
        - Works on raw character counts; does not preserve sentence boundaries.
    """

    def __init__(self, characters_per_chunk: int = 1000, overlap: int = 0) -> None:
        self.characters_per_chunk = characters_per_chunk
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """Split text by character length with overlap.

        Args:
            text: The full text to split.

        Returns:
            List of character-window chunks.
        """
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        step = max(1, self.characters_per_chunk - self.overlap)

        while start < len(text):
            end = min(start + self.characters_per_chunk, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += step

        return chunks


class TokenChunker(BaseChunker):
    """Split text into fixed-size token windows using `FixedTokenChunker`.

    Args:
        tokens_per_chunk: Target tokens per chunk (default: 1000).
        overlap: Overlapping tokens between chunks (default: 0).
        encoding: tiktoken encoding name (default: ``'cl100k_base'``).
    """

    def __init__(self, tokens_per_chunk: int = 1000, overlap: int = 0, encoding: str = "cl100k_base") -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap
        self.encoding = encoding

    def split_text(self, text: str) -> List[str]:
        """Split text into token-sized chunks.

        Args:
            text: Input text.

        Returns:
            List of token-based chunks.
        """
        fixed_token_chunker = FixedTokenChunker(
            chunk_size=self.tokens_per_chunk,
            chunk_overlap=self.overlap,
            encoding_name=self.encoding,
        )
        return fixed_token_chunker.split_text(text)


class RecursiveCharacterChunker(BaseChunker):
    """Recursively split text on structured separators (paragraphs → lines → sentences).

    Uses ``langchain.text_splitter.RecursiveCharacterTextSplitter`` with a
    separator cascade designed from research/common practice.

    Args:
        characters_per_chunk: Target characters per chunk (default: 1000).
        overlap: Character overlap between chunks (default: 0).
    """

    def __init__(self, characters_per_chunk: int = 1000, overlap: int = 0) -> None:
        self.characters_per_chunk = characters_per_chunk
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """Split text using recursive character boundaries.

        Args:
            text: Input text.

        Returns:
            List of chunks respecting structural separators.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.characters_per_chunk,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],  # According to research/common defaults
        )
        return splitter.split_text(text)


class ResTokenChunker(BaseChunker):
    """Recursively split text into token windows using separator cascade.

    Wraps ``RecursiveTokenChunker`` with sensible defaults.

    Args:
        tokens_per_chunk: Target tokens per chunk (default: 1000).
        overlap: Token overlap between chunks (default: 0).
        encoding: tiktoken encoding name (unused by the underlying splitter but kept for parity).
    """

    def __init__(self, tokens_per_chunk: int = 1000, overlap: int = 0, encoding: str = "cl100k_base") -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap
        self.encoding = encoding

    def split_text(self, text: str) -> List[str]:
        """Split text into token-sized chunks with recursive logic.

        Args:
            text: Input text.

        Returns:
            List of token-based chunks.
        """
        splitter = RecursiveTokenChunker(
            chunk_size=self.tokens_per_chunk,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return splitter.split_text(text)


class KamradtChunker(BaseChunker):
    """Semantic chunker using a modified Kamradt algorithm.

    Combines an embedding function and a length function (token counter) to
    produce semantically cohesive chunks around a target length distribution.

    Args:
        avg_chunk_size: Desired average chunk size in tokens (default: 500).
        min_chunk_size: Minimum chunk size in tokens (default: 50).
        model: tiktoken encoding name for token counting (default: ``'cl100k_base'``).
        embedding_model: OpenAI embedding model name (default: ``'text-embedding-3-small'``).

    Notes:
        - Requires a valid OpenAI API key in ``settings.API_KEY``.
    """

    def __init__(
        self,
        avg_chunk_size: int = 500,
        min_chunk_size: int = 50,
        model: str = "cl100k_base",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.avg_chunk_size = avg_chunk_size
        self.min_chunk_size = min_chunk_size
        self.model = model
        self.embedding_function = OpenAIEmbeddingFunction(api_key=settings.API_KEY, model_name=embedding_model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the configured tiktoken encoding.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        encoder = tiktoken.get_encoding(self.model)
        return len(encoder.encode(text))

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically cohesive chunks around a target size.

        Args:
            text: Input text.

        Returns:
            List of semantic chunks.
        """
        splitter = KamradtModifiedChunker(
            avg_chunk_size=self.avg_chunk_size,
            min_chunk_size=self.min_chunk_size,
            embedding_function=self.embedding_function,
            length_function=self.count_tokens,
        )
        return splitter.split_text(text)


class ClusterChunker(BaseChunker):
    """Cluster-based semantic chunker.

    Uses embeddings + a maximum chunk size constraint to cluster contiguous
    segments and then emit chunks under a token budget.

    Args:
        embedding_model: OpenAI embedding model name (default: ``'text-embedding-3-small'``).
        chunk_size: Maximum tokens per chunk (default: 500).
        overlap: (Reserved for future use.) Overlap tokens between chunks (default: 0).

    Notes:
        - Requires a valid OpenAI API key in ``settings.API_KEY``.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small", chunk_size: int = 500, overlap: int = 0) -> None:
        self.embedding_function = OpenAIEmbeddingFunction(api_key=settings.API_KEY, model_name=embedding_model)
        self.max_chunk_size = chunk_size

    def openai_token_count(self, string: str) -> int:
        """Return token count for a string with `cl100k_base`.

        Args:
            string: Input text.

        Returns:
            Token length of ``string``.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string, disallowed_special=()))

    def split_text(self, text: str) -> List[str]:
        """Split text using clustering over embeddings and a token budget.

        Args:
            text: Input text.

        Returns:
            List of semantic chunks under the token budget.
        """
        cluster_chunker = ClusterSemanticChunker(
            embedding_function=self.embedding_function,
            max_chunk_size=self.max_chunk_size,
            length_function=self.openai_token_count,
        )
        return cluster_chunker.split_text(text)


class LLMChunker(BaseChunker):
    """LLM-driven semantic chunker.

    Delegates segmentation to an LLM that is instructed to split text into
    semantically coherent units.

    Args:
        model: Model name to use (default: ``'gpt-3.5-turbo'``). Actual splitter
            below uses ``'gpt-4o'`` by default.
    """

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.model = model

    def split_text(self, text: str) -> List[str]:
        """Split text via an LLM-based semantic chunker.

        Args:
            text: Input text.

        Returns:
            List of LLM-derived chunks.

        Notes:
            - Uses `LLMSemanticChunker` with ``organisation='openai'`` and
              ``model_name='gpt-4o'``; API key from ``settings.API_KEY``.
        """
        llm_chunker = LLMSemanticChunker(organisation="openai", model_name="gpt-4o", api_key=settings.API_KEY)
        return llm_chunker.split_text(text)
