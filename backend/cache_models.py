"""
Model Caching Script

Caches local copies of embedding and reranker models so they can be loaded
from disk at runtime (faster startup, no repeated network downloads).

What it does
------------
- For each ID in EMBEDDING_MODELS, downloads a SentenceTransformer and saves it
  under ./backend/cached_embedding_models/<sanitized_id>.
- For each ID in RERANKER_MODELS, downloads a CrossEncoder reranker and saves it
  under ./backend/cached_reranker_models/<sanitized_id>.

Notes
-----
- `trust_remote_code=True` is enabled for SentenceTransformer to allow custom
  model code from the repository. Only use with sources you trust.
- The imports `AutoTokenizer` and `AutoModel` from `transformers` are not used
  in this script and may be removed unless you rely on them elsewhere.
"""

from pathlib import Path
from typing import Iterable

from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

# -----------------------
# Configuration constants
# -----------------------

# List of SentenceTransformer embedding model IDs to cache.
EMBEDDING_MODELS = [
    "IoannisKat1/multilingual-e5-large-legal-matryoshka",
    "IoannisKat1/modernbert-embed-base-legal-matryoshka-2",
    "IoannisKat1/bge-m3-legal-matryoshka",
    "IoannisKat1/legal-bert-base-uncased-legal-matryoshka",
]
"""List of embedding models to download"""


# List of CrossEncoder reranker model IDs to cache.
RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
    "IoannisKat1/bge-reranker-basefinetuned-new"
]
"""List of reranker models to download"""

# Base output directories for cached models.
EMBEDDINGS_DIR = Path("./backend/cached_embedding_models")
"""Directory in which the embedding models will be cached for future use."""
RERANKERS_DIR = Path("./backend/cached_reranker_models")
"""Directory in which the reranker models will be cached for future use."""

def sanitize_model_id(model_id: str) -> str:
    """
    Convert a model ID to a filesystem-friendly name.

    Parameters
    ----------
    model_id : str
        The model repository ID (e.g., "owner/name").

    Returns
    -------
    str
        A sanitized string with slashes replaced by double underscores.
    """
    return model_id.replace("/", "__")


def cache_sentence_transformers(
    model_ids: Iterable[str],
    output_dir: Path,
    *,
    trust_remote_code: bool = True,
) -> None:
    """
    Download and persist SentenceTransformer models to disk.

    Parameters
    ----------
    model_ids : Iterable[str]
        List/iterable of SentenceTransformer model IDs to cache.
    output_dir : Path
        Directory where models will be saved (one subfolder per model ID).
    trust_remote_code : bool, optional
        Whether to trust and execute custom modeling code from the repository.
        Default is True. Use with caution for untrusted sources.

    Notes
    ------------
    - Creates directories under `output_dir`.
    - Writes model weights/configuration to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        print(f"📦 Caching sentence-transformers model: {model_id}")
        save_path = output_dir / sanitize_model_id(model_id)
        model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
        model.save(str(save_path))


def cache_cross_encoders(model_ids: Iterable[str], output_dir: Path) -> None:
    """
    Download and persist CrossEncoder reranker models to disk.

    Parameters
    ----------
    model_ids : Iterable[str]
        List/iterable of CrossEncoder model IDs to cache.
    output_dir : Path
        Directory where reranker models will be saved.

    Notes
    ------------
    - Creates directories under `output_dir`.
    - Writes model weights/configuration to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        print(f"🔁 Caching reranker: {model_id}")
        save_path = output_dir / sanitize_model_id(model_id)
        reranker = CrossEncoder(model_id)
        reranker.save(str(save_path))


if __name__ == "__main__":
    # Execute the caching routines with the configured model lists and paths.
    cache_sentence_transformers(EMBEDDING_MODELS, EMBEDDINGS_DIR, trust_remote_code=True)
    cache_cross_encoders(RERANKER_MODELS, RERANKERS_DIR)
