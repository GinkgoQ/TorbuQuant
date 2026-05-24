"""Vector search module for TurboQuant.

Provides compressed vector search using TurboQuant's quantization algorithms.
Supports approximate nearest-neighbor search over compressed embedding corpora.

Example:
    ```python
    from turboquant.search import compress_vectors, search, CompressedCorpus

    # Compress embeddings
    corpus = compress_vectors(embeddings, bits=3.5, m=64)

    # Search
    scores, indices = search(query, corpus, top_k=10)
    ```
"""

from turboquant.search.functional import (
    CompressedCorpus,
    compress_vectors,
    search,
    inner_product,
    decompress,
)

__all__ = [
    "CompressedCorpus",
    "compress_vectors",
    "search",
    "inner_product",
    "decompress",
]
