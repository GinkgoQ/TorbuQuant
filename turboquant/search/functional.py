"""TurboQuant vector search functional API.

Provides compressed vector search using TurboQuant's quantization.
Supports approximate nearest-neighbor search without index building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from turboquant.core.mse import TorbuquantMSE
from turboquant.core.polar import TorbuquantProd
from turboquant.core.types import MSEData, ProdData


@dataclass
class CompressedCorpus:
    """Compressed corpus for vector search.

    Attributes:
        mse_data: MSE quantization data (for simple compression).
        prod_data: Product quantization data (for inner-product aware).
        quantizer: Quantizer used for compression.
        dim: Original vector dimension.
        num_vectors: Number of vectors in corpus.
        bits: Bits per element used.
    """
    mse_data: Optional[MSEData] = None
    prod_data: Optional[ProdData] = None
    quantizer: Optional[TorbuquantMSE | TorbuquantProd] = None
    dim: int = 0
    num_vectors: int = 0
    bits: float = 4.0


def compress_vectors(
    embeddings: torch.Tensor,
    bits: float = 3.5,
    use_prod: bool = True,
    seed: int = 42,
) -> CompressedCorpus:
    """Compress a corpus of embedding vectors for fast approximate search.

    Args:
        embeddings: Document embeddings of shape ``[N_docs, dim]``.
        bits: Bits per element. Must be > 1.0 for product quantization.
        use_prod: Use TurboQuantProd (inner-product aware) vs MSE-only.
        seed: Random seed for rotation matrices.

    Returns:
        CompressedCorpus ready for search.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N, dim], got {embeddings.shape}")

    N, dim = embeddings.shape

    if use_prod and bits > 1.0:
        # Use product quantization for inner-product preservation
        mse_bits = int(bits - 1) if bits > 1 else int(bits)
        quantizer = TorbuquantProd(
            dim=dim,
            mse_bits=max(1, mse_bits),
            seed=seed,
        )
        # Reshape to [1, N, dim] for batch processing
        x = embeddings.unsqueeze(0)
        prod_data = quantizer.quantize(x)

        return CompressedCorpus(
            prod_data=prod_data,
            quantizer=quantizer,
            dim=dim,
            num_vectors=N,
            bits=bits,
        )
    else:
        # Use MSE-only quantization
        quantizer = TorbuquantMSE(
            dim=dim,
            bits=int(bits),
            seed=seed,
        )
        x = embeddings.unsqueeze(0)
        mse_data = quantizer.quantize(x)

        return CompressedCorpus(
            mse_data=mse_data,
            quantizer=quantizer,
            dim=dim,
            num_vectors=N,
            bits=bits,
        )


def inner_product(
    queries: torch.Tensor,
    corpus: CompressedCorpus,
) -> torch.Tensor:
    """Compute approximate inner products between queries and corpus.

    Args:
        queries: Query embeddings ``[N_queries, dim]`` or ``[dim]``.
        corpus: Compressed corpus from compress_vectors.

    Returns:
        Inner product scores ``[N_queries, N_docs]``.
    """
    squeeze = queries.ndim == 1
    if squeeze:
        queries = queries.unsqueeze(0)

    if queries.shape[-1] != corpus.dim:
        raise ValueError(
            f"Query dim {queries.shape[-1]} != corpus dim {corpus.dim}"
        )

    N_q = queries.shape[0]

    if corpus.prod_data is not None:
        # Use product quantization's unbiased inner product
        quantizer = corpus.quantizer
        if not isinstance(quantizer, TorbuquantProd):
            raise TypeError("Expected TurboQuantProd quantizer for prod_data")

        # Dequantize corpus
        corpus_vecs = quantizer.dequantize(corpus.prod_data)  # [1, N_docs, dim]
        corpus_vecs = corpus_vecs.squeeze(0)  # [N_docs, dim]

        # Compute scores
        scores = queries.float() @ corpus_vecs.float().T  # [N_q, N_docs]

    elif corpus.mse_data is not None:
        # Use MSE dequantization
        quantizer = corpus.quantizer
        if not isinstance(quantizer, TorbuquantMSE):
            raise TypeError("Expected TurboQuantMSE quantizer for mse_data")

        corpus_vecs = quantizer.dequantize(corpus.mse_data)  # [1, N_docs, dim]
        corpus_vecs = corpus_vecs.squeeze(0)

        scores = queries.float() @ corpus_vecs.float().T

    else:
        raise ValueError("Corpus has no compressed data")

    if squeeze:
        scores = scores.squeeze(0)

    return scores


def search(
    query: torch.Tensor,
    corpus: CompressedCorpus,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate nearest-neighbor search over compressed corpus.

    No index build step required — add documents, query immediately.

    Args:
        query: Query embedding ``[dim]`` or ``[N_q, dim]``.
        corpus: Compressed corpus from compress_vectors.
        top_k: Number of nearest neighbours to return.

    Returns:
        scores: Top-k dot-product scores ``[top_k]`` or ``[N_q, top_k]``.
        indices: Top-k indices into corpus, same shape as scores.
    """
    squeeze = query.ndim == 1
    if squeeze:
        query = query.unsqueeze(0)

    scores = inner_product(query, corpus)  # [N_q, N_docs]

    k = min(top_k, scores.shape[-1])

    # Get top-k indices (descending order)
    top_scores, top_indices = torch.topk(scores, k, dim=-1, largest=True, sorted=True)

    if squeeze:
        return top_scores.squeeze(0), top_indices.squeeze(0)
    return top_scores, top_indices


def decompress(
    corpus: CompressedCorpus,
) -> torch.Tensor:
    """Decompress corpus back to approximate vectors.

    Note:
        This is lossy - the decompressed vectors are approximations.
        Use for debugging and visualization only.

    Args:
        corpus: Compressed corpus.

    Returns:
        Approximate vectors ``[N_docs, dim]``.
    """
    if corpus.prod_data is not None:
        quantizer = corpus.quantizer
        if not isinstance(quantizer, TorbuquantProd):
            raise TypeError("Expected TurboQuantProd quantizer")
        vecs = quantizer.dequantize(corpus.prod_data)
        return vecs.squeeze(0)

    elif corpus.mse_data is not None:
        quantizer = corpus.quantizer
        if not isinstance(quantizer, TorbuquantMSE):
            raise TypeError("Expected TurboQuantMSE quantizer")
        vecs = quantizer.dequantize(corpus.mse_data)
        return vecs.squeeze(0)

    else:
        raise ValueError("Corpus has no compressed data")


def compute_recall(
    queries: torch.Tensor,
    corpus_original: torch.Tensor,
    corpus_compressed: CompressedCorpus,
    top_k: int = 10,
) -> float:
    """Compute recall@k for compressed search vs exact search.

    Args:
        queries: Query embeddings ``[N_q, dim]``.
        corpus_original: Original corpus embeddings ``[N_docs, dim]``.
        corpus_compressed: Compressed corpus.
        top_k: k for recall@k.

    Returns:
        Recall@k as float in [0, 1].
    """
    # Exact search
    exact_scores = queries.float() @ corpus_original.float().T
    _, exact_indices = torch.topk(exact_scores, top_k, dim=-1)

    # Compressed search
    _, compressed_indices = search(queries, corpus_compressed, top_k=top_k)

    # Compute recall
    exact_set = exact_indices.tolist()
    compressed_set = compressed_indices.tolist()

    total_recall = 0.0
    N_q = len(exact_set)

    for i in range(N_q):
        exact_topk = set(exact_set[i])
        compressed_topk = set(compressed_set[i])
        overlap = len(exact_topk & compressed_topk)
        total_recall += overlap / top_k

    return total_recall / N_q
