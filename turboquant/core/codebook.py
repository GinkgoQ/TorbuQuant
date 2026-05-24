"""
turboquant.core.codebook
========================
Lloyd-Max codebook computation for the exact Beta distribution arising from
random rotation of unit-norm vectors.

Math background (paper Section 3.1, Lemma 1):
  After multiplying a unit-norm vector x in S^{d-1} by a Haar-random rotation
  matrix Pi, each coordinate y_j = (Pi x)_j follows:

      f(t) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
             * (1 - t^2)^((d-3)/2), t in [-1, 1]

  which is a scaled-shifted Beta distribution.  For large d this converges to
  N(0, 1/d), but we use the EXACT integral at every d so the codebook is
  numerically precise for all head dimensions such as 64, 96, 128, 256.

Lloyd-Max algorithm:
  1. Initialise 2^bits centroids at quantile midpoints of f.
  2. Alternate:
       a. Boundaries = midpoints between consecutive centroids.
       b. Centroids = conditional means E[y | boundary[i] < y < boundary[i+1]].
  3. Stop when cost delta < tol.

Caching:
  Solved codebooks are stored under <package>/core/codebooks/ as JSON files
  named "d{D}_b{B}.json".  They are loaded on first use and kept in RAM.

Both the exact Beta and the Gaussian approximation are exposed.
Use exact Beta for production; Gaussian is kept as a diagnostic path.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, Literal, Tuple

import numpy as np
import torch
from scipy import integrate, special

# On-disk cache directory.
_CB_DIR = os.path.join(os.path.dirname(__file__), "codebooks")
_CB_RAM: Dict[Tuple[int, int, bool], dict] = {}
logger = logging.getLogger(__name__)


# Beta distribution PDF.

def beta_pdf(t: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a uniformly random point on S^{d-1}.

        f(t) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
               * (1 - t^2)^((d-3)/2)

    Args:
        t  : array of evaluation points in (-1, 1).
        d  : dimension of the embedding space, d >= 3.

    Returns:
        Array of probability density values, same shape as t.
    """
    if d < 3:
        raise ValueError(f"d must be >= 3, got {d}")
    log_const = (
        special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    t_clipped = np.clip(t, -1.0 + 1e-14, 1.0 - 1e-14)
    return np.exp(log_const + exponent * np.log(1.0 - t_clipped ** 2))


def gaussian_approx_pdf(t: np.ndarray, d: int) -> np.ndarray:
    """
    Gaussian approximation N(0, 1/d) of the Beta PDF (diagnostic path only).
    Exact Beta should be preferred for production codebooks.
    """
    sigma2 = 1.0 / d
    return np.exp(-0.5 * t ** 2 / sigma2) / math.sqrt(2.0 * math.pi * sigma2)


# Lloyd-Max solver.

def _conditional_mean(lo: float, hi: float, d: int) -> float:
    """E[y | lo < y < hi] under exact Beta PDF."""
    # numerator: integral t*f(t) dt; denominator: integral f(t) dt
    num, _ = integrate.quad(lambda t: t * beta_pdf(np.array([t]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda t: beta_pdf(np.array([t]), d)[0], lo, hi)
    if den < 1e-30:
        return (lo + hi) / 2.0
    return num / den


def _conditional_mean_gaussian(lo: float, hi: float, d: int) -> float:
    """E[y | lo < y < hi] under Gaussian approximation N(0,1/d)."""
    from scipy import stats
    sigma = 1.0 / math.sqrt(d)
    a = lo / sigma if math.isfinite(lo) else lo
    b = hi / sigma if math.isfinite(hi) else hi
    prob = stats.norm.cdf(b) - stats.norm.cdf(a)
    if prob < 1e-15:
        return (lo + hi) / 2.0
    pdf_diff = stats.norm.pdf(a) - stats.norm.pdf(b)
    return sigma * pdf_diff / prob


def _mse_cost(centroids: np.ndarray, boundaries: np.ndarray, d: int) -> float:
    """Total MSE cost for a set of centroids and boundaries under exact Beta."""
    n = len(centroids)
    cost = 0.0
    for i in range(n):
        lo, hi = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(
            lambda t, _c=c: (t - _c) ** 2 * beta_pdf(np.array([t]), d)[0],
            lo, hi,
        )
        cost += val
    return cost


def _initial_centroids(d: int, n: int, use_exact: bool) -> np.ndarray:
    """Place initial centroids at quantile midpoints of the chosen PDF."""
    x = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, 20_000)
    if use_exact:
        pdf_vals = beta_pdf(x, d)
    else:
        pdf_vals = gaussian_approx_pdf(x, d)
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf_vals) * dx
    cdf /= cdf[-1]
    edges = np.linspace(0.0, 1.0, n + 1)
    centroids = np.zeros(n)
    for i in range(n):
        q_mid = (edges[i] + edges[i + 1]) / 2.0
        idx = int(np.searchsorted(cdf, q_mid))
        idx = min(idx, len(x) - 1)
        centroids[i] = x[idx]
    return centroids


def compute_lloyd_max(
    d: int,
    bits: int,
    *,
    use_exact: bool = True,
    max_iter: int = 300,
    tol: float = 1e-13,
) -> dict:
    """
    Solve the 1-D Lloyd-Max problem for the Beta (or Gaussian) distribution.

    Args:
        d         : head dimension.
        bits      : bits per coordinate (1-8).
        use_exact : if True use exact Beta; if False use Gaussian approximation.
        max_iter  : maximum Lloyd-Max iterations.
        tol       : convergence threshold on absolute cost delta.

    Returns dict with keys:
        centroids   : list of 2^bits float64 centroids (sorted).
        boundaries  : list of 2^bits + 1 boundaries (includes -1 and +1).
        mse_per_dim : scalar, MSE per coordinate dimension.
        d, bits, use_exact.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    n = 2 ** bits
    mean_fn = _conditional_mean if use_exact else (
        lambda lo, hi, _d: _conditional_mean_gaussian(lo, hi, _d)
    )

    centroids = _initial_centroids(d, n, use_exact)
    prev_cost = float("inf")

    for _ in range(max_iter):
        # Step A: boundaries = midpoints between consecutive centroids
        bnd = np.empty(n + 1)
        bnd[0] = -1.0
        bnd[-1] = 1.0
        bnd[1:-1] = (centroids[:-1] + centroids[1:]) / 2.0

        # Step B: centroids = conditional means
        new_c = np.array([mean_fn(bnd[i], bnd[i + 1], d) for i in range(n)])

        # Recompute cost with the EXACT Beta regardless of approximation mode
        cost = _mse_cost(new_c, bnd, d)
        centroids = new_c

        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    # Last boundaries
    bnd = np.empty(n + 1)
    bnd[0] = -1.0
    bnd[-1] = 1.0
    bnd[1:-1] = (centroids[:-1] + centroids[1:]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": bnd.tolist(),
        "mse_per_dim": float(cost),
        "d": d,
        "bits": bits,
        "use_exact": use_exact,
    }


# On-disk cache helpers.

def _cache_path(d: int, bits: int, use_exact: bool) -> str:
    tag = "exact" if use_exact else "gaussian"
    os.makedirs(_CB_DIR, exist_ok=True)
    return os.path.join(_CB_DIR, f"d{d}_b{bits}_{tag}.json")


def get_codebook(
    d: int,
    bits: int,
    *,
    use_exact: bool = True,
) -> dict:
    """
    Return a Lloyd-Max codebook, loading from disk or computing on first call.

    Args:
        d         : head dimension.
        bits      : bits per coordinate.
        use_exact : use exact Beta distribution (True) or Gaussian approx (False).

    Returns:
        dict with 'centroids', 'boundaries', 'mse_per_dim', 'd', 'bits'.
    """
    key = (d, bits, use_exact)
    if key in _CB_RAM:
        return _CB_RAM[key]

    path = _cache_path(d, bits, use_exact)
    if os.path.exists(path):
        with open(path) as fh:
            cb = json.load(fh)
        _CB_RAM[key] = cb
        return cb

    tag = "exact Beta" if use_exact else "Gaussian approx"
    logger.info("computing Lloyd-Max codebook d=%s bits=%s (%s)", d, bits, tag)
    cb = compute_lloyd_max(d, bits, use_exact=use_exact)
    with open(path, "w") as fh:
        json.dump(cb, fh, indent=2)
    logger.info("Lloyd-Max MSE per dim: %.6e", cb["mse_per_dim"])
    _CB_RAM[key] = cb
    return cb


def get_codebook_tensors(
    d: int,
    bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    use_exact: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (centroids, decision_boundaries) as GPU tensors.

    centroids          : (2^bits,) centroid values
    decision_boundaries: (2^bits-1,) interior boundaries for searchsorted

    The decision boundaries are the interior ones, i.e., boundaries[1:-1],
    which is what torch.searchsorted needs to map each coordinate to an index.
    """
    cb = get_codebook(d, bits, use_exact=use_exact)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    all_bnd = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    decision_bnd = all_bnd[1:-1].contiguous()
    return centroids, decision_bnd
