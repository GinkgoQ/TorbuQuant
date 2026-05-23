import numpy as np
import pytest
from scipy import integrate

from torbuquant.core.codebook import beta_pdf, compute_lloyd_max


def test_beta_pdf_integrates_to_one_and_is_symmetric():
    d = 16
    area, _ = integrate.quad(lambda t: beta_pdf(np.array([t]), d)[0], -1.0, 1.0)
    xs = np.linspace(0.0, 0.95, 19)

    assert abs(area - 1.0) < 1e-8
    np.testing.assert_allclose(beta_pdf(xs, d), beta_pdf(-xs, d), rtol=1e-12, atol=1e-12)


def test_exact_lloyd_max_codebook_has_order_and_symmetry():
    cb = compute_lloyd_max(8, 2, max_iter=80, tol=1e-11)
    centroids = np.asarray(cb["centroids"])
    boundaries = np.asarray(cb["boundaries"])

    assert cb["d"] == 8
    assert cb["bits"] == 2
    assert cb["use_exact"] is True
    assert centroids.shape == (4,)
    assert boundaries.shape == (5,)
    assert np.all(np.diff(centroids) > 0)
    assert np.all(np.diff(boundaries) > 0)
    np.testing.assert_allclose(centroids, -centroids[::-1], atol=2e-8)
    np.testing.assert_allclose(boundaries, -boundaries[::-1], atol=2e-8)


def test_lloyd_max_mse_decreases_with_bit_width():
    cb1 = compute_lloyd_max(8, 1, max_iter=80, tol=1e-11)
    cb2 = compute_lloyd_max(8, 2, max_iter=80, tol=1e-11)

    assert cb2["mse_per_dim"] < cb1["mse_per_dim"]


def test_lloyd_max_rejects_invalid_bits():
    with pytest.raises(ValueError, match="bits must"):
        compute_lloyd_max(8, 0)


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (1, 0.36),
        (2, 0.117),
        (3, 0.03),
        (4, 0.009),
    ],
)
def test_lloyd_max_matches_paper_level_distortion_values(bits, expected):
    cb = compute_lloyd_max(128, bits, max_iter=300, tol=1e-13)
    total_mse = cb["mse_per_dim"] * cb["d"]

    assert total_mse == pytest.approx(expected, abs=0.006)
