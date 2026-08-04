"""Runtime tests for the HOG descriptor (pcl/people).

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

Only HOG is wrapped; see src/pcl/_people.pyx for why the rest of
pcl/people is not reachable from a build without VTK.
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def vertical_edge():
    """Black left half, white right half — one dominant gradient."""
    image = np.zeros((64, 32), dtype=np.float32)
    image[:, 16:] = 1.0
    return image


# --- descriptor size ---------------------------------------------------

def test_descriptor_size_is_what_hog_returns(vertical_edge):
    """PCL does not document the output length and writes past the end
    of a short buffer, so the two must not be allowed to drift."""
    descriptor = pcl.hog(vertical_edge, bin_size=8, n_orients=9)
    assert descriptor.shape == (pcl.hog_descriptor_size(64, 32, 8, 9),)


@pytest.mark.parametrize("height,width,bin_size,n_orients", [
    (64, 32, 8, 9),
    (64, 32, 8, 6),
    (64, 32, 4, 9),
    (32, 32, 8, 9),
    (128, 64, 8, 9),
])
def test_descriptor_size_formula(height, width, bin_size, n_orients):
    """(h/bin - 2) * (w/bin - 2) * n_orients * 4, measured against PCL."""
    expected = ((height // bin_size - 2) * (width // bin_size - 2)
                * n_orients * 4)
    assert pcl.hog_descriptor_size(height, width, bin_size, n_orients) \
        == expected


def test_descriptor_size_is_zero_for_a_too_small_image():
    """HOG needs more than two bins on a side; the size says so rather
    than pretending."""
    assert pcl.hog_descriptor_size(16, 16, 8, 9) == 0


def test_descriptor_size_rejects_nonsense():
    with pytest.raises(ValueError, match="bin_size"):
        pcl.hog_descriptor_size(64, 32, 0, 9)
    with pytest.raises(ValueError, match="n_orients"):
        pcl.hog_descriptor_size(64, 32, 8, 0)


# --- the descriptor itself ---------------------------------------------

def test_hog_returns_finite_float32(vertical_edge):
    descriptor = pcl.hog(vertical_edge)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()


def test_hog_responds_to_an_edge(vertical_edge):
    """A flat image has no gradient; an edge does.

    Not exactly zero: PCL's normalization divides by the block norm plus
    an epsilon, so a gradient-free image comes back at ~1e-6 rather than
    dividing by zero.
    """
    flat = np.zeros((64, 32), dtype=np.float32)
    assert pcl.hog(flat).max() < 1e-5
    assert pcl.hog(vertical_edge).max() > 0.1


def test_hog_is_deterministic(vertical_edge):
    assert np.array_equal(pcl.hog(vertical_edge), pcl.hog(vertical_edge))


def test_hog_distinguishes_edge_orientations():
    """A horizontal edge and a vertical one must not describe the same."""
    vertical = np.zeros((64, 64), dtype=np.float32)
    vertical[:, 32:] = 1.0
    horizontal = vertical.T.copy()
    assert not np.allclose(pcl.hog(vertical), pcl.hog(horizontal))


def test_hog_accepts_a_channel_axis(vertical_edge):
    with_channel = vertical_edge[:, :, np.newaxis]
    assert np.array_equal(pcl.hog(with_channel), pcl.hog(vertical_edge))


def test_hog_bin_size_controls_the_length(vertical_edge):
    assert len(pcl.hog(vertical_edge, bin_size=4)) \
        > len(pcl.hog(vertical_edge, bin_size=8))


def test_hog_n_orients_controls_the_length(vertical_edge):
    assert len(pcl.hog(vertical_edge, n_orients=12)) \
        > len(pcl.hog(vertical_edge, n_orients=6))


def test_hog_soft_bin_changes_the_result(vertical_edge):
    assert not np.array_equal(pcl.hog(vertical_edge, soft_bin=True),
                              pcl.hog(vertical_edge, soft_bin=False))


def test_hog_accepts_a_list(vertical_edge):
    """Anything numpy can make an array of, not just an ndarray."""
    assert np.array_equal(pcl.hog(vertical_edge.tolist()),
                          pcl.hog(vertical_edge))


# --- error paths -------------------------------------------------------

def test_hog_rejects_a_too_small_image():
    small = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="too small"):
        pcl.hog(small, bin_size=8)


def test_hog_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="height, width"):
        pcl.hog(np.zeros(64, dtype=np.float32))


def test_hog_rejects_nonsense_settings(vertical_edge):
    with pytest.raises(ValueError, match="bin_size"):
        pcl.hog(vertical_edge, bin_size=0)
    with pytest.raises(ValueError, match="n_orients"):
        pcl.hog(vertical_edge, n_orients=0)
