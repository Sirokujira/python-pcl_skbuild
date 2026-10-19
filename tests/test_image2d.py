"""pcl/2d image functions against numpy references."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._image2d")


def _correlate(image, kernel, mode):
    """out[v, u] = sum k[m, l] image[v + m - kh//2, u + l - kw//2]."""
    kh, kw = kernel.shape
    pad = {"clamp": "edge", "zero": "constant", "mirror": "symmetric"}[mode]
    top, left = kh // 2, kw // 2
    padded = np.pad(image, ((top, kh - 1 - top), (left, kw - 1 - left)),
                    mode=pad)
    out = np.zeros_like(image, dtype=np.float64)
    for m in range(kh):
        for l in range(kw):
            out += kernel[m, l] * padded[m:m + image.shape[0],
                                         l:l + image.shape[1]]
    return out


def test_kernels():
    assert np.array_equal(pcl.image_kernel("sobel_x"),
                          [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    assert pcl.image_kernel("roberts_x").shape == (2, 2)
    assert pcl.image_kernel("derivative_central_x").shape == (1, 3)
    assert pcl.image_kernel("derivative_central_y").shape == (3, 1)
    g = pcl.image_kernel("gaussian", 5, 1.0)
    assert g.shape == (5, 5) and g.sum() == pytest.approx(1.0)
    assert np.allclose(g, g.T) and g[2, 2] == g.max()
    log = pcl.image_kernel("log", 3, 1.0)
    assert log[1, 1] > 0 and log.sum() == pytest.approx(1.0)
    # the Mexican hat sums towards 0 as it grows, so dividing by the sum
    # inflates the larger kernel: same shape, very different scale
    big = pcl.image_kernel("log", 9, 1.0)
    assert big[4, 4] > 100 * log[1, 1]
    with pytest.raises(ValueError):
        pcl.image_kernel("gaussian", 4)
    with pytest.raises(ValueError):
        pcl.image_kernel("blur")


@pytest.mark.parametrize("mode", ["clamp", "zero", "mirror"])
def test_convolve_is_a_correlation(mode):
    rng = np.random.RandomState(0)
    image = rng.rand(12, 17).astype(np.float32)
    for kernel in (rng.rand(3, 3), rng.rand(3, 5), rng.rand(2, 2)):
        got = pcl.image_convolve(image, kernel, mode)
        assert np.allclose(got, _correlate(image, kernel.astype(np.float32),
                                           mode), atol=1e-5)
    with pytest.raises(ValueError):
        pcl.image_convolve(image, np.ones((31, 3)), "mirror")


def test_gradient_edges_and_log():
    image = np.zeros((10, 12), np.float32)
    image[:, 6:] = 100
    edges = pcl.image_edges(image, "sobel")
    gx = _correlate(image, pcl.image_kernel("sobel_x"), "clamp")
    assert np.allclose(edges["x"], gx)
    assert np.allclose(edges["magnitude"], np.hypot(edges["x"], edges["y"]))
    assert edges["magnitude"][5, 5] == pytest.approx(400)
    assert edges["magnitude"][5, 2] == 0
    assert edges["direction"][5, 5] == pytest.approx(0)
    for detector in ("prewitt", "roberts"):
        assert pcl.image_edges(image, detector)["magnitude"].max() > 0
    log = pcl.image_log(image, 1.0, 3)
    assert np.allclose(log, _correlate(image, pcl.image_kernel("log", 3, 1.0),
                                       "clamp"), atol=1e-3)


def test_canny_traces_the_square():
    image = np.zeros((40, 40), np.float32)
    image[10:30, 10:30] = 200
    edges = pcl.image_canny(image, 20, 80)
    assert edges.dtype == bool
    ring = np.zeros_like(edges)
    ring[8:32, 8:32] = True
    ring[12:28, 12:28] = False
    assert edges.any() and not edges[~ring].any()
    # every side of the square is found
    assert edges[8:12, 15:25].any() and edges[28:32, 15:25].any()
    assert edges[15:25, 8:12].any() and edges[15:25, 28:32].any()


def _gray(image, element, reduce):
    eh, ew = element.shape
    out = np.empty_like(image)
    for v in range(image.shape[0]):
        for u in range(image.shape[1]):
            values = [image[v + m - eh // 2, u + l - ew // 2]
                      for m in range(eh) for l in range(ew)
                      if element[m, l] and
                      0 <= v + m - eh // 2 < image.shape[0] and
                      0 <= u + l - ew // 2 < image.shape[1]]
            out[v, u] = reduce(values)
    return out


def test_morphology():
    disk = pcl.structuring_element("circle", radius=2)
    assert disk.shape == (4, 4)
    # pixels strictly closer than 2 to (2, 2) on a 4 x 4 grid
    assert np.array_equal(disk[1:, 1:], np.ones((3, 3)))
    assert not disk[0].any() and not disk[:, 0].any()
    assert pcl.structuring_element("circle", radius=3)[0].sum() == 0
    assert np.array_equal(pcl.structuring_element("rectangle", height=2,
                                                  width=3), np.ones((2, 3)))
    rng = np.random.RandomState(1)
    image = rng.randint(-5, 6, (9, 11)).astype(np.float32)
    image[4, 4] = -1          # PCL's "nothing seen yet" value
    element = np.ones((3, 3), np.float32)
    assert np.array_equal(pcl.image_morphology(image, "erosion", element),
                          _gray(image, element, min))
    assert np.array_equal(pcl.image_morphology(image, "dilation", element),
                          _gray(image, element, max))
    binary = np.zeros((9, 9), np.float32)
    binary[2:7, 2:7] = 1
    binary[0, 8] = 1          # an isolated pixel opening removes
    opened = pcl.image_morphology(binary, "opening", element, binary=True)
    assert opened[0, 8] == 0 and opened[4, 4] == 1
    with pytest.raises(ValueError, match="0 and 1"):
        pcl.image_morphology(image, "erosion", element, binary=True)
    with pytest.raises(ValueError):
        pcl.image_morphology(image, "thinning", element)


def test_binary_operations():
    a = np.array([[1, 1, 0, 0]], np.float32)
    b = np.array([[1, 0, 1, 0]], np.float32)
    assert pcl.image_binary_operation(a, b, "subtraction").tolist() == \
        [[0, 1, 0, 0]]
    assert pcl.image_binary_operation(a, b, "union").tolist() == [[1, 1, 1, 0]]
    assert pcl.image_binary_operation(a, b, "intersection").tolist() == \
        [[1, 0, 0, 0]]
    with pytest.raises(ValueError):
        pcl.image_binary_operation(a, b[:, :3], "union")
