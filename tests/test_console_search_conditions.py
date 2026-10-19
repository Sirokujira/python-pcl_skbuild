"""PCL's verbosity switch, the pcl::search back-ends BruteForce and
FlannSearch, and ConditionalRemoval's OR / nested / quadric / colour
conditions."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._pointcloud")


# --- console ------------------------------------------------------------------


def test_verbosity_round_trips():
    before = pcl.get_verbosity_level()
    try:
        for level in (pcl.L_ALWAYS, pcl.L_ERROR, pcl.L_WARN, pcl.L_INFO,
                      pcl.L_DEBUG, pcl.L_VERBOSE):
            pcl.set_verbosity_level(level)
            assert pcl.get_verbosity_level() == level
        with pytest.raises(ValueError):
            pcl.set_verbosity_level(99)
    finally:
        pcl.set_verbosity_level(before)


def test_verbosity_silences_pcl_info(capfd):
    """GrabCut reports every iteration with PCL_INFO."""
    h, w = 12, 12
    v, u = np.mgrid[0:h, 0:w]
    xyz = np.dstack([u * 0.01, v * 0.01, np.ones((h, w))]).astype(np.float32)
    rgba = np.full((h, w, 4), 255, np.uint8)
    rgba[3:9, 3:9, :3] = (255, 0, 0)
    rgba[~((v >= 3) & (v < 9) & (u >= 3) & (u < 9)), :3] = (0, 0, 255)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)

    def run():
        grabcut = pcl.GrabCut()
        grabcut.set_InputCloud(cloud)
        grabcut.set_ForegroundRegion(
            np.flatnonzero((v >= 1) & (v < 11) & (u >= 1) & (u < 11)))
        grabcut.extract()

    before = pcl.get_verbosity_level()
    try:
        pcl.set_verbosity_level(pcl.L_INFO)
        run()
        assert "pixels changed" in "".join(capfd.readouterr())
        pcl.set_verbosity_level(pcl.L_WARN)
        run()
        assert "pixels changed" not in "".join(capfd.readouterr())
    finally:
        pcl.set_verbosity_level(before)


# --- search -------------------------------------------------------------------


@pytest.fixture(scope="module")
def blob():
    rng = np.random.RandomState(0)
    return pcl.PointCloud(rng.rand(500, 3).astype(np.float32))


def _reference(cloud, k):
    pts = cloud.to_array().astype(np.float64)
    d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, axis=1)[:, :k]
    return order, np.take_along_axis(d, order, 1)


def test_brute_force_matches_exhaustive_search(blob):
    search = pcl.BruteForceSearch(blob)
    ind, sqd = search.nearest_k_search_for_cloud(blob, 4)
    ref_ind, ref_sqd = _reference(blob, 4)
    assert np.allclose(sqd, ref_sqd, atol=1e-6)
    assert np.all(ind[:, 0] == np.arange(blob.size))
    ind, sqd = search.radius_search_for_cloud(blob, 0.1, 50)
    assert ind.shape == (blob.size, 50)
    assert np.all(sqd <= 0.1 ** 2 + 1e-7)
    with pytest.raises(RuntimeError):
        pcl.BruteForceSearch().nearest_k_search_for_cloud(blob, 1)


@pytest.mark.parametrize("index", ["FLANN_INDEX_KDTREE_SINGLE",
                                   "FLANN_INDEX_KDTREE_MULTI",
                                   "FLANN_INDEX_KMEANS"])
def test_flann_search_indices(blob, index):
    search = pcl.FlannSearch(blob, index=getattr(pcl, index), trees=4)
    search.set_Checks(256)
    assert search.get_Checks() == 256
    search.set_Epsilon(0.0)
    ind, sqd = search.nearest_k_search_for_cloud(blob, 4)
    ref_ind, ref_sqd = _reference(blob, 4)
    if index == "FLANN_INDEX_KDTREE_SINGLE":
        assert np.allclose(sqd, ref_sqd, atol=1e-6)
    else:
        # approximate indices: nearly always right at 256 checks
        assert np.mean(np.isclose(sqd, ref_sqd, atol=1e-6)) > 0.95
    i, d = search.nearest_k_search_for_point(blob, 10, 1)
    assert i[0] == 10 and d[0] == pytest.approx(0.0)


def test_flann_search_checks_arguments(blob):
    with pytest.raises(ValueError):
        pcl.FlannSearch(blob, index=7)
    with pytest.raises(ValueError):
        pcl.FlannSearch(blob, trees=0)
    with pytest.raises(ValueError):
        pcl.FlannSearch(blob).set_Checks(0)


# --- conditions ---------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    v = np.linspace(-1, 1, 21)
    x, y, z = np.meshgrid(v, v, v, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)


def _filtered(cloud, condition):
    f = pcl.ConditionalRemoval(condition, cloud)
    return f.filter().to_array()


def test_condition_or_keeps_either_side(grid):
    cond = pcl.ConditionOr()
    cond.add_Comparison2("x", pcl.CompareOp_LT, -0.5)
    cond.add_Comparison2("x", pcl.CompareOp_GT, 0.5)
    out = _filtered(pcl.PointCloud(grid), cond)
    assert len(out) == np.sum((grid[:, 0] < -0.5) | (grid[:, 0] > 0.5))
    assert np.all(np.abs(out[:, 0]) > 0.5)
    # PCL: an empty OR is true
    assert len(_filtered(pcl.PointCloud(grid), pcl.ConditionOr())) == len(grid)


def test_nested_conditions(grid):
    either = pcl.ConditionOr()
    either.add_Comparison2("x", pcl.CompareOp_LT, -0.5)
    either.add_Comparison2("x", pcl.CompareOp_GT, 0.5)
    both = pcl.ConditionAnd()
    both.add_Comparison2("z", pcl.CompareOp_GE, 0.0)
    both.add_Condition(either)
    out = _filtered(pcl.PointCloud(grid), both)
    expect = (grid[:, 2] >= 0) & ((grid[:, 0] < -0.5) | (grid[:, 0] > 0.5))
    assert len(out) == expect.sum()
    loop = pcl.ConditionAnd()
    loop.add_Condition(loop)
    with pytest.raises(ValueError, match="itself"):
        _filtered(pcl.PointCloud(grid), loop)


def test_quadric_comparison_keeps_a_sphere(grid):
    centre = np.array([0.2, 0.0, -0.1])
    radius = 0.5
    cond = pcl.ConditionAnd()
    cond.add_TfQuadraticXYZComparison(pcl.CompareOp_LT, np.eye(3), -centre,
                                      centre @ centre - radius ** 2)
    out = _filtered(pcl.PointCloud(grid), cond)
    distance = np.linalg.norm(grid - centre, axis=1)
    # grid points exactly on the sphere may land either side in float32
    assert (distance < radius - 1e-4).sum() <= len(out) <= \
        (distance < radius + 1e-4).sum()
    assert np.all(np.linalg.norm(out - centre, axis=1) < radius + 1e-5)
    with pytest.raises(ValueError):
        cond.add_TfQuadraticXYZComparison(pcl.CompareOp_LT, np.eye(2),
                                          [0, 0], 0.0)


def _coloured(cls, n=200):
    rng = np.random.RandomState(3)
    xyz = rng.rand(n, 3).astype(np.float32)
    rgb = np.zeros((n, 4), np.uint8)
    rgb[:, 3] = 255
    rgb[: n // 2, :3] = (230, 20, 20)
    rgb[n // 2:, :3] = (20, 20, 230)
    cloud = cls()
    cloud.from_rgba_array(xyz, rgb) if cls is pcl.PointCloud_PointXYZRGBA \
        else cloud.from_rgb_array(xyz, rgb[:, :3])
    return cloud, n


@pytest.mark.parametrize("cls", ["PointCloud_PointXYZRGB",
                                 "PointCloud_PointXYZRGBA"])
def test_packed_colour_comparisons(cls):
    cloud, n = _coloured(getattr(pcl, cls))
    red = pcl.ConditionAnd()
    red.add_PackedRGBComparison("r", pcl.CompareOp_GT, 128)
    out = pcl.ConditionalRemoval(red, cloud).filter()
    assert type(out) is type(cloud)
    assert out.size == n // 2
    bright = pcl.ConditionAnd()
    bright.add_PackedHSIComparison("i", pcl.CompareOp_GT, 50)
    assert pcl.ConditionalRemoval(bright, cloud).filter().size == n
    with pytest.raises(ValueError):
        red.add_PackedRGBComparison("x", pcl.CompareOp_GT, 1)


def test_colour_comparisons_need_a_colour_cloud(grid):
    cond = pcl.ConditionAnd()
    cond.add_PackedRGBComparison("r", pcl.CompareOp_GT, 1)
    with pytest.raises(ValueError, match="colour"):
        _filtered(pcl.PointCloud(grid), cond)


def test_keep_organized_marks_rejected_points(grid):
    cond = pcl.ConditionAnd()
    cond.add_Comparison2("z", pcl.CompareOp_GT, 0.0)
    f = pcl.ConditionalRemoval(cond, pcl.PointCloud(grid))
    f.set_KeepOrganized(True)
    assert f.get_KeepOrganized()
    out = f.filter().to_array()
    assert len(out) == len(grid)
    assert np.all(np.isnan(out[grid[:, 2] <= 0]))
    f.set_UserFilterValue(-7.0)
    out = f.filter().to_array()
    assert np.all(out[grid[:, 2] <= 0] == -7.0)
