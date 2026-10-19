"""Runtime tests for the second round of features and keypoints.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def sphere():
    """1500 points on the unit sphere."""
    rng = np.random.RandomState(0)
    theta = rng.rand(1500) * 2 * np.pi
    phi = np.arccos(2 * rng.rand(1500) - 1)
    return pcl.PointCloud(np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ]).astype(np.float32))


@pytest.fixture
def sphere_normals(sphere):
    values = np.zeros((sphere.size, 4), dtype=np.float32)
    values[:, :3] = sphere.to_array()
    return pcl.PointCloud_Normal(values)


@pytest.fixture
def grid():
    """A flat 30x30 grid on the unit square."""
    u, v = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    return pcl.PointCloud(np.column_stack([
        u.ravel(), v.ravel(), np.zeros(900)]).astype(np.float32))


@pytest.fixture
def grid_normals(grid):
    values = np.zeros((grid.size, 4), dtype=np.float32)
    values[:, 2] = 1.0
    return pcl.PointCloud_Normal(values)


@pytest.fixture
def colour_image():
    """A 40x40 grid at z=0, dark with a bright 10x10 square in the
    middle: intensity structure for SIFT and SUSAN."""
    n = 40
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    xyz = np.column_stack([u.ravel(), v.ravel(), np.zeros(n * n)])
    rgb = np.full((n * n, 3), 30, dtype=np.uint8)
    inside = (np.abs(u.ravel() - 0.5) < 0.13) & (np.abs(v.ravel() - 0.5) < 0.13)
    rgb[inside] = 230
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz.astype(np.float32), rgb)
    return cloud


def _finite_nonneg(arr):
    return np.isfinite(arr).all() and (arr >= 0).all()


# --- per-point descriptors ----------------------------------------------

def test_principal_curvatures_on_a_sphere(sphere, sphere_normals):
    est = sphere.make_PrincipalCurvaturesEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_KSearch(20)
    out = est.compute()
    assert out.shape == (sphere.size, 5)
    assert np.isfinite(out).all()
    # pc1 >= pc2 >= 0, and the principal direction is a unit vector
    assert (out[:, 3] >= out[:, 4] - 1e-6).all()
    assert (out[:, 4] >= -1e-6).all()
    assert np.linalg.norm(out[:, :3], axis=1) == pytest.approx(
        np.ones(sphere.size), abs=1e-3)
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        sphere.make_PrincipalCurvaturesEstimation().compute()


def test_pfh_histograms_sum_to_100(sphere, sphere_normals):
    est = sphere.make_PFHEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_KSearch(10)
    est.set_UseInternalCache(True)
    est.set_MaximumCacheSize(1000)
    assert est.get_UseInternalCache() is True
    assert est.get_MaximumCacheSize() == 1000
    out = est.compute()
    assert out.shape == (sphere.size, 125)
    assert _finite_nonneg(out)
    assert out.sum(axis=1) == pytest.approx(np.full(sphere.size, 100.0), abs=1e-2)


def test_rsd_recovers_the_sphere_radius(sphere, sphere_normals):
    est = sphere.make_RSDEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_RadiusSearch(0.4)
    est.set_PlaneRadius(5.0)       # the default 0.2 would clamp a unit sphere
    est.set_NrSubdivisions(5)
    assert est.get_PlaneRadius() == pytest.approx(5.0)
    assert est.get_NrSubdivisions() == 5
    out = est.compute()
    assert out.shape == (sphere.size, 2)
    assert np.isfinite(out).all()
    assert (out[:, 0] <= out[:, 1] + 1e-6).all()
    # RSD reports a unit sphere as curved with radii of order 1 (the
    # estimate depends on the search radius: ~1.1 at 0.2, ~1.7 at 0.4 on
    # PCL 1.14), nowhere near the planar clamp
    assert 0.3 < np.median(out[:, 0]) < 2.0
    assert 0.5 < np.median(out[:, 1]) < 2.5
    assert not hasattr(est, "set_KSearch")


def test_rsd_clamps_to_the_plane_radius(grid, grid_normals):
    est = grid.make_RSDEstimation()
    est.set_InputNormals(grid_normals)
    est.set_RadiusSearch(0.15)
    out = est.compute()
    # PCL clamps at 1.1 x the plane radius (0.22 for the default 0.2)
    assert out[:, 1].max() <= 0.2 * 1.1 + 1e-5
    assert out[:, 1].max() > 0.2


def test_spin_images(sphere, sphere_normals):
    est = sphere.make_SpinImageEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_RadiusSearch(0.5)
    est.set_SupportAngle(0.0)
    est.set_MinPointCountInNeighbourhood(3)
    out = est.compute()
    assert out.shape == (sphere.size, 153)
    assert _finite_nonneg(out)
    assert out.sum() > 0
    with pytest.raises(ValueError):
        est.set_ImageWidth(16)


def test_moment_invariants(sphere):
    est = sphere.make_MomentInvariantsEstimation()
    est.set_KSearch(15)
    out = est.compute()
    assert out.shape == (sphere.size, 3)
    assert np.isfinite(out).all()


def test_normal_estimation_omp_agrees_with_the_serial_one(sphere):
    serial = sphere.make_NormalEstimation()
    serial.set_KSearch(20)
    parallel = sphere.make_NormalEstimationOMP()
    parallel.set_KSearch(20)
    parallel.set_NumberOfThreads(2)
    assert parallel.get_KSearch() == 20
    a = serial.compute()
    b = parallel.compute()
    assert b.shape == a.shape
    # normals may be flipped point-by-point only if the viewpoint logic
    # differs; it does not, so they match outright
    assert b == pytest.approx(a, abs=1e-4, nan_ok=True)
    cloud = parallel.compute_cloud()
    assert isinstance(cloud, pcl.PointCloud_Normal)
    assert cloud.size == sphere.size


# --- global descriptors --------------------------------------------------

def test_gasd_describes_the_whole_cloud(sphere, grid):
    est = sphere.make_GASDEstimation()
    est.set_ShapeHistsInterpMethod(pcl.INTERP_TRILINEAR)
    ball = est.compute()
    assert ball.shape == (512,)
    assert _finite_nonneg(ball) and ball.sum() > 0
    flat = grid.make_GASDEstimation().compute()
    assert not np.allclose(ball, flat)
    with pytest.raises(ValueError):
        est.set_ShapeHistsInterpMethod(7)
    assert (pcl.INTERP_NONE, pcl.INTERP_TRILINEAR, pcl.INTERP_QUADRILINEAR) == (0, 1, 2)


def test_gasd_colour(colour_image):
    est = colour_image.make_GASDColorEstimation()
    est.set_ColorHistsInterpMethod(pcl.INTERP_NONE)
    out = est.compute()
    assert out.shape == (984,)
    assert _finite_nonneg(out) and out[512:].sum() > 0


def test_grsd(sphere, sphere_normals):
    est = sphere.make_GRSDEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_RadiusSearch(0.3)
    assert est.get_RadiusSearch() == pytest.approx(0.3)
    out = est.compute()
    assert out.shape == (21,)
    assert _finite_nonneg(out) and out.sum() > 0
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        sphere.make_GRSDEstimation().compute()


def test_our_cvfh(grid, grid_normals):
    est = grid.make_OURCVFHEstimation()
    est.set_InputNormals(grid_normals)
    est.set_ClusterTolerance(0.1)
    est.set_RadiusNormals(0.15)
    est.set_MinPoints(50)
    est.set_RefineClusters(1.0)
    est.set_AxisRatio(0.8)
    est.set_MinAxisValue(0.925)
    out = est.compute()
    assert out.ndim == 2 and out.shape[1] == 308 and out.shape[0] >= 1
    assert _finite_nonneg(out) and out.sum() > 0
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        grid.make_OURCVFHEstimation().compute()


def test_crh(sphere, sphere_normals):
    est = sphere.make_CRHEstimation()
    est.set_InputNormals(sphere_normals)
    est.set_ViewPoint(0.0, 0.0, 5.0)
    out = est.compute()
    assert out.shape == (90,)
    assert np.isfinite(out).all() and out.sum() > 0
    est.set_Centroid(0.0, 0.0, 0.0)
    explicit = est.compute()
    assert explicit.shape == (90,)
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        sphere.make_CRHEstimation().compute()


# --- keypoints -------------------------------------------------------------

def test_iss_keypoints_on_a_sphere(sphere):
    iss = sphere.make_ISSKeypoint3D()
    iss.set_SalientRadius(0.3)
    iss.set_NonMaxRadius(0.2)
    iss.set_Threshold21(0.975)
    iss.set_Threshold32(0.975)
    iss.set_MinNeighbors(5)
    iss.set_NumberOfThreads(2)
    keypoints = iss.compute()
    assert isinstance(keypoints, pcl.PointCloud)
    assert 0 < keypoints.size < sphere.size
    with pytest.raises(ValueError):
        iss.set_SalientRadius(0)


def test_sift_finds_the_bright_square(colour_image):
    sift = colour_image.make_SIFTKeypoint()
    with pytest.raises(RuntimeError, match="set_Scales"):
        sift.compute()
    sift.set_Scales(0.03, 3, 3)
    sift.set_MinimumContrast(1.0)
    out = sift.compute()
    assert out.ndim == 2 and out.shape[1] == 7
    assert out.shape[0] > 0
    assert (out[:, 3] > 0).all()                  # every keypoint has a scale
    # the strongest response sits at the square's edge or centre
    best = out[np.argmax(out[:, 5])]
    assert np.abs(best[:2] - 0.5).max() < 0.2
    with pytest.raises(ValueError):
        sift.set_Scales(0.0, 3, 3)


def test_susan_finds_colour_corners_once_the_normal_test_is_off(colour_image):
    """SUSAN's similarity test is intensity OR normal: on a flat image
    every neighbour passes the normal test, so colour corners only show
    up with the angular threshold disabled (negative). Non-maximum
    suppression discards everything here because the responses tie."""
    susan = colour_image.make_SUSANKeypoint()
    susan.set_Radius(0.08)
    susan.set_DistanceThreshold(0.001)
    susan.set_AngularThreshold(0.0001)
    susan.set_IntensityThreshold(7.0)
    susan.set_NonMaxSupression(False)
    susan.set_GeometricValidation(False)
    susan.set_NumberOfThreads(2)
    assert susan.compute().size == 0          # the normal test passes everywhere
    susan.set_AngularThreshold(-1.0)
    keypoints = susan.compute()
    assert isinstance(keypoints, pcl.PointCloud)
    assert 0 < keypoints.size < colour_image.size
    # the corners of the bright square are where the intensity changes
    # in two directions: every keypoint lies near its outline
    xy = keypoints.to_array()[:, :2]
    assert (np.abs(np.abs(xy - 0.5) - 0.13).min(axis=1) < 0.08).all()
    with pytest.raises(ValueError):
        susan.set_Radius(0)
