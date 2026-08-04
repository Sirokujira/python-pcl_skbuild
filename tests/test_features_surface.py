"""Runtime tests for the feature and surface wrappers.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (pcl/pxi/Features/,
pcl/pxi/Surface/), except that normal outputs come back as numpy arrays
rather than a Normal-cloud object — see src/pcl/_features.pyx.
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def cloud():
    return pcl.PointCloud(
        np.random.RandomState(1).rand(1000, 3).astype(np.float32))


@pytest.fixture
def plane():
    """400 points on z=0, whose normals must all point along z."""
    rng = np.random.RandomState(4)
    points = np.zeros((400, 3), dtype=np.float32)
    points[:, :2] = rng.rand(400, 2)
    return pcl.PointCloud(points)


# --- normal estimation -----------------------------------------------

def test_normal_estimation_returns_unit_normals(cloud):
    estimator = cloud.make_NormalEstimation()
    estimator.set_KSearch(10)
    normals = estimator.compute()
    assert normals.shape == (cloud.size, 4)
    assert normals.dtype == np.float32
    lengths = np.linalg.norm(normals[:, :3], axis=1)
    assert lengths == pytest.approx(np.ones(cloud.size), abs=1e-3)


def test_normal_estimation_on_a_plane_points_along_z(plane):
    estimator = plane.make_NormalEstimation()
    estimator.set_KSearch(10)
    normals = estimator.compute()
    # Sign depends on the view point; the axis does not.
    assert np.abs(normals[:, 2]) == pytest.approx(
        np.ones(plane.size), abs=1e-3)
    assert np.abs(normals[:, :2]).max() < 1e-3
    # A perfect plane has no curvature.
    assert normals[:, 3].max() < 1e-3


def test_normal_estimation_radius_search(cloud):
    estimator = cloud.make_NormalEstimation()
    estimator.set_RadiusSearch(0.2)
    assert estimator.get_RadiusSearch() == pytest.approx(0.2)
    assert estimator.compute().shape == (cloud.size, 4)


def test_normal_estimation_ksearch_roundtrip(cloud):
    estimator = cloud.make_NormalEstimation()
    estimator.set_KSearch(12)
    assert estimator.get_KSearch() == 12


def test_normal_estimation_view_point_flips_the_sign(plane):
    signs = []
    for z in (10.0, -10.0):
        estimator = plane.make_NormalEstimation()
        estimator.set_KSearch(10)
        estimator.set_ViewPoint(0.0, 0.0, z)
        signs.append(np.sign(estimator.compute()[0, 2]))
    assert signs[0] == -signs[1]


# --- moment of inertia -----------------------------------------------

def test_moment_of_inertia_descriptors(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.compute()
    assert len(estimator.get_MomentOfInertia()) > 0
    assert len(estimator.get_Eccentricity()) == len(
        estimator.get_MomentOfInertia())


def test_moment_of_inertia_mass_center_is_the_centroid(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.compute()
    center = estimator.get_MassCenter()
    assert center.shape == (3,)
    assert center == pytest.approx(cloud.to_array().mean(axis=0), abs=1e-3)


def test_moment_of_inertia_aabb_matches_the_cloud_bounds(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.compute()
    minimum, maximum = estimator.get_AABB()
    points = cloud.to_array()
    assert minimum == pytest.approx(points.min(axis=0), abs=1e-5)
    assert maximum == pytest.approx(points.max(axis=0), abs=1e-5)


def test_moment_of_inertia_obb_shapes(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.compute()
    minimum, maximum, position, rotation = estimator.get_OBB()
    assert minimum.shape == maximum.shape == position.shape == (3,)
    assert rotation.shape == (3, 3)
    # A rotation matrix: orthonormal, determinant +/-1.
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-3)
    assert abs(abs(np.linalg.det(rotation)) - 1.0) < 1e-3


def test_moment_of_inertia_eigen_decomposition(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.compute()
    major, middle, minor = estimator.get_EigenValues()
    assert major >= middle >= minor
    vectors = estimator.get_EigenVectors()
    assert len(vectors) == 3
    for vector in vectors:
        assert vector.shape == (3,)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)


def test_moment_of_inertia_settings(cloud):
    estimator = cloud.make_MomentOfInertiaEstimation()
    estimator.set_AngleStep(15.0)
    estimator.set_NormalizePointMassFlag(True)
    estimator.set_PointMass(0.5)
    assert estimator.get_AngleStep() == pytest.approx(15.0)
    estimator.compute()


# --- moving least squares --------------------------------------------

def test_mls_keeps_the_cloud_size(cloud):
    mls = cloud.make_moving_least_squares()
    mls.set_search_radius(0.3)
    mls.set_polynomial_order(2)
    smoothed = mls.process()
    assert isinstance(smoothed, pcl.PointCloud)
    assert smoothed.size > 0


def test_mls_smooths_a_noisy_plane():
    """Points scattered around z=0 must end up closer to it."""
    rng = np.random.RandomState(5)
    points = np.zeros((500, 3), dtype=np.float32)
    points[:, :2] = rng.rand(500, 2)
    points[:, 2] = rng.normal(0.0, 0.01, 500)
    cloud = pcl.PointCloud(points)

    mls = cloud.make_moving_least_squares()
    mls.set_search_radius(0.2)
    mls.set_polynomial_order(2)
    smoothed = mls.process().to_array()
    assert np.abs(smoothed[:, 2]).mean() < np.abs(points[:, 2]).mean()


def test_mls_with_normals_returns_both(cloud):
    mls = cloud.make_moving_least_squares()
    mls.set_search_radius(0.3)
    mls.set_Compute_Normals(True)
    smoothed, normals = mls.process_with_normals()
    assert isinstance(smoothed, pcl.PointCloud)
    assert normals.shape == (smoothed.size, 4)
    assert normals.dtype == np.float32


def test_mls_polynomial_fit_alias(cloud):
    mls = cloud.make_moving_least_squares()
    mls.set_polynomial_fit(True)
    assert mls.get_polynomial_order() == 2
    mls.set_polynomial_fit(False)
    assert mls.get_polynomial_order() == 1


# --- hulls ------------------------------------------------------------

def test_convex_hull_is_a_subset_with_area_and_volume(cloud):
    hull = cloud.make_ConvexHull()
    hull.set_Dimension(3)
    hull.set_ComputeAreaVolume(True)
    points = hull.reconstruct()
    assert 0 < points.size < cloud.size
    assert hull.get_TotalArea() > 0
    assert hull.get_TotalVolume() > 0


def test_convex_hull_of_a_unit_cube_has_the_right_volume():
    corners = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
    cloud = pcl.PointCloud(corners.astype(np.float32))
    hull = cloud.make_ConvexHull()
    hull.set_Dimension(3)
    hull.set_ComputeAreaVolume(True)
    hull.reconstruct()
    assert hull.get_TotalVolume() == pytest.approx(1.0, abs=1e-4)
    assert hull.get_TotalArea() == pytest.approx(6.0, abs=1e-4)


def test_concave_hull_needs_an_alpha(cloud):
    hull = cloud.make_ConcaveHull()
    hull.set_Alpha(0.5)
    hull.set_Dimension(3)
    assert hull.get_Alpha() == pytest.approx(0.5)
    assert hull.reconstruct().size > 0


def test_concave_hull_alpha_controls_detail(cloud):
    sizes = []
    for alpha in (0.1, 1.0):
        hull = cloud.make_ConcaveHull()
        hull.set_Dimension(3)
        hull.set_Alpha(alpha)
        sizes.append(hull.reconstruct().size)
    # A tighter alpha follows the surface more closely, so it keeps more
    # points than one large enough to approach the convex hull.
    assert sizes[0] >= sizes[1]


# --- difference of normals -------------------------------------------

@pytest.fixture
def plane_with_a_rough_patch():
    """A flat grid with one bumpy square: structure at the small scale
    only, which is exactly what DoN is meant to isolate."""
    rng = np.random.RandomState(0)
    grid = np.mgrid[0:40, 0:40].reshape(2, -1).T * 0.05
    points = np.zeros((grid.shape[0], 3), dtype=np.float32)
    points[:, :2] = grid
    rough = (np.abs(grid[:, 0] - 1.0) < 0.3) & (np.abs(grid[:, 1] - 1.0) < 0.3)
    points[rough, 2] = rng.uniform(-0.05, 0.05, rough.sum())
    return pcl.PointCloud(points), rough


def _don_for(cloud, small_radius=0.1, large_radius=0.5):
    small = cloud.make_NormalEstimation()
    small.set_RadiusSearch(small_radius)
    large = cloud.make_NormalEstimation()
    large.set_RadiusSearch(large_radius)

    don = cloud.make_DifferenceOfNormalsEstimation()
    don.set_NormalScaleSmall(small.compute_cloud())
    don.set_NormalScaleLarge(large.compute_cloud())
    return don


def test_don_returns_one_row_per_point(plane_with_a_rough_patch):
    cloud, _ = plane_with_a_rough_patch
    result = _don_for(cloud).compute()
    assert result.shape == (cloud.size, 4)
    assert result.dtype == np.float32


def test_don_output_is_finite(plane_with_a_rough_patch):
    """PCL zeroes any non-finite difference itself, so nothing here has
    to filter NaNs out."""
    cloud, _ = plane_with_a_rough_patch
    assert np.isfinite(_don_for(cloud).compute()).all()


def test_don_responds_where_the_surface_has_structure(
        plane_with_a_rough_patch):
    cloud, rough = plane_with_a_rough_patch
    magnitude = _don_for(cloud).compute()[:, 3]
    assert magnitude[rough].mean() > magnitude[~rough].mean()


def test_don_is_zero_on_a_plane(plane):
    """Both scales see the same normal, so the difference vanishes —
    the property that makes a threshold on it meaningful."""
    magnitude = _don_for(plane, 0.1, 0.4).compute()[:, 3]
    assert magnitude.max() == pytest.approx(0.0, abs=1e-5)


def test_don_without_input_reports_it():
    with pytest.raises(RuntimeError, match="set_InputCloud"):
        pcl.DifferenceOfNormalsEstimation().compute()


def test_don_without_both_scales_reports_it(plane):
    don = plane.make_DifferenceOfNormalsEstimation()
    with pytest.raises(RuntimeError, match="NormalScaleSmall"):
        don.compute()

    small = plane.make_NormalEstimation()
    small.set_RadiusSearch(0.1)
    don.set_NormalScaleSmall(small.compute_cloud())
    with pytest.raises(RuntimeError, match="NormalScaleLarge"):
        don.compute()


def test_don_rejects_mismatched_normal_clouds(plane):
    """PCL requires one normal per input point; without the check its
    computeFeature() writes past the end of the output cloud."""
    large = plane.make_NormalEstimation()
    large.set_RadiusSearch(0.4)

    don = plane.make_DifferenceOfNormalsEstimation()
    don.set_NormalScaleSmall(
        pcl.PointCloud_Normal(np.zeros((5, 4), dtype=np.float32)))
    don.set_NormalScaleLarge(large.compute_cloud())
    with pytest.raises(ValueError, match="one normal per input point"):
        don.compute()
