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
