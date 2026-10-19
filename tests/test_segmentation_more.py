"""Runtime tests for the second round of segmentation and the extended
sample-consensus models and methods.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _normals_cloud(values):
    arr = np.zeros((len(values), 4), dtype=np.float32)
    arr[:, :3] = values
    return pcl.PointCloud_Normal(arr)


@pytest.fixture
def cylinder():
    """2000 points on a cylinder of radius 0.5 about the z axis, with
    their analytic outward normals."""
    rng = np.random.RandomState(0)
    theta = rng.rand(2000) * 2 * np.pi
    z = rng.rand(2000) * 2.0
    points = np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta), z])
    normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(2000)])
    return (pcl.PointCloud(points.astype(np.float32)),
            _normals_cloud(normals.astype(np.float32)))


@pytest.fixture
def tilted_plane():
    """800 points on the plane z = 0.3 x, plus 200 outliers."""
    rng = np.random.RandomState(1)
    xy = rng.rand(800, 2) * 2 - 1
    plane = np.column_stack([xy, 0.3 * xy[:, 0]])
    noise = rng.rand(200, 3) * 2 - 1
    noise[:, 2] += 3.0
    return pcl.PointCloud(np.vstack([plane, noise]).astype(np.float32))


# --- sample consensus: models ----------------------------------------------

def test_cylinder_model_needs_normals_and_fits(cylinder):
    cloud, normals = cylinder
    model = pcl.SampleConsensusModelCylinder(cloud)
    assert model.model_type == pcl.SACMODEL_CYLINDER
    model.set_InputNormals(normals, 0.1)
    model.set_RadiusLimits(0.1, 1.0)
    model.set_Axis(0, 0, 1)
    model.set_EpsAngle(0.2)
    sac = pcl.RandomSampleConsensus(model)
    sac.set_DistanceThreshold(0.02)
    sac.set_MaxIterations(2000)
    assert sac.computeModel()
    coefficients = sac.get_ModelCoefficients()
    assert len(coefficients) == 7
    assert abs(coefficients[6]) == pytest.approx(0.5, abs=0.05)   # radius
    assert abs(coefficients[5]) == pytest.approx(1.0, abs=0.05)   # axis z
    assert len(sac.get_Inliers()) > 1800


def test_model_settings_are_checked_per_family(cylinder, tilted_plane):
    cloud, normals = cylinder
    plane = pcl.SampleConsensusModelPlane(tilted_plane)
    with pytest.raises(TypeError, match="normals"):
        plane.set_InputNormals(normals)
    with pytest.raises(TypeError, match="axis"):
        plane.set_Axis(0, 0, 1)
    with pytest.raises(TypeError, match="cone"):
        plane.set_MinMaxOpeningAngle(0.1, 0.5)
    with pytest.raises(TypeError, match="registration"):
        plane.set_InputTarget(cloud)
    with pytest.raises(ValueError):
        pcl.SampleConsensusModelCylinder(cloud).set_InputNormals(normals, 2.0)
    with pytest.raises(ValueError):
        plane.set_RadiusLimits(1.0, 0.5)
    with pytest.raises(ValueError, match="not wrapped"):
        pcl.SampleConsensusModel(cloud, 99)


def test_perpendicular_plane_model_constrains_the_normal(tilted_plane):
    """Asking for a plane perpendicular to z within 5 degrees refuses the
    tilted plane (17 degrees), while 30 degrees finds it."""
    strict = pcl.SampleConsensusModelPerpendicularPlane(tilted_plane)
    strict.set_Axis(0, 0, 1)
    strict.set_EpsAngle(np.deg2rad(5))
    sac = pcl.RandomSampleConsensus(strict)
    sac.set_DistanceThreshold(0.02)
    sac.set_MaxIterations(500)
    found = sac.computeModel()
    assert not found or len(sac.get_Inliers()) < 400

    loose = pcl.SampleConsensusModelPerpendicularPlane(tilted_plane)
    loose.set_Axis(0, 0, 1)
    loose.set_EpsAngle(np.deg2rad(30))
    sac = pcl.RandomSampleConsensus(loose)
    sac.set_DistanceThreshold(0.02)
    sac.set_MaxIterations(500)
    assert sac.computeModel()
    assert len(sac.get_Inliers()) > 700


def test_normal_plane_model(tilted_plane):
    estimator = tilted_plane.make_NormalEstimation()
    estimator.set_KSearch(15)
    model = pcl.SampleConsensusModelNormalPlane(tilted_plane)
    model.set_InputNormals(estimator.compute_cloud(), 0.2)
    sac = pcl.SampleConsensus(model, pcl.SAC_MSAC)
    sac.set_DistanceThreshold(0.02)
    assert sac.computeModel()
    a, b, c, d = sac.get_ModelCoefficients()
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    expected = np.array([-0.3, 0, 1]) / np.linalg.norm([-0.3, 0, 1])
    assert abs(normal @ expected) == pytest.approx(1.0, abs=0.02)


def test_registration_model_recovers_a_rigid_transform():
    rng = np.random.RandomState(2)
    source = rng.rand(100, 3).astype(np.float32)
    shift = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    target = source + shift
    model = pcl.SampleConsensusModelRegistration(pcl.PointCloud(source))
    model.set_InputTarget(pcl.PointCloud(target))
    sac = pcl.RandomSampleConsensus(model)
    sac.set_DistanceThreshold(0.01)
    assert sac.computeModel()
    matrix = np.array(sac.get_ModelCoefficients()).reshape(4, 4)
    assert matrix[:3, 3] == pytest.approx(shift, abs=1e-3)
    assert matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-3)
    assert len(sac.get_Inliers()) == 100


# --- sample consensus: methods ---------------------------------------------

@pytest.mark.parametrize("method", [
    pcl.SAC_RANSAC, pcl.SAC_LMEDS, pcl.SAC_MSAC, pcl.SAC_MLESAC,
    pcl.SAC_PROSAC])
def test_every_method_fits_the_plane(tilted_plane, method):
    model = pcl.SampleConsensusModelPlane(tilted_plane)
    sac = pcl.SampleConsensus(model, method)
    assert sac.method == method
    sac.set_DistanceThreshold(0.02)
    sac.set_MaxIterations(500)
    sac.set_Probability(0.99)
    assert sac.get_DistanceThreshold() == pytest.approx(0.02)
    assert sac.computeModel()
    assert len(sac.get_Inliers()) > 600
    assert len(sac.get_ModelCoefficients()) == 4


def test_rmsac_needs_nearly_clean_data(tilted_plane):
    """RMSAC pre-tests a random 10% of the points: one outlier among
    them rejects the candidate, so with 20% outliers it returns a poor
    model (measured: ~25 inliers of 800) -- while a clean plane fits."""
    noisy = pcl.SampleConsensus(
        pcl.SampleConsensusModelPlane(tilted_plane), pcl.SAC_RMSAC)
    noisy.set_DistanceThreshold(0.02)
    noisy.set_MaxIterations(200)
    noisy.computeModel()
    assert len(noisy.get_Inliers()) < 600

    clean = pcl.PointCloud(tilted_plane.to_array()[:800, :3])
    sac = pcl.SampleConsensus(pcl.SampleConsensusModelPlane(clean),
                              pcl.SAC_RMSAC)
    sac.set_DistanceThreshold(0.02)
    sac.set_MaxIterations(200)
    assert sac.computeModel()
    assert len(sac.get_Inliers()) > 700


def test_rransac_is_refused_because_it_never_returns(tilted_plane):
    model = pcl.SampleConsensusModelPlane(tilted_plane)
    with pytest.raises(ValueError, match="never returns"):
        pcl.SampleConsensus(model, pcl.SAC_RRANSAC)
    with pytest.raises(ValueError, match="never returns"):
        tilted_plane.make_segmenter().set_method_type(pcl.SAC_RRANSAC)


def test_random_sample_consensus_is_the_ransac_spelling(tilted_plane):
    sac = pcl.RandomSampleConsensus(pcl.SampleConsensusModelPlane(tilted_plane))
    assert isinstance(sac, pcl.SampleConsensus)
    assert sac.method == pcl.SAC_RANSAC
    sac.set_NumberOfThreads(1)
    assert sac.get_NumberOfThreads() == 1
    with pytest.raises(ValueError, match="SAC_"):
        pcl.SampleConsensus(pcl.SampleConsensusModelPlane(tilted_plane), 42)


# --- ExtractPolygonalPrismData ---------------------------------------------

def test_prism_keeps_the_points_above_the_polygon():
    hull = pcl.PointCloud(np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32))
    points = np.array([
        [0.5, 0.5, 0.2],    # inside, above: kept
        [0.2, 0.8, 0.9],    # inside, above: kept
        [1.5, 0.5, 0.2],    # outside the polygon: dropped
        [0.5, 0.5, 5.0],    # above the height limit: dropped
        [0.5, 0.5, -0.5],   # below the plane: dropped
    ], dtype=np.float32)
    cloud = pcl.PointCloud(points)
    prism = cloud.make_ExtractPolygonalPrismData()
    with pytest.raises(RuntimeError, match="set_InputPlanarHull"):
        prism.segment()
    prism.set_InputPlanarHull(hull)
    prism.set_HeightLimits(0.0, 1.0)
    prism.set_ViewPoint(0.0, 0.0, 10.0)
    assert sorted(prism.segment()) == [0, 1]
    with pytest.raises(ValueError):
        prism.set_InputPlanarHull(pcl.PointCloud(points[:2]))
    with pytest.raises(ValueError):
        prism.set_HeightLimits(1.0, 0.0)


# --- SegmentDifferences ------------------------------------------------------

def test_segment_differences_finds_the_new_points():
    rng = np.random.RandomState(3)
    base = rng.rand(300, 3).astype(np.float32)
    added = rng.rand(20, 3).astype(np.float32) + 5.0
    scene = pcl.PointCloud(np.vstack([base, added]))
    differ = scene.make_SegmentDifferences()
    with pytest.raises(RuntimeError, match="set_TargetCloud"):
        differ.segment()
    differ.set_TargetCloud(pcl.PointCloud(base))
    differ.set_DistanceThreshold(0.001 ** 2)
    assert differ.get_DistanceThreshold() == pytest.approx(1e-6)
    new = differ.segment()
    assert isinstance(new, pcl.PointCloud)
    assert new.size == 20
    assert (new.to_array()[:, :3] >= 5.0).all()
    with pytest.raises(ValueError):
        differ.set_DistanceThreshold(-1.0)


# --- ApproximateProgressiveMorphologicalFilter -----------------------------

def test_approximate_morphological_filter_separates_ground():
    rng = np.random.RandomState(4)
    ground = np.zeros((600, 3), dtype=np.float32)
    ground[:, :2] = rng.rand(600, 2) * 20
    objects = rng.rand(100, 3).astype(np.float32)
    objects[:, :2] *= 20
    objects[:, 2] += 3.0
    cloud = pcl.PointCloud(np.vstack([ground, objects]))
    pmf = cloud.make_ApproximateProgressiveMorphologicalFilter()
    pmf.set_MaxWindowSize(20)
    pmf.set_Slope(1.0)
    pmf.set_InitialDistance(0.5)
    pmf.set_MaxDistance(3.0)
    pmf.set_CellSize(1.0)
    pmf.set_Base(2.0)
    pmf.set_Exponential(True)
    pmf.set_NumberOfThreads(2)
    assert pmf.get_MaxWindowSize() == 20
    assert pmf.get_CellSize() == pytest.approx(1.0)
    assert pmf.get_Base() == pytest.approx(2.0)
    assert pmf.get_Exponential() is True
    ground_idx = pmf.extract()
    assert len(ground_idx) >= 550
    assert all(i < 600 for i in ground_idx)


# --- SeededHueSegmentation ---------------------------------------------------

def test_seeded_hue_grows_the_red_half_only():
    n = 20
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    xyz = np.column_stack([u.ravel(), v.ravel(), np.zeros(n * n)]).astype(np.float32)
    rgb = np.zeros((n * n, 3), dtype=np.uint8)
    left = u.ravel() < 0.5
    rgb[left] = (255, 0, 0)
    rgb[~left] = (0, 0, 255)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    grower = cloud.make_SeededHueSegmentation()
    grower.set_ClusterTolerance(0.1)
    grower.set_DeltaHue(10.0)
    assert grower.get_ClusterTolerance() == pytest.approx(0.1)
    assert grower.get_DeltaHue() == pytest.approx(10.0)
    seed = int(np.flatnonzero(left)[0])
    region = grower.segment([seed])
    assert len(region) > 0
    assert all(left[i] for i in region)
    with pytest.raises(IndexError):
        grower.segment([n * n])
    with pytest.raises(ValueError):
        grower.segment([])


# --- OrganizedMultiPlaneSegmentation ---------------------------------------

def test_organized_multi_plane_finds_both_faces():
    """A 60x60 depth image: the left half a plane at z=1, the right half
    a plane tilted 45 degrees. Two planes, both large."""
    n = 60
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    z = np.where(u < 0.5, 1.0, 1.0 + (u - 0.5))
    grid = np.stack([u, v, z], axis=-1).astype(np.float32)
    cloud = pcl.PointCloud()
    cloud.from_organized_array(grid)
    normals = cloud.make_IntegralImageNormalEstimation()
    normals.set_NormalEstimationMethod(pcl.COVARIANCE_MATRIX)
    normals.set_MaxDepthChangeFactor(0.5)
    normals.set_NormalSmoothingSize(5.0)
    normal_cloud = normals.compute_cloud()

    mps = cloud.make_OrganizedMultiPlaneSegmentation()
    with pytest.raises(RuntimeError, match="set_InputNormals"):
        mps.segment()
    mps.set_InputNormals(normal_cloud)
    mps.set_MinInliers(200)
    mps.set_AngularThreshold(np.deg2rad(3.0))
    mps.set_DistanceThreshold(0.02)
    assert mps.get_MinInliers() == 200
    coefficients, inliers = mps.segment()
    assert len(coefficients) >= 2
    assert len(inliers) == len(coefficients)
    sizes = sorted(len(i) for i in inliers)
    assert sizes[-1] >= 1000 and sizes[-2] >= 1000
    assert all(len(c) == 4 for c in coefficients)
    flat = pcl.PointCloud(grid.reshape(-1, 3))
    with pytest.raises(ValueError, match="organized"):
        mps2 = flat.make_OrganizedMultiPlaneSegmentation()
        mps2.set_InputNormals(normal_cloud)
        mps2.segment()


# --- SupervoxelClustering + LCCP --------------------------------------------

def _two_cubes(rgba):
    rng = np.random.RandomState(5)
    a = rng.rand(1500, 3)
    b = rng.rand(1500, 3) + np.array([3.0, 0.0, 0.0])
    xyz = np.vstack([a, b]).astype(np.float32)
    if not rgba:
        return pcl.PointCloud(xyz), 1500
    rgba = np.full((3000, 4), 255, dtype=np.uint8)
    rgba[:1500, :3] = (255, 0, 0)
    rgba[1500:, :3] = (0, 0, 255)
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(xyz, rgba)
    return cloud, 1500


@pytest.mark.parametrize("rgba", [True, False])
def test_supervoxels_cover_both_cubes(rgba):
    cloud, half = _two_cubes(rgba)
    sv = cloud.make_SupervoxelClustering(0.1, 0.5)
    assert isinstance(sv, pcl.SupervoxelClustering if rgba
                      else pcl.SupervoxelClusteringXYZ)
    sv.set_UseSingleCameraTransform(False)
    sv.set_ColorImportance(0.2)
    sv.set_SpatialImportance(0.4)
    sv.set_NormalImportance(1.0)
    assert sv.get_VoxelResolution() == pytest.approx(0.1)
    assert sv.get_SeedResolution() == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="extract"):
        sv.get_LabeledCloud()

    labels, centroids, normals, adjacency = sv.extract()
    k = len(labels)
    assert k >= 4
    assert centroids.shape == (k, 3) and normals.shape == (k, 4)
    assert adjacency.ndim == 2 and adjacency.shape[1] == 2
    assert set(adjacency.ravel().tolist()) <= set(labels.tolist())
    assert sv.get_MaxLabel() >= labels.max()
    # centroids sit inside one cube or the other, never in between
    assert ((centroids[:, 0] < 1.2) | (centroids[:, 0] > 2.8)).all()

    point_labels = sv.get_LabeledCloud()
    assert point_labels.shape == (cloud.size,)
    assert (point_labels > 0).mean() > 0.9
    # no supervoxel straddles the gap
    for label in labels:
        members = np.flatnonzero(point_labels == label)
        assert (members < half).all() or (members >= half).all()


def test_lccp_keeps_the_two_cubes_apart():
    cloud, half = _two_cubes(True)
    sv = cloud.make_SupervoxelClustering(0.1, 0.5)
    sv.set_UseSingleCameraTransform(False)
    with pytest.raises(RuntimeError, match="extract"):
        sv.lccp()
    labels, _, _, _ = sv.extract()
    point_labels, mapping = sv.lccp(concavity_tolerance=10.0,
                                    smoothness_check=True,
                                    min_segment_size=0)
    assert point_labels.shape == (cloud.size,)
    assert set(mapping) == set(labels.tolist())
    left = set(point_labels[:half][point_labels[:half] > 0].tolist())
    right = set(point_labels[half:][point_labels[half:] > 0].tolist())
    assert left and right and not (left & right)


def test_supervoxel_resolutions_are_checked():
    cloud, _ = _two_cubes(False)
    with pytest.raises(ValueError):
        cloud.make_SupervoxelClustering(0.0, 0.5)
    sv = cloud.make_SupervoxelClustering(0.1, 0.5)
    with pytest.raises(ValueError):
        sv.set_SeedResolution(-1.0)
    with pytest.raises(RuntimeError):
        pcl.SupervoxelClusteringXYZ(0.1, 0.5).extract()
