"""Runtime tests for normal-based segmentation, ground extraction and
conditional removal.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (`make_segmenter_normals`,
`make_ConditionAnd` / `make_ConditionalRemoval`).
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

CYLINDER_RADIUS = 0.5


@pytest.fixture
def cylinder():
    """800 points on a cylinder of radius 0.5 about the z axis."""
    rng = np.random.RandomState(0)
    theta = rng.rand(800) * 2 * np.pi
    height = rng.rand(800) * 2.0
    points = np.column_stack([
        np.cos(theta) * CYLINDER_RADIUS,
        np.sin(theta) * CYLINDER_RADIUS,
        height,
    ]).astype(np.float32)
    return pcl.PointCloud(points)


@pytest.fixture
def terrain():
    """600 ground points at z=0 under 100 objects 3 units up."""
    rng = np.random.RandomState(2)
    ground = np.zeros((600, 3), dtype=np.float32)
    ground[:, :2] = rng.rand(600, 2) * 20
    objects = rng.rand(100, 3).astype(np.float32)
    objects[:, :2] *= 20
    objects[:, 2] += 3.0
    return pcl.PointCloud(np.vstack([ground, objects]))


# --- normals as a cloud -----------------------------------------------

def test_normal_estimation_can_return_a_cloud(cylinder):
    estimator = cylinder.make_NormalEstimation()
    estimator.set_KSearch(20)
    normals = estimator.compute_cloud()
    assert isinstance(normals, pcl.PointCloud_Normal)
    assert normals.size == cylinder.size
    assert normals.to_array().shape == (cylinder.size, 4)


def test_normal_cloud_matches_the_array_form(cylinder):
    estimator = cylinder.make_NormalEstimation()
    estimator.set_KSearch(20)
    from_cloud = estimator.compute_cloud().to_array()
    from_array = estimator.compute()
    assert from_cloud == pytest.approx(from_array, abs=1e-5, nan_ok=True)


def test_normal_cloud_roundtrips_through_an_array():
    values = np.random.RandomState(3).rand(50, 4).astype(np.float32)
    cloud = pcl.PointCloud_Normal(values)
    assert cloud.size == 50
    assert cloud.to_array() == pytest.approx(values, abs=1e-6)
    assert cloud[0] == pytest.approx(tuple(values[0]), abs=1e-6)


# --- segmentation with normals ----------------------------------------

def test_segmenter_normals_finds_the_cylinder(cylinder):
    estimator = cylinder.make_NormalEstimation()
    estimator.set_KSearch(20)

    seg = cylinder.make_segmenter_normals()
    seg.set_optimize_coefficients(True)
    seg.set_InputNormals(estimator.compute_cloud())
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_normal_distance_weight(0.1)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(0.05)
    seg.set_radius_limits(0.0, 1.0)
    indices, coefficients = seg.segment()

    assert len(indices) > cylinder.size * 0.9
    # Cylinder coefficients are point(3), axis(3), radius.
    assert len(coefficients) == 7
    assert coefficients[6] == pytest.approx(CYLINDER_RADIUS, abs=0.05)


def test_segmenter_normals_axis_and_eps_angle_are_accepted(cylinder):
    estimator = cylinder.make_NormalEstimation()
    estimator.set_KSearch(20)
    seg = cylinder.make_segmenter_normals()
    seg.set_InputNormals(estimator.compute_cloud())
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.05)
    seg.set_radius_limits(0.0, 1.0)
    seg.set_axis(0.0, 0.0, 1.0)
    seg.set_eps_angle(0.1)
    indices, _ = seg.segment()
    assert len(indices) > 0


def test_plain_segmenter_gained_axis_and_eps_angle():
    """SACSegmentation's Eigen-typed setAxis, reached through the shim."""
    rng = np.random.RandomState(4)
    points = np.zeros((400, 3), dtype=np.float32)
    points[:, :2] = rng.rand(400, 2)
    cloud = pcl.PointCloud(points)

    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PERPENDICULAR_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    seg.set_axis(0.0, 0.0, 1.0)
    seg.set_eps_angle(0.1)
    indices, _ = seg.segment()
    assert len(indices) > 300


# --- ground extraction -------------------------------------------------

def test_progressive_morphological_filter_finds_the_ground(terrain):
    pmf = terrain.make_ProgressiveMorphologicalFilter()
    pmf.set_MaxWindowSize(20)
    pmf.set_Slope(1.0)
    pmf.set_InitialDistance(0.5)
    pmf.set_MaxDistance(3.0)
    ground = pmf.extract()

    assert ground
    # The planted ground is the first 600 points; nothing above may leak in.
    assert max(ground) < 600


def test_progressive_morphological_filter_feeds_extract_indices(terrain):
    pmf = terrain.make_ProgressiveMorphologicalFilter()
    pmf.set_MaxWindowSize(20)
    pmf.set_Slope(1.0)
    pmf.set_InitialDistance(0.5)
    pmf.set_MaxDistance(3.0)
    ground = pmf.extract()

    extract = terrain.make_ExtractIndices()
    extract.set_indices(ground)
    assert extract.filter().size == len(ground)
    extract.set_negative(True)
    assert extract.filter().size == terrain.size - len(ground)


def test_progressive_morphological_settings_roundtrip(terrain):
    pmf = terrain.make_ProgressiveMorphologicalFilter()
    pmf.set_MaxWindowSize(33)
    pmf.set_Slope(1.5)
    pmf.set_InitialDistance(0.25)
    pmf.set_MaxDistance(4.0)
    pmf.set_CellSize(1.5)
    assert pmf.get_MaxWindowSize() == 33
    assert pmf.get_Slope() == pytest.approx(1.5)
    assert pmf.get_InitialDistance() == pytest.approx(0.25)
    assert pmf.get_MaxDistance() == pytest.approx(4.0)
    assert pmf.get_CellSize() == pytest.approx(1.5)


# --- conditional removal -----------------------------------------------

def test_conditional_removal_keeps_only_the_range(cylinder):
    condition = cylinder.make_ConditionAnd()
    condition.add_Comparison2("z", pcl.CompareOp_GT, 0.5)
    condition.add_Comparison2("z", pcl.CompareOp_LT, 1.5)

    kept = cylinder.make_ConditionalRemoval(condition).filter().to_array()
    assert len(kept) > 0
    assert kept[:, 2].min() > 0.5
    assert kept[:, 2].max() < 1.5


def test_conditional_removal_condition_can_be_set_later(cylinder):
    condition = cylinder.make_ConditionAnd()
    condition.add_Comparison2("z", pcl.CompareOp_GT, 1.0)

    remover = pcl.ConditionalRemoval()
    remover.set_Condition(condition)
    remover.set_InputCloud(cylinder)
    assert remover.filter().to_array()[:, 2].min() > 1.0


def test_conditional_removal_add_comparison_chains(cylinder):
    condition = cylinder.make_ConditionAnd()
    assert condition.add_Comparison2("z", pcl.CompareOp_GE, 0.0) is condition


def test_compare_op_constants_match_pcls_enum():
    """Taken from the header through the shim, not copied into Python."""
    assert (pcl.CompareOp_GT, pcl.CompareOp_GE, pcl.CompareOp_LT,
            pcl.CompareOp_LE, pcl.CompareOp_EQ) == (0, 1, 2, 3, 4)


def test_conditional_removal_keep_organized_roundtrip(cylinder):
    condition = cylinder.make_ConditionAnd()
    condition.add_Comparison2("z", pcl.CompareOp_GT, 0.0)
    remover = cylinder.make_ConditionalRemoval(condition)
    remover.set_KeepOrganized(True)
    assert remover.get_KeepOrganized() is True


# --- model constants ---------------------------------------------------

def test_every_sac_model_constant_is_exported():
    """python-pcl's tests reference all of these by name."""
    for name in ("SACMODEL_PLANE", "SACMODEL_LINE", "SACMODEL_CIRCLE2D",
                 "SACMODEL_CIRCLE3D", "SACMODEL_SPHERE", "SACMODEL_CYLINDER",
                 "SACMODEL_CONE", "SACMODEL_TORUS", "SACMODEL_PARALLEL_LINE",
                 "SACMODEL_PERPENDICULAR_PLANE", "SACMODEL_PARALLEL_LINES",
                 "SACMODEL_NORMAL_PLANE", "SACMODEL_NORMAL_SPHERE",
                 "SACMODEL_REGISTRATION", "SACMODEL_REGISTRATION_2D",
                 "SACMODEL_PARALLEL_PLANE",
                 "SACMODEL_NORMAL_PARALLEL_PLANE", "SACMODEL_STICK",
                 "SACMODEL_ELLIPSE3D"):
        assert isinstance(getattr(pcl, name), int), name
