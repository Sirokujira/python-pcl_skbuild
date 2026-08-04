"""Runtime tests for the index/geometry filters and Harris keypoints.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

These are the pieces that close the segmentation workflow: PCL hands
back indices and coefficients, and ExtractIndices / ProjectInliers turn
them into clouds.
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

PLANE_POINTS = 500
OUTLIER_POINTS = 50


@pytest.fixture
def plane_with_outliers():
    """500 points on z=0 followed by 50 well above it, in that order —
    the index ranges are what the assertions check."""
    rng = np.random.RandomState(1)
    plane = np.zeros((PLANE_POINTS, 3), dtype=np.float32)
    plane[:, :2] = rng.rand(PLANE_POINTS, 2)
    outliers = rng.rand(OUTLIER_POINTS, 3).astype(np.float32)
    outliers[:, 2] += 2.0
    return pcl.PointCloud(np.vstack([plane, outliers]))


@pytest.fixture
def segmented(plane_with_outliers):
    seg = plane_with_outliers.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    return seg.segment()


# --- extract indices --------------------------------------------------

def test_extract_indices_pulls_out_the_segmented_plane(
        plane_with_outliers, segmented):
    indices, _ = segmented
    extract = plane_with_outliers.make_ExtractIndices()
    extract.set_indices(indices)
    assert extract.filter().size == PLANE_POINTS


def test_extract_indices_negative_gives_the_complement(
        plane_with_outliers, segmented):
    indices, _ = segmented
    extract = plane_with_outliers.make_ExtractIndices()
    extract.set_indices(indices)
    extract.set_negative(True)
    assert extract.get_negative() is True
    assert extract.filter().size == OUTLIER_POINTS


def test_extract_indices_halves_sum_to_the_whole(
        plane_with_outliers, segmented):
    indices, _ = segmented
    sizes = []
    for negative in (False, True):
        extract = plane_with_outliers.make_ExtractIndices()
        extract.set_indices(indices)
        extract.set_negative(negative)
        sizes.append(extract.filter().size)
    assert sum(sizes) == plane_with_outliers.size


def test_extract_indices_accepts_any_iterable(plane_with_outliers):
    extract = plane_with_outliers.make_ExtractIndices()
    extract.set_indices(range(10))
    assert extract.filter().size == 10


def test_extract_indices_of_nothing_is_empty(plane_with_outliers):
    extract = plane_with_outliers.make_ExtractIndices()
    extract.set_indices([])
    assert extract.filter().size == 0


# --- project inliers --------------------------------------------------

def test_project_inliers_flattens_onto_the_plane(
        plane_with_outliers, segmented):
    _, coefficients = segmented
    project = plane_with_outliers.make_ProjectInliers()
    project.set_model_type(pcl.SACMODEL_PLANE)
    project.set_model_coefficients(coefficients)
    projected = project.filter()

    assert projected.size == plane_with_outliers.size
    # Every point now lies on the fitted plane, outliers included.
    assert np.abs(projected.to_array()[:, 2]).max() < 1e-4


def test_project_inliers_model_type_roundtrip(plane_with_outliers):
    project = plane_with_outliers.make_ProjectInliers()
    project.set_model_type(pcl.SACMODEL_PLANE)
    assert project.get_model_type() == pcl.SACMODEL_PLANE


# --- crop box ---------------------------------------------------------

def test_crop_box_keeps_only_the_box(plane_with_outliers):
    crop = plane_with_outliers.make_cropbox()
    crop.set_Min(0.0, 0.0, -1.0)
    crop.set_Max(0.5, 0.5, 1.0)
    kept = crop.filter().to_array()
    assert kept.size > 0
    assert (kept[:, 0] <= 0.5 + 1e-6).all()
    assert (kept[:, 1] <= 0.5 + 1e-6).all()
    assert (np.abs(kept[:, 2]) <= 1.0 + 1e-6).all()


def test_crop_box_negative_is_the_complement(plane_with_outliers):
    sizes = []
    for negative in (False, True):
        crop = plane_with_outliers.make_cropbox()
        crop.set_Min(0.0, 0.0, -1.0)
        crop.set_Max(0.5, 0.5, 1.0)
        crop.set_negative(negative)
        sizes.append(crop.filter().size)
    assert sum(sizes) == plane_with_outliers.size


def test_crop_box_translation_moves_the_box(plane_with_outliers):
    """Shifting the box off the cloud must empty it."""
    crop = plane_with_outliers.make_cropbox()
    crop.set_Min(0.0, 0.0, -1.0)
    crop.set_Max(0.5, 0.5, 1.0)
    crop.set_Translation(100.0, 100.0, 100.0)
    assert crop.filter().size == 0


def test_crop_box_rotation_is_accepted(plane_with_outliers):
    crop = plane_with_outliers.make_cropbox()
    crop.set_Min(-1.0, -1.0, -1.0)
    crop.set_Max(1.0, 1.0, 1.0)
    crop.set_Rotation(0.0, 0.0, np.pi / 4)
    assert crop.filter().size > 0


# --- sampling ---------------------------------------------------------

def test_random_sample_returns_the_requested_count(plane_with_outliers):
    sample = plane_with_outliers.make_RandomSample()
    sample.set_sample(100)
    assert sample.get_sample() == 100
    assert sample.filter().size == 100


def test_random_sample_is_reproducible_with_a_seed(plane_with_outliers):
    results = []
    for _ in range(2):
        sample = plane_with_outliers.make_RandomSample()
        sample.set_sample(50)
        sample.set_seed(42)
        results.append(sample.filter().to_array())
    assert results[0] == pytest.approx(results[1])


def test_random_sample_seed_roundtrip(plane_with_outliers):
    sample = plane_with_outliers.make_RandomSample()
    sample.set_seed(7)
    assert sample.get_seed() == 7


def test_uniform_sampling_thins_the_cloud(plane_with_outliers):
    sampling = plane_with_outliers.make_UniformSampling()
    sampling.set_RadiusSearch(0.1)
    assert 0 < sampling.filter().size < plane_with_outliers.size


def test_uniform_sampling_radius_controls_density(plane_with_outliers):
    sizes = []
    for radius in (0.05, 0.3):
        sampling = plane_with_outliers.make_UniformSampling()
        sampling.set_RadiusSearch(radius)
        sizes.append(sampling.filter().size)
    assert sizes[0] > sizes[1]


# --- Harris keypoints -------------------------------------------------

def test_harris_returns_points_with_a_response(plane_with_outliers):
    harris = plane_with_outliers.make_HarrisKeypoint3D()
    harris.set_Radius(0.1)
    keypoints = harris.compute()
    assert keypoints.ndim == 2
    assert keypoints.shape[1] == 4
    assert keypoints.dtype == np.float32


def test_harris_non_max_suppression_reduces_the_count(plane_with_outliers):
    counts = []
    for suppress in (False, True):
        harris = plane_with_outliers.make_HarrisKeypoint3D()
        harris.set_Radius(0.1)
        harris.set_NonMaxSupression(suppress)
        counts.append(len(harris.compute()))
    assert counts[1] <= counts[0]


@pytest.mark.parametrize(
    "method", [pcl.HARRIS, pcl.NOBLE, pcl.LOWE, pcl.TOMASI, pcl.CURVATURE])
def test_harris_response_methods(plane_with_outliers, method):
    harris = plane_with_outliers.make_HarrisKeypoint3D()
    harris.set_Radius(0.1)
    harris.set_Method(method)
    assert harris.compute().shape[1] == 4


def test_harris_method_constants_match_pcls_enum():
    """Taken from the header through the shim, not copied into Python."""
    assert (pcl.HARRIS, pcl.NOBLE, pcl.LOWE, pcl.TOMASI, pcl.CURVATURE) == (
        1, 2, 3, 4, 5)
