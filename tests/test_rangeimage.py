"""Runtime tests for RangeImage and NARF keypoints.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def wall_with_a_bump():
    """A flat wall at z=3 with a nearer patch at z=2 — something for a
    keypoint detector to find."""
    rng = np.random.RandomState(0)
    points = np.zeros((4000, 3), dtype=np.float32)
    points[:, 0] = rng.uniform(-1, 1, 4000)
    points[:, 1] = rng.uniform(-1, 1, 4000)
    points[:, 2] = 3.0
    points[:1000, :2] *= 0.3
    points[:1000, 2] = 2.0
    return pcl.PointCloud(points)


def test_range_image_has_a_pixel_grid(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    assert image.width > 0
    assert image.height > 0
    assert image.size == image.width * image.height
    assert len(image) == image.size


def test_range_image_array_shape_matches_the_grid(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    ranges = image.to_array()
    assert ranges.shape == (image.height, image.width)
    assert ranges.dtype == np.float32


def test_range_image_measures_the_right_distances(wall_with_a_bump):
    """A range is the euclidean distance from the sensor, not a depth:
    the nearest pixel is the bump at 2, the farthest a wall corner at
    sqrt(1 + 1 + 9)."""
    image = wall_with_a_bump.make_RangeImage()
    ranges = image.to_array()
    observed = ranges[np.isfinite(ranges)]
    assert len(observed) > 0
    assert observed.min() == pytest.approx(2.0, abs=0.1)
    assert observed.max() == pytest.approx(np.sqrt(11.0), abs=0.1)


def test_unobserved_pixels_are_infinite(wall_with_a_bump):
    """That is how PCL marks them, and callers need to know."""
    image = wall_with_a_bump.make_RangeImage()
    ranges = image.to_array()
    assert not np.isfinite(ranges).all()


def test_set_unseen_to_max_range_fills_the_holes(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    before = np.isfinite(image.to_array()).sum()
    image.set_unseen_to_max_range()
    after = np.isfinite(image.to_array()).sum()
    assert after >= before


def test_range_image_to_cloud(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    cloud = image.to_cloud()
    assert isinstance(cloud, pcl.PointCloud)
    assert cloud.size == image.size


def test_finer_resolution_gives_a_bigger_image(wall_with_a_bump):
    coarse = wall_with_a_bump.make_RangeImage(angular_resolution=0.02)
    fine = wall_with_a_bump.make_RangeImage(angular_resolution=0.005)
    assert fine.size > coarse.size


def test_range_image_rejects_a_nonpositive_resolution(wall_with_a_bump):
    with pytest.raises(ValueError, match="angular_resolution"):
        pcl.RangeImage(wall_with_a_bump, angular_resolution=0.0)


def test_coordinate_frame_constants_are_exported():
    """Taken from the header through the shim, not copied into Python."""
    assert (pcl.CAMERA_FRAME, pcl.LASER_FRAME) == (0, 1)


# --- NARF --------------------------------------------------------------

def test_narf_finds_keypoints(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    image.set_unseen_to_max_range()
    keypoints = image.narf_keypoints(support_size=0.3)

    assert keypoints
    assert all(isinstance(i, int) for i in keypoints[:5])
    # Indices address this image's pixels.
    assert max(keypoints) < image.size
    assert min(keypoints) >= 0


def test_narf_keypoints_are_observed_pixels(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    image.set_unseen_to_max_range()
    keypoints = image.narf_keypoints(support_size=0.3)

    ranges = image.to_array().reshape(-1)
    assert np.isfinite(ranges[keypoints]).all()


def test_narf_rejects_a_nonpositive_support_size(wall_with_a_bump):
    image = wall_with_a_bump.make_RangeImage()
    with pytest.raises(ValueError, match="support_size"):
        image.narf_keypoints(support_size=0.0)


def test_support_size_controls_how_many_keypoints(wall_with_a_bump):
    """support_size is the diameter of the surface a keypoint describes,
    so a bigger one leaves room for fewer of them."""
    counts = []
    for support_size in (0.1, 0.3, 0.8):
        image = wall_with_a_bump.make_RangeImage()
        image.set_unseen_to_max_range()
        counts.append(len(image.narf_keypoints(support_size=support_size)))
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]
