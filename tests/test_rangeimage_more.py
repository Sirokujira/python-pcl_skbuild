"""Runtime tests for RangeImagePlanar, RangeImageSpherical,
RangeImageBorderExtractor and NarfDescriptor.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import sys

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

W, H, F, CX, CY = 64, 48, 60.0, 32.0, 24.0


def _camera_cloud(z=2.0):
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = np.full((H, W), z, dtype=np.float32)
    xyz = np.dstack([(u - CX) * depth / F, (v - CY) * depth / F, depth])
    return pcl.PointCloud(xyz.reshape(-1, 3).astype(np.float32))


@pytest.fixture
def box_on_a_wall():
    """A wall at z=5 with a box face at z=3 in front of it."""
    rng = np.random.RandomState(0)
    points = np.zeros((6000, 3), dtype=np.float32)
    points[:, :2] = rng.uniform(-1.5, 1.5, (6000, 2))
    points[:, 2] = 5.0
    points[:2000, :2] *= 0.25
    points[:2000, 2] = 3.0
    return pcl.PointCloud(points)


# --- RangeImagePlanar ------------------------------------------------------


def test_planar_image_from_a_cloud():
    image = pcl.RangeImagePlanar(_camera_cloud(), W, H, CX, CY, F, F)
    assert isinstance(image, pcl.RangeImage)
    assert (image.width, image.height) == (W, H)
    assert image.get_FocalLength() == pytest.approx((F, F))
    assert image.get_Center() == pytest.approx((CX, CY))
    ranges = image.to_array()
    assert ranges.shape == (H, W)
    assert np.mean(np.isfinite(ranges)) > 0.9


def test_planar_image_from_a_depth_image():
    depth = np.full((H, W), 2.0, dtype=np.float32)
    image = pcl.RangeImagePlanar.from_depth_image(depth, CX, CY, F, F)
    assert (image.width, image.height) == (W, H)
    ranges = image.to_array()
    # Range is the distance along the ray, so it grows off-centre.
    assert ranges[int(CY), int(CX)] == pytest.approx(2.0, abs=1e-3)
    assert ranges[0, 0] > ranges[int(CY), int(CX)]
    points = image.to_cloud().to_array()
    assert points[:, 2] == pytest.approx(np.full(len(points), 2.0), abs=1e-3)


def test_planar_image_checks_its_intrinsics():
    with pytest.raises(ValueError):
        pcl.RangeImagePlanar(_camera_cloud(), 0, H, CX, CY, F, F)
    with pytest.raises(ValueError):
        pcl.RangeImagePlanar.from_depth_image(
            np.ones((H, W), np.float32), CX, CY, -1.0, F)
    with pytest.raises(ValueError):
        pcl.RangeImagePlanar.from_depth_image(np.ones(5, np.float32), CX,
                                              CY, F, F)


def test_planar_image_works_with_narf(box_on_a_wall):
    v, u = np.mgrid[0:120, 0:160].astype(np.float32)
    depth = np.full((120, 160), 5.0, dtype=np.float32)
    depth[40:80, 60:100] = 3.0
    image = pcl.RangeImagePlanar.from_depth_image(depth, 80, 60, 100, 100)
    keypoints = image.narf_keypoints(support_size=0.5)
    assert all(0 <= k < image.size for k in keypoints)


# --- RangeImageSpherical ---------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32",
                    reason="RangeImageSpherical cannot be linked outside PCL "
                           "on Windows")
def test_spherical_image(box_on_a_wall):
    image = pcl.RangeImageSpherical(box_on_a_wall,
                                    angular_resolution=np.radians(0.5))
    assert isinstance(image, pcl.RangeImage)
    assert image.width > 0 and image.height > 0
    assert np.isfinite(image.to_array()).any()
    with pytest.raises(ValueError):
        pcl.RangeImageSpherical(box_on_a_wall, angular_resolution=0)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only refusal")
def test_spherical_image_refused_on_windows(box_on_a_wall):
    with pytest.raises(RuntimeError, match="Windows"):
        pcl.RangeImageSpherical(box_on_a_wall)


# --- RangeImageBorderExtractor ---------------------------------------------


def test_border_extractor_finds_the_box_outline(box_on_a_wall):
    image = box_on_a_wall.make_RangeImage(angular_resolution=np.radians(0.5))
    traits = pcl.RangeImageBorderExtractor(image).compute()
    assert traits.shape == (image.height, image.width)
    assert traits.dtype == np.uint32
    obstacle = traits & (1 << pcl.BORDER_TRAIT__OBSTACLE_BORDER) != 0
    shadow = traits & (1 << pcl.BORDER_TRAIT__SHADOW_BORDER) != 0
    assert obstacle.sum() > 0
    assert shadow.sum() > 0
    # An obstacle border is the near side of the jump: box, not wall.
    ranges = image.to_array()
    assert np.median(ranges[obstacle]) < 4.0


def test_border_extractor_settings(box_on_a_wall):
    extractor = pcl.RangeImageBorderExtractor(box_on_a_wall.make_RangeImage())
    extractor.set_PixelRadiusBorders(2)
    assert extractor.get_PixelRadiusBorders() == 2
    extractor.set_MinimumBorderProbability(0.5)
    assert extractor.get_MinimumBorderProbability() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        extractor.set_MinimumBorderProbability(2.0)
    with pytest.raises(ValueError):
        extractor.set_PixelRadiusBorders(0)


def test_border_trait_constants_are_bit_positions():
    assert pcl.BORDER_TRAIT__OBSTACLE_BORDER == 0
    assert pcl.BORDER_TRAIT__VEIL_POINT_LEFT == 14


# --- NarfDescriptor --------------------------------------------------------


def test_narf_descriptors_at_valid_pixels(box_on_a_wall):
    image = box_on_a_wall.make_RangeImage(angular_resolution=np.radians(0.5))
    ranges = image.to_array().ravel()
    valid = np.flatnonzero(np.isfinite(ranges))[::200][:10].tolist()
    descriptor = pcl.NarfDescriptor(image, support_size=0.5,
                                    rotation_invariant=False)
    features = descriptor.compute(valid)
    assert features.dtype == np.float32
    assert features.shape[1] == 42
    assert 0 < len(features) <= len(valid)
    assert np.isfinite(features).all()
    # x, y, z is the descriptor frame's origin: at (PCL refines it to
    # within a few millimetres of) one of the requested pixels.
    points = image.to_cloud().to_array()[valid]
    for row in features:
        assert np.min(np.linalg.norm(points - row[:3], axis=1)) < 0.05


def test_narf_descriptors_at_keypoints(box_on_a_wall):
    image = box_on_a_wall.make_RangeImage(angular_resolution=np.radians(0.5))
    keypoints = image.narf_keypoints(support_size=0.5)
    features = pcl.NarfDescriptor(image, support_size=0.5).compute(keypoints)
    assert features.shape[1] == 42
    if keypoints:
        assert len(features) >= 1


def test_narf_descriptor_checks_its_inputs(box_on_a_wall):
    image = box_on_a_wall.make_RangeImage()
    with pytest.raises(ValueError):
        pcl.NarfDescriptor(image, support_size=0)
    descriptor = pcl.NarfDescriptor(image)
    assert descriptor.get_RotationInvariant() is True
    descriptor.set_RotationInvariant(False)
    assert descriptor.get_RotationInvariant() is False
    with pytest.raises(IndexError):
        descriptor.compute([image.size])
