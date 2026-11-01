"""Dense stereo matching, disparity -> cloud and the digital elevation
map, on synthetic rectified pairs whose disparity is known."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._stereo")


H, W = 120, 160


def _texture(x, y):
    return (128 + 50 * np.sin(x / 3.1) * np.cos(y / 4.3)
            + 40 * np.sin((x + 2 * y) / 5.7))


def _pair(near=12, far=8):
    """Two fronto-parallel planes: disparity `far` on the left half,
    `near` on the right. A point at left column x shows up at x - d in
    the right image, so right(x) = left(x + d)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.where(u < W // 2, far, near)
    left = _texture(u, v).astype(np.uint8)
    right = _texture(u + d, v).astype(np.uint8)
    return left, right


@pytest.mark.parametrize("cls", ["BlockBasedStereoMatching",
                                 "AdaptiveCostSOStereoMatching"])
def test_matching_recovers_both_planes(cls):
    left, right = _pair()
    matcher = getattr(pcl, cls)(max_disparity=32, radius=5, ratio_filter=20,
                                left_right_check=True)
    matcher.compute(left, right)
    disparity = matcher.get_DisparityMap()
    assert disparity.shape == (H, W) and disparity.dtype == np.float32
    # the first max_disparity columns have nothing to match against
    assert np.isnan(disparity[:, :20]).all()
    far = disparity[10:-10, 45:W // 2 - 8]
    near = disparity[10:-10, W // 2 + 8:-10]
    for region, expected in ((far, 8), (near, 12)):
        valid = region[np.isfinite(region)]
        assert valid.size > 0.6 * region.size
        assert np.median(valid) == pytest.approx(expected, abs=0.25)


def test_point_cloud_depth_follows_disparity():
    left, right = _pair()
    bm = pcl.BlockBasedStereoMatching(max_disparity=32)
    bm.compute(left, right)
    focal, baseline = 100.0, 0.1
    cloud = bm.get_PointCloud(W / 2, H / 2, focal, baseline)
    assert (cloud.width, cloud.height) == (W, H)
    xyz = cloud.to_array().reshape(H, W, 3)
    disparity = bm.get_DisparityMap()
    ok = np.isfinite(disparity)
    assert np.allclose(xyz[ok][:, 2], focal * baseline / disparity[ok],
                       rtol=1e-5)
    assert np.isnan(xyz[~ok]).all()
    # x = (u - u_c) z / f
    v, u = np.nonzero(ok)
    assert np.allclose(xyz[ok][:, 0], (u - W / 2) * xyz[ok][:, 2] / focal,
                       atol=1e-5)

    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 1] = 77
    coloured = bm.get_PointCloud(W / 2, H / 2, focal, baseline, rgb=rgb)
    assert isinstance(coloured, pcl.PointCloud_PointXYZRGB)
    colours = coloured.to_rgb_array().reshape(H, W, 3)
    assert np.all(colours[ok][:, 1] == 77)


def test_median_filter_and_arguments():
    left, right = _pair()
    bm = pcl.BlockBasedStereoMatching(max_disparity=32)
    with pytest.raises(RuntimeError, match="compute"):
        bm.median_filter(2)
    with pytest.raises(RuntimeError, match="compute"):
        bm.get_DisparityMap()
    bm.compute(left, right)
    bm.median_filter(2)
    filtered = bm.get_DisparityMap()
    assert np.isnan(filtered[:2]).all() and np.isnan(filtered[:, -2:]).all()
    with pytest.raises(ValueError):
        bm.compute(left, right[:, :10])
    with pytest.raises(ValueError):
        bm.compute(left.astype(np.float32), right)
    with pytest.raises(ValueError):
        pcl.BlockBasedStereoMatching(max_disparity=0)
    with pytest.raises(ValueError, match="rgb"):
        bm.get_PointCloud(80, 60, 100, 0.1, rgb=np.zeros((2, 2, 3), np.uint8))


def test_disparity_to_cloud():
    disparity = np.full((6, 8), 10.0, np.float32)
    disparity[0, 0] = 0.0          # not above min_disparity
    disparity[1, 1] = 50.0         # not below max_disparity
    cloud = pcl.disparity_to_cloud(disparity, focal=100, baseline=0.2,
                                   max_disparity=40)
    xyz = cloud.to_array().reshape(6, 8, 3)
    assert (cloud.width, cloud.height) == (8, 6)
    assert np.isnan(xyz[0, 0]).all() and np.isnan(xyz[1, 1]).all()
    assert xyz[3, 5, 2] == pytest.approx(2.0)
    assert xyz[3, 5, 0] == pytest.approx((5 - 4) * 2.0 / 100)
    image = np.zeros((6, 8, 3), np.uint8)
    image[..., 0] = 30
    image[..., 2] = 90
    with_intensity = pcl.disparity_to_cloud(disparity, 100, 0.2, image=image)
    assert isinstance(with_intensity, pcl.PointCloud_PointXYZI)
    assert np.allclose(with_intensity.to_array()[:, 3], 40.0)
    with pytest.raises(ValueError):
        pcl.disparity_to_cloud(disparity, 0, 0.2)


def _ground(baseline=0.2, cy=H / 2):
    """A flat ground 1 m below the camera: below the horizon row,
    d = baseline * (v - cy) (from y = (v - cy) z / f = 1)."""
    v = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    d = baseline * (v - cy)
    d[d <= 0.5] = np.nan
    return d.astype(np.float32)


def test_digital_elevation_map_of_flat_ground():
    disparity = _ground()
    image = np.full((H, W, 3), 120, np.uint8)
    dem = pcl.digital_elevation_map(disparity, image, focal=100, baseline=0.2,
                                    columns=16, rows=8)
    assert dem.shape == (8, 16, 6) and dem.dtype == np.float32
    filled = np.isfinite(dem[..., 1])
    assert filled.sum() > 0.5 * filled.size
    # every filled cell sits on the ground and carries the image's
    # intensity; PCL reports the centre of the histogram's mode bin (1 cm
    # for height, 255 / 256 for intensity), not a mean
    assert np.allclose(dem[..., 1][filled], 1.0, atol=0.02)
    assert np.allclose(dem[..., 3][filled], 120.0, atol=1.0)
    # nearer cells (larger disparity) are further down the grid, and the
    # first bin is not put at the camera (z = 0) as PCL does for min 0
    z = np.nanmean(np.where(filled, dem[..., 2], np.nan), axis=1)
    assert np.all(z > 0) and np.all(np.diff(z) < 0)
    at_zero = pcl.digital_elevation_map(disparity, image, 100, 0.2,
                                        min_disparity=0.0, columns=16, rows=8)
    assert np.all(at_zero[0, :, 2] == 0)


def test_elevation_map_drops_what_would_land_past_the_last_bin():
    # (d - 0) / (10 / 3) rounds to 3 in float for d just under 10: PCL
    # would write one histogram past the end
    disparity = np.full((H, W), np.nextafter(np.float32(10), np.float32(0)),
                        np.float32)
    disparity[:, :W // 2] = 5.0
    image = np.zeros((H, W, 3), np.uint8)
    dem = pcl.digital_elevation_map(disparity, image, 100, 0.2,
                                    min_disparity=0.0, max_disparity=10.0,
                                    columns=4, rows=3)
    assert dem.shape == (3, 4, 6)
    assert np.isfinite(dem[1, :2, 1]).all()        # the d = 5 half
    with pytest.raises(ValueError):
        pcl.digital_elevation_map(np.full((4, 4), np.nan), np.zeros((4, 4, 3)),
                                  100, 0.2)
