"""Runtime tests for the remaining keypoint detectors and filters.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

H, W, F = 60, 80, 60.0


def _image(depth=None, square=True):
    """An organized RGBA frame of a plane at z=2 with a bright square."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.full((H, W), 2.0, np.float32) if depth is None else depth
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z]).astype(
        np.float32)
    rgba = np.full((H, W, 4), 20, dtype=np.uint8)
    rgba[:, :, 3] = 255
    if square:
        rgba[20:40, 30:50, :3] = 230
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)
    return cloud


def _square_corners():
    return np.array([[30, 20], [49, 20], [30, 39], [49, 39]], dtype=np.float32)


def _near_a_corner(uv, tolerance=4.0):
    d = np.linalg.norm(uv[:, None, :] - _square_corners()[None], axis=2)
    return d.min(axis=1) < tolerance


# --- 2-D keypoints on the image ------------------------------------------


def test_harris_2d_finds_the_square_corners():
    image = _image()
    harris = image.make_HarrisKeypoint2D()
    harris.set_Method(pcl.HARRIS)
    harris.set_NonMaxSupression(True)
    harris.set_Threshold(1e-6)
    out = harris.compute()
    assert out.shape[1] == 4
    assert len(out) >= 4
    # Back to pixels through the camera.
    uv = np.column_stack([out[:, 0] * F / out[:, 2] + W / 2,
                          out[:, 1] * F / out[:, 2] + H / 2])
    assert _near_a_corner(uv).mean() > 0.5


def test_harris_2d_method_checks():
    harris = pcl.HarrisKeypoint2D()
    with pytest.raises(ValueError):
        harris.set_Method(pcl.CURVATURE)
    with pytest.raises(RuntimeError):
        harris.compute()
    flat = pcl.PointCloud_PointXYZRGBA()
    flat.from_rgba_array(np.zeros((4, 3), np.float32),
                         np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        pcl.HarrisKeypoint2D(flat)


def test_trajkovic_2d():
    detector = _image().make_TrajkovicKeypoint2D()
    detector.set_Method(pcl.EIGHT_CORNERS)
    detector.set_WindowSize(5)
    detector.set_FirstThreshold(0.01)
    detector.set_SecondThreshold(1e-4)
    assert detector.get_WindowSize() == 5
    out = detector.compute()
    assert out.shape[1] == 4
    assert len(out) > 0
    with pytest.raises(ValueError):
        detector.set_WindowSize(4)


def test_agast_returns_pixels_near_the_corners():
    agast = _image().make_AgastKeypoint2D()
    agast.set_Threshold(30)
    agast.set_MaxDataValue(255)
    agast.set_NonMaxSuppression(True)
    assert agast.get_NonMaxSuppression() is True
    uv = agast.compute()
    assert uv.shape[1] == 2
    assert len(uv) >= 4
    assert _near_a_corner(uv).mean() > 0.5


def test_brisk_keypoints_and_descriptors():
    rng = np.random.RandomState(0)
    image = _image()
    # A textured patch gives BRISK something at several scales.
    rgba = np.full((H, W, 4), 20, dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[10:50, 15:65, :3] = rng.randint(0, 255, (40, 50, 1))
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.full((H, W), 2.0, np.float32)
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z]).astype(
        np.float32)
    image.from_organized_rgba_array(xyz, rgba)
    detector = image.make_BriskKeypoint2D()
    detector.set_Threshold(20)
    detector.set_Octaves(2)
    keypoints = detector.compute()
    assert keypoints.shape[1] == 7
    assert len(keypoints) > 0
    descriptors, kept = image.make_BRISK2DEstimation().compute(keypoints)
    assert descriptors.dtype == np.uint8
    assert descriptors.shape == (len(kept), 64)
    assert 0 < len(kept) <= len(keypoints)
    with pytest.raises(ValueError):
        image.make_BRISK2DEstimation().compute(np.zeros((3, 4)))


# --- 3-D keypoints --------------------------------------------------------


def test_harris_6d_on_a_coloured_cube_corner():
    rng = np.random.RandomState(1)
    faces = []
    for axis in range(3):
        pts = rng.rand(1500, 3)
        pts[:, axis] = 0.0
        faces.append(pts)
    points = np.vstack(faces).astype(np.float32)
    colours = (points * 255).astype(np.uint8)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(points, colours)
    harris = cloud.make_HarrisKeypoint6D()
    harris.set_Radius(0.1)
    harris.set_NonMaxSupression(True)
    harris.set_Threshold(1e-4)
    out = harris.compute()
    assert out.shape[1] == 4
    assert len(out) > 0
    with pytest.raises(ValueError):
        harris.set_Radius(0)


def test_trajkovic_3d_on_a_box_edge():
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.where(u < W / 2, 2.0, 2.0 + 0.02 * (u - W / 2)).astype(np.float32)
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z]).astype(
        np.float32)
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    normals = cloud.make_IntegralImageNormalEstimation().compute_cloud()
    detector = cloud.make_TrajkovicKeypoint3D()
    detector.set_Normals(normals)
    detector.set_FirstThreshold(1e-4)
    detector.set_SecondThreshold(1e-6)
    out = detector.compute()
    assert out.shape[1] == 4
    with pytest.raises(RuntimeError):
        pcl.TrajkovicKeypoint3D(cloud).compute()


def test_smoothed_surfaces_keypoint():
    rng = np.random.RandomState(2)
    points = rng.rand(2000, 3).astype(np.float32)
    points[:, 2] = 0.2 * np.exp(-((points[:, 0] - 0.5) ** 2 +
                                  (points[:, 1] - 0.5) ** 2) / 0.01)
    cloud = pcl.PointCloud(points)
    est = cloud.make_NormalEstimation()
    est.set_KSearch(15)
    normals = est.compute_cloud()
    detector = pcl.SmoothedSurfacesKeypoint(cloud, normals, scale=0.01)
    for scale in (0.02, 0.04):
        smoothed = cloud.make_moving_least_squares()
        smoothed.set_search_radius(scale * 3)
        sm = smoothed.process()
        if sm.size != cloud.size:
            pytest.skip("MLS resampled the cloud; need same-size clouds")
        sm_est = sm.make_NormalEstimation()
        sm_est.set_KSearch(15)
        detector.add_SmoothedPointCloud(sm, sm_est.compute_cloud(), scale)
    keypoints = detector.compute()
    assert isinstance(keypoints, pcl.PointCloud)
    with pytest.raises(ValueError):
        detector.add_SmoothedPointCloud(pcl.PointCloud(points[:10]),
                                        normals, 0.1)


# --- sampling ----------------------------------------------------------------


@pytest.fixture
def blob():
    rng = np.random.RandomState(3)
    return pcl.PointCloud(rng.rand(1000, 3).astype(np.float32))


def test_farthest_point_sampling_spreads_out(blob):
    sampler = blob.make_FarthestPointSampling()
    sampler.set_Sample(50)
    sampler.set_Seed(7)
    picked = sampler.filter()
    assert picked.size == 50
    indices = sampler.filter_indices()
    assert len(set(indices.tolist())) == 50
    # More spread out than a random pick of the same size.
    pts = picked.to_array()
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(50) * 9
    rand = blob.to_array()[np.random.RandomState(0).choice(1000, 50, False)]
    dr = np.linalg.norm(rand[:, None] - rand[None], axis=2) + np.eye(50) * 9
    assert d.min() > dr.min()
    with pytest.raises(ValueError):
        sampler.set_Sample(5000)
        sampler.filter()


def test_covariance_sampling(blob):
    pts = blob.to_array()
    pts[:, 2] = 0.0
    cloud = pcl.PointCloud(pts)
    normals = np.zeros((1000, 4), np.float32)
    normals[:, 2] = 1.0
    sampler = cloud.make_CovarianceSampling()
    sampler.set_Normals(pcl.PointCloud_Normal(normals))
    sampler.set_NumberOfSamples(100)
    assert sampler.filter().size == 100
    assert len(sampler.filter_indices()) == 100
    # A plane leaves 3 of the 6 rigid DOF unconstrained.
    assert sampler.compute_ConditionNumber() > 1e3


def test_fast_bilateral_omp_matches_the_serial_filter():
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.RandomState(4)
    z = (2.0 + rng.randn(H, W) * 0.005).astype(np.float32)
    xyz = np.dstack([(u - W / 2) * z / F, (v - H / 2) * z / F, z]).astype(
        np.float32)
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    omp = cloud.make_FastBilateralFilterOMP()
    omp.set_sigma_s(3)
    omp.set_sigma_r(0.05)
    serial = cloud.make_fast_bilateral_filter() if hasattr(
        cloud, "make_fast_bilateral_filter") else pcl.FastBilateralFilter(cloud)
    serial.set_sigma_s(3)
    serial.set_sigma_r(0.05)
    a = omp.filter().to_array()
    b = serial.filter().to_array()
    assert a == pytest.approx(b, abs=1e-5)


# --- normals / voxel grids ---------------------------------------------------


def test_normal_refinement_smooths_noisy_normals():
    rng = np.random.RandomState(5)
    pts = rng.rand(800, 3).astype(np.float32)
    pts[:, 2] = 0.0
    noisy = np.zeros((800, 4), np.float32)
    noisy[:, :3] = [0, 0, 1] + rng.randn(800, 3) * 0.2
    noisy[:, :3] /= np.linalg.norm(noisy[:, :3], axis=1, keepdims=True)
    refined = pcl.NormalRefinement(k=10).filter(
        pcl.PointCloud(pts), pcl.PointCloud_Normal(noisy))
    r = refined.to_array()[:, :3]
    assert np.mean(np.abs(r[:, 2])) > np.mean(np.abs(noisy[:, 2]))


def test_voxel_grid_label_keeps_the_majority_label():
    rng = np.random.RandomState(6)
    a = rng.rand(100, 3) * 0.1
    b = rng.rand(100, 3) * 0.1 + [0.5, 0, 0]
    cloud = pcl.PointCloud(np.vstack([a, b]).astype(np.float32))
    labels = np.r_[np.full(100, 3), np.full(100, 8)]
    labels[:10] = 8          # a minority inside the first cluster
    points, out = pcl.VoxelGridLabel(0.2).filter(cloud, labels)
    assert points.size == len(out) == 2
    order = np.argsort(points.to_array()[:, 0])
    assert out[order].tolist() == [3, 8]


def test_voxel_grid_covariance():
    rng = np.random.RandomState(7)
    pts = rng.randn(4000, 3).astype(np.float32) * [0.2, 0.05, 0.01]
    means, covs, counts = pcl.VoxelGridCovariance(
        2.0, min_points_per_voxel=10).compute(pcl.PointCloud(pts))
    assert means.shape[1:] == (3,) and covs.shape[1:] == (3, 3)
    assert counts.sum() <= 4000
    big = np.argmax(counts)
    assert covs[big] == pytest.approx(covs[big].T, abs=1e-6)
    assert covs[big][0, 0] > covs[big][1, 1] > covs[big][2, 2]
    with pytest.raises(ValueError):
        pcl.VoxelGridCovariance(1.0, min_points_per_voxel=2)


def test_voxel_grid_occlusion_behind_a_wall():
    rng = np.random.RandomState(8)
    wall = rng.rand(3000, 3).astype(np.float32)
    wall[:, 2] = 1.0
    back = rng.rand(3000, 3).astype(np.float32)
    back[:, 2] = 2.0
    cloud = pcl.PointCloud(np.vstack([wall, back]))
    occl = pcl.VoxelGridOcclusionEstimation(0.1, origin=(0.5, 0.5, -1.0))
    ijk, centers = occl.occluded_voxels(cloud)
    assert ijk.shape[1] == 3 and centers.shape == ijk.shape
    assert len(ijk) > 0
    assert np.all(centers[:, 2] > 1.0)
    states = occl.occlusion_states(cloud, pcl.PointCloud(np.array(
        [[0.55, 0.55, 2.0], [0.55, 0.55, 1.0], [9, 9, 9]], np.float32)))
    assert states.tolist() == [1, 0, -1]


# --- clipping --------------------------------------------------------------------


def test_plane_clipper(blob):
    kept = pcl.PlaneClipper3D(0, 0, 1, -0.5).clip(blob)
    z = blob.to_array()[:, 2]
    assert set(kept.tolist()) == set(np.flatnonzero(z >= 0.5).tolist())
    with pytest.raises(ValueError):
        pcl.PlaneClipper3D(0, 0, 0, 1)


def test_box_clipper_axis_aligned_and_rotated(blob):
    pts = blob.to_array()
    kept = pcl.BoxClipper3D(center=(0.5, 0.5, 0.5), size=(0.4, 0.4, 0.4)).clip(blob)
    inside = np.all(np.abs(pts - 0.5) <= 0.2, axis=1)
    assert set(kept.tolist()) == set(np.flatnonzero(inside).tolist())
    # 90 degrees about z swaps the x and y extents.
    long_x = pcl.BoxClipper3D(center=(0.5, 0.5, 0.5), size=(0.8, 0.2, 1.0))
    rotated = pcl.BoxClipper3D(center=(0.5, 0.5, 0.5), size=(0.8, 0.2, 1.0),
                               rodrigues=(0, 0, np.pi / 2))
    a = pts[long_x.clip(blob)]
    b = pts[rotated.clip(blob)]
    assert np.ptp(a[:, 0]) > np.ptp(a[:, 1])
    assert np.ptp(b[:, 1]) > np.ptp(b[:, 0])
    with pytest.raises(ValueError):
        pcl.BoxClipper3D(size=(0, 1, 1))


# --- convolution / morphology ------------------------------------------------------


def test_convolution_3d_smooths(blob):
    pts = blob.to_array()
    pts[:, 2] = np.random.RandomState(9).randn(1000) * 0.02
    cloud = pcl.PointCloud(pts)
    smoothed = pcl.Convolution3D(sigma=0.05, radius=0.15).convolve(cloud)
    assert smoothed.size == cloud.size
    assert np.std(smoothed.to_array()[:, 2]) < np.std(pts[:, 2])
    coloured = pcl.PointCloud_PointXYZRGB()
    coloured.from_rgb_array(pts, np.full((1000, 3), 100, np.uint8))
    out = pcl.Convolution3D(sigma=0.05, radius=0.15).convolve(coloured)
    assert isinstance(out, pcl.PointCloud_PointXYZRGB)
    with pytest.raises(TypeError):
        pcl.Convolution3D().convolve(np.zeros((3, 3)))


def test_convolution_2d_on_an_organized_cloud():
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.RandomState(10)
    z = (2.0 + rng.randn(H, W) * 0.01).astype(np.float32)
    xyz = np.dstack([u * 0.01, v * 0.01, z]).astype(np.float32)
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    kernel = [0.25, 0.5, 0.25]
    out = pcl.Convolution(kernel, kernel).convolve(cloud)
    assert (out.width, out.height) == (W, H)
    oz = out.to_array()[:, 2].reshape(H, W)
    assert np.std(oz[2:-2, 2:-2]) < np.std(z[2:-2, 2:-2])
    with pytest.raises(ValueError):
        pcl.Convolution([0.5, 0.5])
    with pytest.raises(RuntimeError):
        pcl.Convolution().convolve(cloud)


def test_morphological_operators():
    u, v = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    z = np.zeros_like(u)
    z[15, 15] = 1.0                      # a single spike
    cloud = pcl.PointCloud(np.column_stack(
        [u.ravel(), v.ravel(), z.ravel()]).astype(np.float32))
    opened = pcl.apply_morphological_operator(cloud, 0.1, pcl.MORPH_OPEN)
    assert opened.size == cloud.size
    assert opened.to_array()[:, 2].max() < 0.5      # the spike is gone
    dilated = pcl.apply_morphological_operator(cloud, 0.1, pcl.MORPH_DILATE)
    assert (dilated.to_array()[:, 2] > 0.5).sum() > 1   # and spreads
    with pytest.raises(ValueError):
        pcl.apply_morphological_operator(cloud, 0.1, 42)
