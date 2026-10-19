"""Runtime tests for the second round of filters, OBJ loading and octree
compression.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def grid():
    """A 20x20 grid on the unit square at z=1 (so every point is in
    front of a camera at the origin)."""
    u, v = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    return pcl.PointCloud(np.column_stack([
        u.ravel(), v.ravel(), np.ones(400)]).astype(np.float32))


def _normals(values):
    arr = np.zeros((len(values), 4), dtype=np.float32)
    arr[:, :3] = values
    return pcl.PointCloud_Normal(arr)


# --- remove_nan ----------------------------------------------------------

def test_remove_nan_drops_the_bad_rows_and_reports_the_kept_indices():
    points = np.array([[0, 0, 0], [np.nan, 1, 1], [2, 2, 2], [3, np.nan, 3]],
                      dtype=np.float32)
    cloud = pcl.PointCloud(points)
    clean, kept = cloud.remove_nan()
    assert isinstance(clean, pcl.PointCloud)
    assert clean.size == 2
    assert kept == [0, 2]
    assert clean.to_array()[:, :3] == pytest.approx(points[[0, 2]])
    assert cloud.size == 4


# --- ShadowPoints ---------------------------------------------------------

def test_shadow_points_drops_the_edge_on_points():
    points = np.zeros((20, 3), dtype=np.float32)
    points[:, 1] = np.linspace(0, 1, 20)
    points[:, 2] = 1.0
    normals = np.zeros((20, 3), dtype=np.float32)
    normals[:10, 2] = 1.0        # facing the camera: kept
    normals[10:, 0] = 1.0        # perpendicular to the ray: shadow
    cloud = pcl.PointCloud(points)
    shadow = cloud.make_ShadowPoints()
    shadow.set_Normals(_normals(normals))
    shadow.set_Threshold(0.1)
    assert shadow.get_Threshold() == pytest.approx(0.1)
    kept = shadow.filter()
    assert kept.size == 10
    assert kept.to_array()[:, 1] == pytest.approx(points[:10, 1])
    shadow.set_negative(True)
    assert shadow.filter().size == 10


def test_shadow_points_needs_normals(grid):
    with pytest.raises(RuntimeError, match="set_Normals"):
        grid.make_ShadowPoints().filter()


# --- NormalSpaceSampling --------------------------------------------------

def test_normal_space_sampling_keeps_the_requested_count():
    rng = np.random.RandomState(0)
    cloud = pcl.PointCloud(rng.rand(1000, 3).astype(np.float32))
    directions = rng.randn(1000, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sampler = cloud.make_NormalSpaceSampling()
    sampler.set_Normals(_normals(directions.astype(np.float32)))
    sampler.set_Sample(100)
    sampler.set_Seed(7)
    sampler.set_Bins(4, 4, 4)
    assert sampler.get_Sample() == 100
    assert sampler.get_Seed() == 7
    assert sampler.filter().size == 100


def test_normal_space_sampling_needs_normals(grid):
    with pytest.raises(RuntimeError, match="set_Normals"):
        grid.make_NormalSpaceSampling().filter()


# --- SamplingSurfaceNormal ------------------------------------------------

def test_sampling_surface_normal_subsamples_and_writes_normals(grid):
    joined = pcl.PointCloud_PointNormal().from_xyz_and_normals(
        grid, pcl.PointCloud_Normal(np.zeros((grid.size, 4), np.float32)))
    sampler = joined.make_SamplingSurfaceNormal()
    sampler.set_Sample(20)
    sampler.set_Ratio(0.25)
    sampler.set_Seed(1)
    assert sampler.get_Sample() == 20
    assert sampler.get_Ratio() == pytest.approx(0.25)
    out = sampler.filter()
    assert isinstance(out, pcl.PointCloud_PointNormal)
    assert 0 < out.size < grid.size
    normals = out.to_array()[:, 3:6]
    # the grid is flat: every normal is +-z
    assert np.abs(normals[:, 2]) == pytest.approx(np.ones(out.size), abs=1e-3)
    with pytest.raises(ValueError):
        sampler.set_Ratio(1.5)


# --- GridMinimum / LocalMaximum --------------------------------------------

def test_grid_minimum_keeps_the_lowest_point_per_cell():
    rng = np.random.RandomState(2)
    points = rng.rand(200, 3).astype(np.float32)
    cloud = pcl.PointCloud(points)
    ground = cloud.make_GridMinimum(0.5)
    assert ground.get_Resolution() == pytest.approx(0.5)
    out = ground.filter()
    assert out.size == 4
    cells = (points[:, :2] // 0.5).astype(int)
    for x, y, z in out.to_array()[:, :3]:
        same_cell = np.all(cells == (np.array([x, y]) // 0.5).astype(int), axis=1)
        assert z == pytest.approx(points[same_cell, 2].min())
    with pytest.raises(ValueError):
        cloud.make_GridMinimum(0.0)


def test_local_maximum_drops_the_peak_and_keeps_its_neighbours():
    """PCL's LocalMaximum REMOVES the local maxima (it thins a surface by
    its peaks); the points around a peak are kept, and `negative`
    returns the peaks."""
    u, v = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    points = np.column_stack([u.ravel(), v.ravel(), np.zeros(121)])
    points[60, 2] = 1.0           # the centre point rises
    cloud = pcl.PointCloud(points.astype(np.float32))
    maxima = cloud.make_LocalMaximum()
    maxima.set_Radius(0.25)
    assert maxima.get_Radius() == pytest.approx(0.25)
    out = maxima.filter().to_array()[:, :3]
    assert 1.0 not in out[:, 2]
    assert 0 < len(out) < 121
    # every flat point within the radius of the peak is kept
    near = np.linalg.norm(points[:, :2] - points[60, :2], axis=1) < 0.25
    for p in points[near & (points[:, 2] == 0), :2]:
        assert np.linalg.norm(out[:, :2] - p, axis=1).min() < 1e-5
    maxima.set_negative(True)
    peaks = maxima.filter().to_array()[:, :3]
    assert 1.0 in peaks[:, 2]


# --- FrustumCulling -------------------------------------------------------

def test_frustum_culling_keeps_what_the_camera_sees():
    points = np.array([
        [5, 0, 0],      # straight ahead: kept
        [-5, 0, 0],     # behind: dropped
        [5, 10, 0],     # far outside the horizontal FOV: dropped
        [0.5, 0, 0],    # nearer than the near plane: dropped
        [50, 0, 0],     # beyond the far plane: dropped
    ], dtype=np.float32)
    cloud = pcl.PointCloud(points)
    frustum = cloud.make_FrustumCulling()
    frustum.set_CameraPose(np.eye(4, dtype=np.float32))
    frustum.set_HorizontalFOV(60)
    frustum.set_VerticalFOV(60)
    frustum.set_NearPlaneDistance(1.0)
    frustum.set_FarPlaneDistance(20.0)
    assert frustum.get_NearPlaneDistance() == pytest.approx(1.0)
    assert frustum.get_FarPlaneDistance() == pytest.approx(20.0)
    out = frustum.filter()
    assert out.size == 1
    assert out[0] == pytest.approx((5, 0, 0))
    frustum.set_negative(True)
    assert frustum.filter().size == 4
    with pytest.raises(ValueError):
        frustum.set_CameraPose(np.eye(3))


def test_frustum_culling_pose_moves_the_camera():
    cloud = pcl.PointCloud(np.array([[5, 0, 0]], dtype=np.float32))
    frustum = cloud.make_FrustumCulling()
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 10.0            # camera at x=10, looking along +x
    frustum.set_CameraPose(pose)
    frustum.set_HorizontalFOV(60)
    frustum.set_VerticalFOV(60)
    frustum.set_NearPlaneDistance(0.5)
    frustum.set_FarPlaneDistance(20.0)
    assert frustum.filter().size == 0


# --- MedianFilter ---------------------------------------------------------

def test_median_filter_flattens_a_spike_in_an_organized_cloud():
    u, v = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    points = np.column_stack([u.ravel(), v.ravel(), np.ones(100)])
    points[55, 2] = 5.0
    cloud = pcl.PointCloud(points.astype(np.float32))
    cloud.resize(100)
    assert cloud.height == 1
    with pytest.raises(ValueError, match="organized"):
        cloud.make_MedianFilter().filter()

    organized = pcl.PointCloud()
    organized.from_organized_array(
        points.astype(np.float32).reshape(10, 10, 3))
    assert organized.is_organized
    median = organized.make_MedianFilter()
    median.set_WindowSize(5)
    median.set_MaxAllowedMovement(10.0)
    assert median.get_WindowSize() == 5
    assert median.get_MaxAllowedMovement() == pytest.approx(10.0)
    out = median.filter()
    assert out.size == 100
    assert out.to_organized_array()[5, 5, 2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        median.set_WindowSize(0)


# --- BilateralFilter ------------------------------------------------------

def test_bilateral_filter_smooths_an_intensity_outlier():
    u, v = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    values = np.column_stack([u.ravel(), v.ravel(), np.zeros(100),
                              np.ones(100)]).astype(np.float32)
    values[55, 3] = 10.0
    cloud = pcl.PointCloud_PointXYZI(values)
    bilateral = cloud.make_BilateralFilter()
    bilateral.set_HalfSize(0.2)
    bilateral.set_StdDev(5.0)
    assert bilateral.get_HalfSize() == pytest.approx(0.2)
    assert bilateral.get_StdDev() == pytest.approx(5.0)
    out = bilateral.filter()
    assert isinstance(out, pcl.PointCloud_PointXYZI)
    assert out.size == 100
    assert 1.0 < out.to_array()[55, 3] < 10.0
    # positions are untouched
    assert out.to_array()[:, :3] == pytest.approx(values[:, :3])


# --- ModelOutlierRemoval --------------------------------------------------

def test_model_outlier_removal_keeps_the_points_on_a_known_plane():
    rng = np.random.RandomState(3)
    on_plane = rng.rand(50, 3).astype(np.float32)
    on_plane[:, 2] = 0.0
    off_plane = rng.rand(20, 3).astype(np.float32)
    off_plane[:, 2] += 1.0
    cloud = pcl.PointCloud(np.vstack([on_plane, off_plane]))
    remover = cloud.make_ModelOutlierRemoval()
    remover.set_ModelType(pcl.SACMODEL_PLANE)
    remover.set_ModelCoefficients([0.0, 0.0, 1.0, 0.0])
    remover.set_Threshold(0.1)
    assert remover.get_ModelType() == pcl.SACMODEL_PLANE
    assert remover.get_Threshold() == pytest.approx(0.1)
    assert remover.filter().size == 50
    remover.set_negative(True)
    assert remover.filter().size == 20
    with pytest.raises(ValueError):
        remover.set_ModelCoefficients([])


# --- OBJ loading ----------------------------------------------------------

def test_obj_vertices_load_as_a_cloud(tmp_path):
    path = tmp_path / "tri.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    cloud = pcl.load(str(path))
    assert cloud.size == 3
    assert cloud[1] == pytest.approx((1, 0, 0))
    typed = pcl.load_XYZI(str(path))
    assert typed.size == 3
    with pytest.raises(ValueError, match="unsupported"):
        pcl.save(cloud, str(tmp_path / "out.obj"))


# --- octree compression ---------------------------------------------------

def test_compression_roundtrips_a_cloud_within_the_point_resolution():
    rng = np.random.RandomState(4)
    points = rng.rand(500, 3).astype(np.float32)
    cloud = pcl.PointCloud(points)
    encoder = pcl.OctreePointCloudCompression(
        pcl.MANUAL_CONFIGURATION, point_resolution=0.001,
        octree_resolution=0.01)
    data = encoder.encode(cloud)
    assert isinstance(data, bytes) and len(data) > 0
    decoder = pcl.OctreePointCloudCompression(
        pcl.MANUAL_CONFIGURATION, point_resolution=0.001,
        octree_resolution=0.01)
    back = decoder.decode(data)
    assert isinstance(back, pcl.PointCloud)
    assert back.size == cloud.size
    # the octree reorders points: every original has a restored point
    # within the quantisation step
    restored = back.to_array()[:, :3]
    gaps = np.linalg.norm(points[:, None, :] - restored[None, :, :], axis=2)
    assert gaps.min(axis=1).max() < 0.01


def test_compression_with_colour_keeps_the_colours_roughly():
    rng = np.random.RandomState(5)
    xyz = rng.rand(300, 3).astype(np.float32)
    rgb = rng.randint(0, 256, (300, 3)).astype(np.uint8)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    encoder = pcl.OctreePointCloudCompressionRGB(
        pcl.HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR)
    decoder = pcl.OctreePointCloudCompressionRGB(
        pcl.HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR)
    back = decoder.decode(encoder.encode(cloud))
    assert isinstance(back, pcl.PointCloud_PointXYZRGB)
    assert back.size == cloud.size
    # colours survive to within the profile's bit depth; match by
    # nearest position since the octree reorders points
    out_xyz = back.to_xyz_array()
    out_rgb = back.to_rgb_array().astype(int)
    for i in range(0, 300, 30):
        j = np.argmin(np.linalg.norm(out_xyz - xyz[i], axis=1))
        assert np.abs(out_rgb[j] - rgb[i].astype(int)).max() <= 8


def test_compression_profiles_are_exported_and_checked():
    assert pcl.LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR == 0
    assert pcl.HIGH_RES_OFFLINE_COMPRESSION_WITH_COLOR == 11
    # 12 is PCL's COMPRESSION_PROFILE_COUNT: not a profile, and used as
    # one it indexes past the profile table
    assert pcl.MANUAL_CONFIGURATION == 13
    with pytest.raises(ValueError):
        pcl.OctreePointCloudCompression(99)
    with pytest.raises(ValueError):
        pcl.OctreePointCloudCompression(12)
    with pytest.raises(ValueError):
        pcl.OctreePointCloudCompression(octree_resolution=0)
    codec = pcl.OctreePointCloudCompression()
    with pytest.raises(ValueError):
        codec.encode(pcl.PointCloud())
    with pytest.raises(TypeError):
        codec.decode("not bytes")
    with pytest.raises(ValueError):
        codec.decode(b"")
