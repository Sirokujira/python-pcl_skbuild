"""OctreePointCloudOccupancy / PointVector / SinglePoint / Adjacency,
OrganizedPointCloudCompression and the bearing angle image."""

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._octree")


@pytest.fixture(scope="module")
def clusters():
    """Three tight clusters in separate voxels of a 1 m grid; the first
    point sits away from them because PCL aligns the grid on it unless a
    bounding box is defined (see OctreePointCloudVoxelCentroid)."""
    rng = np.random.RandomState(0)
    centres = np.array([[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.5, 2.5, 0.5]])
    pts = np.vstack([c + rng.uniform(-0.2, 0.2, (20, 3)) for c in centres])
    return pts.astype(np.float32), centres


def _octree(cls, pts, resolution=1.0):
    octree = cls(resolution)
    octree.define_bounding_box((0, 0, 0), (4, 4, 4))
    octree.set_InputCloud(pcl.PointCloud(pts))
    octree.add_points_from_input_cloud()
    return octree


def test_point_vector_keeps_every_index(clusters):
    pts, centres = clusters
    octree = _octree(pcl.OctreePointCloudPointVector, pts)
    assert octree.get_leaf_count() == 3
    voxel_centres, indices = octree.get_voxels()
    assert sorted(len(i) for i in indices) == [20, 20, 20]
    assert sorted(np.concatenate(indices).tolist()) == list(range(60))
    for centre, members in zip(voxel_centres, indices):
        assert np.allclose(pts[members].mean(axis=0), centre, atol=0.1)
    assert octree.is_voxel_occupied_at_point((0.5, 0.5, 0.5))
    octree.delete_voxel_at_point((0.5, 0.5, 0.5))
    assert not octree.is_voxel_occupied_at_point((0.5, 0.5, 0.5))
    assert octree.get_leaf_count() == 2


def test_single_point_keeps_one_original_point(clusters):
    pts, _ = clusters
    octree = _octree(pcl.OctreePointCloudSinglePoint, pts)
    centres, kept = octree.get_voxels()
    assert centres.shape == (3, 3) and kept.shape == (3,)
    # the kept index is one of the voxel's own points
    for centre, index in zip(centres, kept):
        assert np.all(np.abs(pts[index] - centre) <= 0.5)


def test_occupancy_marks_voxels_without_points(clusters):
    pts, centres = clusters
    octree = pcl.OctreePointCloudOccupancy(1.0)
    octree.define_bounding_box((0, 0, 0), (4, 4, 4))
    octree.set_occupied_voxel_at_point((3.5, 3.5, 3.5))
    assert octree.get_leaf_count() == 1
    octree.set_occupied_voxels_at_points_from_cloud(pcl.PointCloud(pts))
    assert octree.get_leaf_count() == 4
    got = octree.get_occupied_voxel_centers()
    assert sorted(map(tuple, np.round(got, 3))) == sorted(
        map(tuple, np.vstack([centres, [[3.5, 3.5, 3.5]]])))
    with pytest.raises(RuntimeError, match="before"):
        octree.define_bounding_box((0, 0, 0), (8, 8, 8))


def test_adjacency_graph_links_touching_voxels():
    # a 3 x 1 x 1 row of voxels plus one far away; PCL starts the grid at
    # the cloud's minimum, so voxel centres sit half a voxel past the points
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5],
                    [6.5, 6.5, 6.5]], np.float32)
    octree = pcl.OctreePointCloudAdjacency(1.0, pcl.PointCloud(
        np.repeat(pts, 3, 0)))
    assert not hasattr(octree, "define_bounding_box")
    centres, counts, edges = octree.get_adjacency()
    assert len(centres) == 4 and list(counts) == [3, 3, 3, 3]
    order = {tuple(np.round(c, 3)): i for i, c in enumerate(centres)}
    a, b, c, far = (order[tuple(p + 0.5)] for p in pts)
    pairs = {tuple(sorted(e)) for e in edges.tolist()}
    assert pairs == {tuple(sorted((a, b))), tuple(sorted((b, c)))}
    assert pcl.OctreePointCloudAdjacency(1.0).get_leaf_count() == 0


def test_adjacency_occlusion_with_the_camera_outside_the_box():
    """PCL's walk wraps its keys once it leaves the box and reports the
    near point occluded by nothing; the wrapper stops at the box."""
    pts = np.array([[0.5, 0.5, 2.5], [0.5, 0.5, 5.5]], np.float32)
    octree = pcl.OctreePointCloudAdjacency(1.0, pcl.PointCloud(pts))
    camera = (0.5, 0.5, -3.0)
    assert not octree.test_for_occlusion((0.5, 0.5, 2.5), camera)
    assert octree.test_for_occlusion((0.5, 0.5, 5.5), camera)


# --- organized compression ------------------------------------------------------


def _organized(h=24, w=32, f=525.0):
    # a Kinect-like focal length: PCL quantises depth as disparity for it
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 1.0 + 0.002 * u + 0.001 * v
    return np.dstack([(u - w / 2) * z / f, (v - h / 2) * z / f, z])


needs_organized_compression = pytest.mark.skipif(
    not pcl.OrganizedPointCloudCompression.available(),
    reason="this PCL was built without libpng")


@needs_organized_compression
def test_organized_compression_round_trip():
    xyz = _organized().astype(np.float32)
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    codec = pcl.OrganizedPointCloudCompression()
    data = codec.encode(cloud, png_level=9)
    assert isinstance(data, bytes) and len(data) < cloud.size * 12
    back = codec.decode(data)
    assert (back.width, back.height) == (32, 24)
    # depth goes through a disparity image: ~2 mm at 1 m for f = 525
    assert np.allclose(back.to_array(), cloud.to_array(), atol=3e-3)
    with pytest.raises(ValueError, match="organized"):
        codec.encode(pcl.PointCloud(xyz.reshape(-1, 3)))
    with pytest.raises(ValueError):
        codec.decode(b"not a cloud")


@needs_organized_compression
def test_organized_compression_keeps_colour():
    xyz = _organized().astype(np.float32)
    rgba = np.zeros(xyz.shape[:2] + (4,), np.uint8)
    rgba[..., 0] = 180
    rgba[..., 2] = 40
    rgba[..., 3] = 255
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_organized_rgba_array(xyz, rgba)
    codec = pcl.OrganizedPointCloudCompression()
    back = codec.decode(codec.encode(cloud), color=True)
    assert isinstance(back, pcl.PointCloud_PointXYZRGBA)
    colour = back.to_rgba_array()
    assert (back.width, back.height) == (32, 24)
    assert np.all(np.abs(colour[:, 0].astype(int) - 180) <= 2)
    assert np.all(np.abs(colour[:, 2].astype(int) - 40) <= 2)


# --- bearing angle image ----------------------------------------------------------


def test_bearing_angle_image_shows_the_step():
    h, w = 20, 30
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.where(u < w // 2, 2.0, 3.0).astype(np.float32)
    xyz = np.dstack([(u - w / 2) * z / 25, (v - h / 2) * z / 25, z])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    image = pcl.bearing_angle_image(cloud)
    assert image.shape == (h, w) and image.dtype == np.uint8
    row = image[5].astype(int)
    # across a plane the angle drifts smoothly (the beam direction
    # changes); at the depth step it jumps
    steps = np.abs(np.diff(row[1:w - 2]))
    jump = w // 2 - 2            # the step between columns w/2 - 1 and w/2
    assert steps[jump] > 50
    assert np.max(np.delete(steps, [jump - 1, jump, jump + 1])) <= 4
    assert pcl.bearing_angle((0, 0, 1), (0.1, 0, 1)) == pytest.approx(
        90.0, abs=6.0)
    with pytest.raises(ValueError, match="organized"):
        pcl.bearing_angle_image(pcl.PointCloud(xyz.reshape(-1, 3)))
