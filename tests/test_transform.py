"""Runtime tests for applying a 4x4 transform to a cloud.

This is what closes the loop on registration and recognition: both hand
back a 4x4 matrix, and until `transform` existed there was no way to
apply one except a numpy round trip.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def rotation_z(angle, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    matrix[:3, 3] = translation
    return matrix


@pytest.fixture
def axes():
    """Two unit vectors, so a rotation's effect is readable by hand."""
    return pcl.PointCloud(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))


@pytest.fixture
def cloud():
    return pcl.PointCloud(
        np.random.RandomState(0).rand(500, 3).astype(np.float32))


# --- the transform itself ----------------------------------------------

def test_rotation_and_translation(axes):
    matrix = rotation_z(np.pi / 2, (10.0, 20.0, 30.0))
    moved = axes.transform(matrix)
    assert moved.to_array() == pytest.approx(
        np.array([[10.0, 21.0, 30.0], [9.0, 20.0, 30.0]]), abs=1e-4)


def test_identity_is_a_no_op(cloud):
    assert cloud.transform(np.eye(4)).to_array() == pytest.approx(
        cloud.to_array())


def test_transform_returns_a_new_cloud(cloud):
    moved = cloud.transform(rotation_z(0.5))
    assert isinstance(moved, pcl.PointCloud)
    assert moved.size == cloud.size
    # The source is untouched.
    assert not np.allclose(moved.to_array(), cloud.to_array())


def test_inverse_round_trips(cloud):
    matrix = rotation_z(0.7, (1.0, -2.0, 3.0))
    there = cloud.transform(matrix)
    back = there.transform(np.linalg.inv(matrix))
    assert back.to_array() == pytest.approx(cloud.to_array(), abs=1e-4)


def test_composition_matches_two_steps(cloud):
    first = rotation_z(0.3, (1.0, 0.0, 0.0))
    second = rotation_z(0.4, (0.0, 2.0, 0.0))
    stepwise = cloud.transform(first).transform(second)
    combined = cloud.transform(second @ first)
    assert stepwise.to_array() == pytest.approx(combined.to_array(), abs=1e-4)


def test_transform_preserves_distances(cloud):
    """A rigid motion cannot change the shape."""
    moved = cloud.transform(rotation_z(1.1, (5.0, 5.0, 5.0)))
    before = np.linalg.norm(cloud.to_array()[1:] - cloud.to_array()[:-1],
                            axis=1)
    after = np.linalg.norm(moved.to_array()[1:] - moved.to_array()[:-1],
                           axis=1)
    assert after == pytest.approx(before, abs=1e-4)


def test_transform_of_an_empty_cloud_is_empty():
    assert pcl.PointCloud().transform(np.eye(4)).size == 0


# --- the coloured point types ------------------------------------------

def test_colour_survives_the_transform():
    xyz = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    rgb = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.uint8)
    cloud = pcl.PointCloud_PointXYZRGB(np.zeros((2, 4), dtype=np.float32))
    cloud.from_rgb_array(xyz, rgb)

    moved = cloud.transform(rotation_z(np.pi / 2, (10.0, 20.0, 30.0)))
    assert np.array_equal(moved.to_rgb_array(), rgb)
    assert moved.to_array()[:, :3] == pytest.approx(
        np.array([[10.0, 21.0, 30.0], [9.0, 20.0, 30.0]]), abs=1e-4)


def test_intensity_survives_the_transform():
    values = np.array([[1, 0, 0, 0.25], [0, 1, 0, 0.75]], dtype=np.float32)
    cloud = pcl.PointCloud_PointXYZI(values)
    moved = cloud.transform(rotation_z(np.pi / 2))
    assert moved.to_array()[:, 3] == pytest.approx([0.25, 0.75])


def test_rgba_transforms_too():
    cloud = pcl.PointCloud_PointXYZRGBA(
        np.zeros((3, 4), dtype=np.float32))
    assert cloud.transform(np.eye(4)).size == 3


# --- what it was added for ---------------------------------------------

def test_an_icp_result_can_be_applied(cloud):
    """The loop this closes: registration returns a matrix, and applying
    it to the source must land on the target."""
    truth = rotation_z(0.15, (0.1, -0.05, 0.02))
    target = pcl.PointCloud(
        (cloud.to_array() @ truth[:3, :3].T + truth[:3, 3]).astype(
            np.float32))

    icp = cloud.make_IterativeClosestPoint()
    converged, matrix, _, _ = icp.icp(cloud, target)
    assert converged

    aligned = cloud.transform(matrix)
    assert aligned.to_array() == pytest.approx(target.to_array(), abs=1e-3)


def test_a_recognition_result_can_be_applied():
    """Same for correspondence grouping: the returned pose must place the
    model onto its instance in the scene."""
    grid = (np.mgrid[0:8, 0:8, 0:3].reshape(3, -1).T * 0.05).astype(
        np.float32)
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = pcl.PointCloud(grid)
    scene = pcl.PointCloud(grid + translation)

    grouping = pcl.GeometricConsistencyGrouping(model, scene)
    grouping.set_GCSize(0.05)
    grouping.set_GCThreshold(5)
    transform, _ = grouping.recognize([(i, i) for i in range(model.size)])[0]

    placed = model.transform(transform)
    assert placed.to_array() == pytest.approx(scene.to_array(), abs=1e-3)


# --- guards -------------------------------------------------------------

def test_transform_rejects_a_wrong_shape(axes):
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        axes.transform(np.eye(3))


def test_transform_rejects_a_non_rigid_matrix(axes):
    """PCL applies a scaling or reflecting matrix silently, producing a
    distorted cloud with no indication anything went wrong."""
    with pytest.raises(ValueError, match="rigid"):
        axes.transform(np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="rigid"):
        axes.transform(np.diag([-1.0, 1.0, 1.0, 1.0]))

    shear = np.eye(4)
    shear[0, 1] = 0.5
    with pytest.raises(ValueError, match="rigid"):
        axes.transform(shear)

    bad_last_row = rotation_z(0.2)
    bad_last_row[3] = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="rigid"):
        axes.transform(bad_last_row)


def test_transform_rejects_non_finite_values(axes):
    with pytest.raises(ValueError, match="non-finite"):
        axes.transform(np.full((4, 4), np.nan))


def test_transform_cloud_free_function(axes):
    matrix = rotation_z(np.pi / 2, (10.0, 20.0, 30.0))
    assert pcl.transform_cloud(axes, matrix).to_array() == pytest.approx(
        axes.transform(matrix).to_array())


def test_transform_cloud_rejects_a_non_cloud():
    with pytest.raises(TypeError, match="point cloud"):
        pcl.transform_cloud([1, 2, 3], np.eye(4))
