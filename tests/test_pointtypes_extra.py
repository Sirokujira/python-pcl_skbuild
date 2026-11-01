"""Runtime tests for the PointNormal and PointWithViewpoint cloud types.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl
(pcl/pxi/PointCloud_PointNormal.pxi, PointCloud_PointWithViewpoint.pxi).
"""

import pickle

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

TYPES = [
    (pcl.PointCloud_PointNormal, 7),
    (pcl.PointCloud_PointWithViewpoint, 6),
]
LOADERS = {
    pcl.PointCloud_PointNormal: pcl.load_PointNormal,
    pcl.PointCloud_PointWithViewpoint: pcl.load_PointWithViewpoint,
}


def _values(cols, n=60, seed=0):
    return np.random.RandomState(seed).rand(n, cols).astype(np.float32)


# --- construction, shared across the two types ------------------------

@pytest.mark.parametrize("cls,cols", TYPES)
def test_construct_empty(cls, cols):
    cloud = cls()
    assert cloud.size == 0
    assert len(cloud) == 0
    assert cloud.empty()
    assert cloud.to_array().shape == (0, cols)


@pytest.mark.parametrize("cls,cols", TYPES)
def test_construct_from_int(cls, cols):
    cloud = cls(7)
    assert cloud.size == 7
    assert cloud.width == 7 and cloud.height == 1


@pytest.mark.parametrize("cls,cols", TYPES)
def test_array_roundtrip(cls, cols):
    values = _values(cols)
    cloud = cls(values)
    assert cloud.size == len(values)
    assert cloud.to_array() == pytest.approx(values, abs=1e-6)
    assert np.asarray(cloud.to_list()) == pytest.approx(values, abs=1e-6)


@pytest.mark.parametrize("cls,cols", TYPES)
def test_array_column_count_is_checked(cls, cols):
    with pytest.raises(ValueError):
        cls().from_array(_values(cols + 1))
    with pytest.raises(ValueError):
        cls().from_array(_values(3))


@pytest.mark.parametrize("cls,cols", TYPES)
def test_list_construction_and_indexing(cls, cols):
    values = _values(cols, n=5)
    cloud = cls(values.tolist())
    assert cloud[0] == pytest.approx(tuple(values[0]), abs=1e-6)
    assert cloud[-1] == pytest.approx(tuple(values[-1]), abs=1e-6)
    with pytest.raises(IndexError):
        cloud[5]


@pytest.mark.parametrize("cls,cols", TYPES)
def test_copy_construction(cls, cols):
    original = cls(_values(cols))
    copy = cls(original)
    assert copy.to_array() == pytest.approx(original.to_array())
    copy.resize(1)
    assert original.size == 60


@pytest.mark.parametrize("cls,cols", TYPES)
def test_pickle_roundtrip(cls, cols):
    cloud = cls(_values(cols))
    restored = pickle.loads(pickle.dumps(cloud))
    assert isinstance(restored, cls)
    assert restored.to_array() == pytest.approx(cloud.to_array(), abs=1e-6)


@pytest.mark.parametrize("cls,cols", TYPES)
@pytest.mark.parametrize("suffix", [".pcd", ".ply", ".pcd.gz"])
@pytest.mark.parametrize("binary", [False, True])
def test_file_roundtrip(tmp_path, cls, cols, suffix, binary):
    values = _values(cols)
    path = tmp_path / ("cloud" + suffix)
    pcl.save(cls(values), str(path), binary=binary)
    back = LOADERS[cls](str(path))
    assert isinstance(back, cls)
    assert back.to_array() == pytest.approx(values, abs=1e-6)


@pytest.mark.parametrize("cls,cols", TYPES)
def test_positions_split_off_as_a_plain_cloud(cls, cols):
    values = _values(cols)
    xyz = cls(values).to_xyz_cloud()
    assert isinstance(xyz, pcl.PointCloud)
    assert xyz.to_array() == pytest.approx(values[:, :3], abs=1e-6)


# --- PointNormal only -------------------------------------------------

def test_pointnormal_assembles_from_a_cloud_and_its_normals():
    rng = np.random.RandomState(1)
    theta = rng.rand(300) * 2 * np.pi
    height = rng.rand(300)
    cylinder = pcl.PointCloud(np.column_stack([
        np.cos(theta), np.sin(theta), height]).astype(np.float32))
    estimator = cylinder.make_NormalEstimation()
    estimator.set_KSearch(15)
    normals = estimator.compute_cloud()

    joined = pcl.PointCloud_PointNormal().from_xyz_and_normals(
        cylinder, normals)
    assert joined.size == 300
    array = joined.to_array()
    assert array[:, :3] == pytest.approx(cylinder.to_array()[:, :3])
    assert array[:, 3:] == pytest.approx(normals.to_array(), nan_ok=True)
    assert joined.to_normal_cloud().to_array() == pytest.approx(
        normals.to_array(), nan_ok=True)


def test_pointnormal_assembly_checks_the_sizes():
    cloud = pcl.PointCloud(np.zeros((4, 3), dtype=np.float32))
    normals = pcl.PointCloud_Normal(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="4 points"):
        pcl.PointCloud_PointNormal().from_xyz_and_normals(cloud, normals)


def test_pointnormal_transform_rotates_the_normals_too():
    """A quarter turn about z moves (1, 0, 0) to (0, 1, 0) — for the
    position AND the normal. Plain transformPointCloud would move the
    point and leave the normal behind."""
    cloud = pcl.PointCloud_PointNormal([[1, 0, 0, 1, 0, 0, 0.5]])
    quarter = np.array([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    moved = cloud.transform(quarter)
    assert isinstance(moved, pcl.PointCloud_PointNormal)
    assert moved[0] == pytest.approx((0, 1, 0, 0, 1, 0, 0.5), abs=1e-6)
    # a translation leaves the normal alone
    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = (1, 2, 3)
    assert cloud.transform(shift)[0] == pytest.approx(
        (2, 2, 3, 1, 0, 0, 0.5), abs=1e-6)


def test_pointnormal_transform_rejects_a_non_rigid_matrix():
    cloud = pcl.PointCloud_PointNormal([[1, 0, 0, 1, 0, 0, 0.0]])
    with pytest.raises(ValueError):
        cloud.transform(np.diag([2, 1, 1, 1]).astype(np.float32))


def test_viewpoint_cloud_has_no_transform():
    assert not hasattr(pcl.PointCloud_PointWithViewpoint, "transform")
