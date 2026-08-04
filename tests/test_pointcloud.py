"""Runtime tests for the built extension. Skipped when the package is not
built (requires PCL at build time): pip install . && pytest tests/

The API under test follows sirokujira/python-pcl (PointCloud of
pcl::PointXYZ + pcl.load / pcl.save)."""

import pickle

import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def test_construct_empty():
    cloud = pcl.PointCloud()
    assert cloud.size == 0
    assert len(cloud) == 0
    assert cloud.empty()


def test_construct_from_int():
    cloud = pcl.PointCloud(7)
    assert cloud.size == 7
    assert cloud.width == 7
    assert cloud.height == 1


def test_construct_from_list():
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 4.0, 8.0)]
    cloud = pcl.PointCloud(pts)
    assert cloud.size == 3
    assert cloud.to_list() == pytest.approx(pts)


def test_construct_copy():
    a = pcl.PointCloud([(1.0, 2.0, 3.0)])
    b = pcl.PointCloud(a)
    a[0] = (9.0, 9.0, 9.0)
    assert b[0] == pytest.approx((1.0, 2.0, 3.0))
    assert b.size == 1


def test_backwards_compatible_alias():
    assert pcl.PointCloudXYZ is pcl.PointCloud


def test_append_and_read():
    cloud = pcl.PointCloud()
    cloud.append(1.0, 2.0, 3.0)
    assert cloud.size == 1
    assert cloud.width == 1
    x, y, z = cloud[0]
    assert (x, y, z) == pytest.approx((1.0, 2.0, 3.0))


def test_resize_set_get():
    cloud = pcl.PointCloud()
    cloud.resize(5)
    assert cloud.size == 5
    cloud[2] = (9.0, 8.0, 7.0)
    assert cloud[2] == pytest.approx((9.0, 8.0, 7.0))
    assert cloud[-3] == pytest.approx((9.0, 8.0, 7.0))
    assert cloud.get_point(2) == pytest.approx((9.0, 8.0, 7.0))


def test_from_to_list_instance_methods():
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 4.0, 8.0)]
    cloud = pcl.PointCloud()
    cloud.from_list(pts)
    assert cloud.size == 3
    assert cloud.width == 3
    assert cloud.height == 1
    assert cloud.to_list() == pytest.approx(pts)


def test_index_error():
    cloud = pcl.PointCloud(2)
    with pytest.raises(IndexError):
        cloud[5]
    with pytest.raises(IndexError):
        cloud[-3]


def test_clear():
    cloud = pcl.PointCloud([(1.0, 2.0, 3.0)])
    cloud.clear()
    assert cloud.size == 0


def test_repr():
    assert repr(pcl.PointCloud(4)) == "<PointCloud of 4 points>"


def test_pickle_roundtrip():
    cloud = pcl.PointCloud([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    clone = pickle.loads(pickle.dumps(cloud))
    assert clone.to_list() == pytest.approx(cloud.to_list())


# --- numpy interop -----------------------------------------------------

np = pytest.importorskip("numpy")


def test_from_array_to_array_roundtrip():
    arr = np.array(
        [[0.0, 0.5, 1.0], [2.0, 4.0, 8.0], [-1.0, -2.0, -3.0]],
        dtype=np.float32,
    )
    cloud = pcl.PointCloud()
    cloud.from_array(arr)
    assert cloud.size == 3
    assert cloud.width == 3
    assert cloud.height == 1
    out = cloud.to_array()
    assert out.dtype == np.float32
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, arr)


def test_construct_from_ndarray():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    cloud = pcl.PointCloud(arr)
    assert cloud.size == 1
    assert cloud[0] == pytest.approx((1.0, 2.0, 3.0))


def test_from_array_rejects_bad_shape():
    with pytest.raises(ValueError):
        pcl.PointCloud().from_array(
            np.zeros((2, 4), dtype=np.float32)
        )


def test_empty_to_array():
    out = pcl.PointCloud().to_array()
    assert out.shape == (0, 3)


# --- PCD file I/O ------------------------------------------------------

def test_pcd_save_load_roundtrip(tmp_path):
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 4.0, 8.0)]
    cloud = pcl.PointCloud(pts)
    path = str(tmp_path / "cloud.pcd")

    pcl.save(cloud, path)
    loaded = pcl.load(path)
    assert loaded.size == 3
    assert loaded.to_list() == pytest.approx(pts)


def test_pcd_binary_roundtrip(tmp_path):
    pts = [(1.5, -2.25, 3.125)]
    cloud = pcl.PointCloud(pts)
    path = str(tmp_path / "cloud_bin.pcd")

    pcl.save(cloud, path, binary=True)
    loaded = pcl.load(path)
    assert loaded.to_list() == pytest.approx(pts)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        pcl.load(str(tmp_path / "nope.pcd"))


def test_unsupported_format_raises(tmp_path):
    with pytest.raises(ValueError):
        pcl.load(str(tmp_path / "cloud.xyz"))
    with pytest.raises(ValueError):
        pcl.save(pcl.PointCloud(1), str(tmp_path / "cloud.obj"))
