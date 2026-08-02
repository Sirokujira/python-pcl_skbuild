"""Runtime tests for the built extension. Skipped when the package is not
built (requires PCL at build time): pip install . && pytest tests/"""

import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def test_construct_empty():
    cloud = pcl.PointCloudXYZ()
    assert cloud.size == 0
    assert len(cloud) == 0
    assert cloud.empty()


def test_append_and_read():
    cloud = pcl.PointCloudXYZ()
    cloud.append(1.0, 2.0, 3.0)
    assert cloud.size == 1
    x, y, z = cloud[0]
    assert (x, y, z) == pytest.approx((1.0, 2.0, 3.0))


def test_resize_set_get():
    cloud = pcl.PointCloudXYZ()
    cloud.resize(5)
    assert cloud.size == 5
    cloud[2] = (9.0, 8.0, 7.0)
    assert cloud[2] == pytest.approx((9.0, 8.0, 7.0))
    assert cloud[-3] == pytest.approx((9.0, 8.0, 7.0))


def test_from_to_list():
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 4.0, 8.0)]
    cloud = pcl.PointCloudXYZ.from_list(pts)
    assert cloud.size == 3
    assert cloud.to_list() == pytest.approx(pts)


def test_index_error():
    cloud = pcl.PointCloudXYZ()
    cloud.resize(2)
    with pytest.raises(IndexError):
        cloud[5]
    with pytest.raises(IndexError):
        cloud[-3]


def test_clear():
    cloud = pcl.PointCloudXYZ.from_list([(1.0, 2.0, 3.0)])
    cloud.clear()
    assert cloud.size == 0
