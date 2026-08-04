"""Runtime tests for the coloured and intensity point-cloud types.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

The API follows sirokujira/python-pcl (pcl/pxi/PointCloud_PointXYZI.pxi
and friends), plus the uint8 colour views this package adds.
"""

import pickle

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

COLOURED = [pcl.PointCloud_PointXYZRGB, pcl.PointCloud_PointXYZRGBA]
ALL_TYPES = [pcl.PointCloud_PointXYZI] + COLOURED


@pytest.fixture
def xyz():
    return np.random.RandomState(0).rand(100, 3).astype(np.float32)


@pytest.fixture
def rgb():
    return np.random.RandomState(1).randint(
        0, 256, (100, 3)).astype(np.uint8)


# --- construction, shared across the types ---------------------------

@pytest.mark.parametrize("cls", ALL_TYPES)
def test_construct_empty(cls):
    cloud = cls()
    assert cloud.size == 0
    assert len(cloud) == 0
    assert cloud.empty()


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_construct_from_int(cls):
    cloud = cls(7)
    assert cloud.size == 7
    assert cloud.width == 7
    assert cloud.height == 1


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_array_roundtrip(cls, xyz):
    data = np.hstack([xyz, np.arange(100, dtype=np.float32)[:, None]])
    cloud = cls(data)
    assert cloud.size == 100
    assert cloud.to_array() == pytest.approx(data, abs=1e-6)


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_array_must_have_four_columns(cls, xyz):
    with pytest.raises(ValueError):
        cls().from_array(xyz)


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_indexing(cls, xyz):
    data = np.hstack([xyz, np.zeros((100, 1), dtype=np.float32)])
    cloud = cls(data)
    assert cloud[0][:3] == pytest.approx(tuple(data[0, :3]), abs=1e-6)
    assert cloud[-1][:3] == pytest.approx(tuple(data[-1, :3]), abs=1e-6)
    with pytest.raises(IndexError):
        cloud[100]


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_copy_construction(cls, xyz):
    data = np.hstack([xyz, np.zeros((100, 1), dtype=np.float32)])
    original = cls(data)
    assert cls(original).to_array() == pytest.approx(
        original.to_array(), abs=1e-6)


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_pickle_roundtrip(cls, xyz):
    data = np.hstack([xyz, np.zeros((100, 1), dtype=np.float32)])
    cloud = cls(data)
    assert pickle.loads(pickle.dumps(cloud)).to_array() == pytest.approx(
        cloud.to_array(), abs=1e-6)


# --- intensity --------------------------------------------------------

def test_xyzi_keeps_intensity(xyz):
    intensity = np.linspace(0, 1, 100, dtype=np.float32)[:, None]
    cloud = pcl.PointCloud_PointXYZI(np.hstack([xyz, intensity]))
    assert cloud.to_array()[:, 3] == pytest.approx(
        intensity.ravel(), abs=1e-6)
    assert cloud[5][3] == pytest.approx(intensity[5, 0], abs=1e-6)


# --- colour -----------------------------------------------------------

def test_rgb_array_roundtrip(xyz, rgb):
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    assert cloud.size == 100
    assert cloud.to_xyz_array() == pytest.approx(xyz, abs=1e-6)
    assert (cloud.to_rgb_array() == rgb).all()


def test_rgba_array_roundtrip(xyz, rgb):
    rgba = np.hstack([rgb, np.full((100, 1), 200, dtype=np.uint8)])
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(xyz, rgba)
    assert cloud.to_xyz_array() == pytest.approx(xyz, abs=1e-6)
    assert (cloud.to_rgba_array() == rgba).all()


def test_packed_and_unpacked_colour_are_the_same_bytes(xyz, rgb):
    """to_array()'s fourth column and to_rgb_array() read one union, so a
    cloud rebuilt from the packed form keeps the exact colours."""
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    rebuilt = pcl.PointCloud_PointXYZRGB(cloud.to_array())
    assert (rebuilt.to_rgb_array() == rgb).all()


def test_rgb_shape_validation(xyz, rgb):
    cloud = pcl.PointCloud_PointXYZRGB()
    with pytest.raises(ValueError, match="n, 3"):
        cloud.from_rgb_array(xyz[:, :2].copy(), rgb)
    with pytest.raises(ValueError, match="same length"):
        cloud.from_rgb_array(xyz, rgb[:50])


def test_rgba_shape_validation(xyz, rgb):
    cloud = pcl.PointCloud_PointXYZRGBA()
    with pytest.raises(ValueError, match="n, 4"):
        cloud.from_rgba_array(xyz, rgb)


def test_rgb_alpha_defaults_to_opaque(xyz, rgb):
    """from_rgb_array leaves no uninitialised alpha behind — a cloud saved
    from it must not come back transparent."""
    cloud = pcl.PointCloud_PointXYZRGBA()
    rgba = np.hstack([rgb, np.full((100, 1), 255, dtype=np.uint8)])
    cloud.from_rgba_array(xyz, rgba)
    assert (cloud.to_rgba_array()[:, 3] == 255).all()


# --- file I/O ---------------------------------------------------------

@pytest.mark.parametrize("suffix", [".pcd", ".ply"])
@pytest.mark.parametrize("binary", [False, True])
def test_xyzi_file_roundtrip(tmp_path, xyz, suffix, binary):
    intensity = np.linspace(0, 1, 100, dtype=np.float32)[:, None]
    cloud = pcl.PointCloud_PointXYZI(np.hstack([xyz, intensity]))
    path = str(tmp_path / ("cloud" + suffix))
    pcl.save(cloud, path, binary=binary)

    loaded = pcl.load_XYZI(path)
    assert loaded.size == 100
    assert loaded.to_array()[:, :3] == pytest.approx(xyz, abs=1e-5)
    assert loaded.to_array()[:, 3] == pytest.approx(
        intensity.ravel(), abs=1e-5)


@pytest.mark.parametrize("suffix", [".pcd", ".ply"])
def test_xyzrgb_file_roundtrip_preserves_colour(tmp_path, xyz, rgb, suffix):
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    path = str(tmp_path / ("cloud" + suffix))
    pcl.save(cloud, path)

    loaded = pcl.load_XYZRGB(path)
    assert loaded.to_xyz_array() == pytest.approx(xyz, abs=1e-5)
    assert (loaded.to_rgb_array() == rgb).all()


def test_xyzrgba_file_roundtrip(tmp_path, xyz, rgb):
    rgba = np.hstack([rgb, np.full((100, 1), 128, dtype=np.uint8)])
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(xyz, rgba)
    path = str(tmp_path / "cloud.pcd")
    pcl.save(cloud, path)
    assert (pcl.load_XYZRGBA(path).to_rgba_array() == rgba).all()


def test_typed_loaders_reject_unsupported_formats(tmp_path):
    for loader in (pcl.load_XYZI, pcl.load_XYZRGB, pcl.load_XYZRGBA):
        with pytest.raises(ValueError, match="pcd, ply"):
            loader(str(tmp_path / "cloud.obj"))


def test_typed_loaders_handle_gzip(tmp_path, xyz, rgb):
    import gzip

    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_rgb_array(xyz, rgb)
    plain = str(tmp_path / "cloud.pcd")
    pcl.save(cloud, plain)
    with open(plain, "rb") as src, gzip.open(plain + ".gz", "wb") as dst:
        dst.write(src.read())

    assert (pcl.load_XYZRGB(plain + ".gz").to_rgb_array() == rgb).all()
