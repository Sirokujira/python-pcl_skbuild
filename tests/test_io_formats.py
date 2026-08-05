"""Runtime tests for the file formats pcl.load / pcl.save handle.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import gzip
import os

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


@pytest.fixture
def points():
    return np.random.RandomState(6).rand(200, 3).astype(np.float32)


@pytest.fixture
def cloud(points):
    return pcl.PointCloud(points)


@pytest.mark.parametrize("suffix", [".pcd", ".ply"])
@pytest.mark.parametrize("binary", [False, True])
def test_roundtrip(tmp_path, cloud, points, suffix, binary):
    path = str(tmp_path / ("cloud" + suffix))
    pcl.save(cloud, path, binary=binary)
    assert os.path.getsize(path) > 0

    loaded = pcl.load(path)
    assert loaded.size == cloud.size
    assert loaded.to_array() == pytest.approx(points, abs=1e-5)


def test_format_argument_overrides_the_extension(tmp_path, cloud, points):
    path = str(tmp_path / "cloud.dat")
    pcl.save(cloud, path, format="ply")
    assert pcl.load(path, format="ply").to_array() == pytest.approx(
        points, abs=1e-5)


def test_unsupported_format_names_what_is_wrapped(tmp_path, cloud):
    with pytest.raises(ValueError, match="pcd, ply"):
        pcl.save(cloud, str(tmp_path / "cloud.obj"))
    with pytest.raises(ValueError, match="pcd, ply"):
        pcl.load(str(tmp_path / "cloud.obj"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        pcl.load(str(tmp_path / "does_not_exist.pcd"))


@pytest.mark.parametrize("suffix", [".pcd", ".ply"])
def test_gzipped_clouds_load(tmp_path, cloud, points, suffix):
    """PCL's readers take a filename and open it themselves, so a gzipped
    cloud is unreadable without decompressing first
    (strawlab/python-pcl#131)."""
    plain = str(tmp_path / ("cloud" + suffix))
    pcl.save(cloud, plain)
    compressed = plain + ".gz"
    with open(plain, "rb") as src, gzip.open(compressed, "wb") as dst:
        dst.write(src.read())
    os.unlink(plain)

    loaded = pcl.load(compressed)
    assert loaded.size == cloud.size
    assert loaded.to_array() == pytest.approx(points, abs=1e-5)


def test_gzip_leaves_no_temporary_behind(tmp_path, cloud):
    plain = str(tmp_path / "cloud.pcd")
    pcl.save(cloud, plain)
    compressed = plain + ".gz"
    with open(plain, "rb") as src, gzip.open(compressed, "wb") as dst:
        dst.write(src.read())

    import tempfile
    before = set(os.listdir(tempfile.gettempdir()))
    pcl.load(compressed)
    after = set(os.listdir(tempfile.gettempdir()))
    assert not (after - before)


def test_gzip_of_an_unsupported_format_still_reports_the_format(tmp_path):
    with pytest.raises(ValueError, match="pcd, ply"):
        pcl.load(str(tmp_path / "cloud.obj.gz"))


@pytest.mark.parametrize("suffix", [".pcd", ".ply"])
def test_gzipped_save_round_trips(tmp_path, cloud, points, suffix):
    """`save` to a .gz path must produce something `load` reads back.

    Every other gzip test compresses externally and only exercises the
    read side; without this one, `save` wrote an UNCOMPRESSED cloud to a
    .gz path and `load` then raised BadGzipFile on it.
    """
    path = str(tmp_path / ("cloud" + suffix + ".gz"))
    pcl.save(cloud, path)

    loaded = pcl.load(path)
    assert loaded.size == cloud.size
    assert loaded.to_array() == pytest.approx(points, abs=1e-5)


def test_gzipped_save_really_compresses(tmp_path, cloud):
    """Not just readable by us — a real gzip stream."""
    path = str(tmp_path / "cloud.pcd.gz")
    pcl.save(cloud, path)

    with open(path, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"
    # And any gzip reader gets a usable PCD out of it.
    with gzip.open(path, "rb") as handle:
        assert handle.read(11) == b"# .PCD v0.7"


def test_gzipped_save_works_for_binary_too(tmp_path, cloud, points):
    path = str(tmp_path / "cloud.pcd.gz")
    pcl.save(cloud, path, binary=True)
    assert pcl.load(path).to_array() == pytest.approx(points, abs=1e-5)


def test_gzipped_save_leaves_no_temporary_behind(tmp_path, cloud):
    import tempfile
    before = set(os.listdir(tempfile.gettempdir()))
    pcl.save(cloud, str(tmp_path / "cloud.pcd.gz"))
    after = set(os.listdir(tempfile.gettempdir()))
    assert not (after - before)


def test_explicit_format_overrides_the_gz_suffix(tmp_path, cloud, points):
    """`format=` names the format AND opts out of compression, on both
    sides — otherwise the two would disagree about a file called .gz."""
    path = str(tmp_path / "cloud.pcd.gz")
    pcl.save(cloud, path, format="pcd")

    with open(path, "rb") as handle:
        assert handle.read(2) != b"\x1f\x8b"
    assert pcl.load(path, format="pcd").to_array() == pytest.approx(
        points, abs=1e-5)
