"""IFS and ASCII clouds, PNG, image extraction, LZF and .pclzf files,
debayering, ImageGrabber, and VLP-16 replay from a synthetic capture."""

import math
import struct
import threading
import time

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._io")


def _organized(h=6, w=8):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.dstack([u * 0.1, v * 0.1, 1.0 + 0.01 * u]).astype(np.float32)


# --- IFS / ASCII ----------------------------------------------------------------


def test_ifs_round_trip(tmp_path):
    pts = np.random.RandomState(0).rand(50, 3).astype(np.float32)
    path = tmp_path / "cloud.ifs"
    pcl.save(pcl.PointCloud(pts), path)
    back = pcl.load(path)
    # PCL's own writer would leak PointXYZ's padding into the file
    assert np.allclose(back.to_array(), pts)
    holes = pts.copy()
    holes[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pcl.save(pcl.PointCloud(holes), tmp_path / "holes.ifs")
    with pytest.raises(ValueError, match="x, y, z only"):
        pcl.load_XYZRGB(path)


def test_ifs_mesh_reads_as_cloud_and_polygons(tmp_path):
    """PCL writes IFS clouds only; a mesh written by hand reads back."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                        np.float32)
    faces = [(0, 1, 2), (1, 3, 2)]

    def string(text):
        data = text.encode() + b"\0"
        return struct.pack("<I", len(data)) + data

    blob = string("IFS") + struct.pack("<f", 1.0) + string("square")
    blob += string("VERTICES") + struct.pack("<I", len(vertices))
    blob += vertices.tobytes()
    blob += string("TRIANGLES") + struct.pack("<I", len(faces))
    blob += np.asarray(faces, np.uint32).tobytes()
    path = tmp_path / "square.ifs"
    path.write_bytes(blob)
    cloud, polygons = pcl.load_mesh(path)
    assert np.allclose(cloud.to_array(), vertices)
    assert [tuple(p) for p in polygons] == faces
    with pytest.raises(ValueError, match="writable"):
        pcl.save_mesh(cloud, polygons, tmp_path / "out.ifs")


def test_ascii_cloud(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("0 0 0\n1,2,3\n4\t5\t6\n")
    cloud = pcl.load(path)
    assert cloud.to_array().tolist() == [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
    other = tmp_path / "points.csv"
    other.write_text("1;2;3\n")
    assert pcl.load_ascii(other, separators=";").to_array().tolist() == \
        [[1, 2, 3]]


# --- PNG and image extraction ------------------------------------------------------


def test_extract_depth_and_colour(tmp_path):
    xyz = _organized()
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    depth = pcl.extract_image(cloud, "z")
    assert depth.shape == (6, 8) and depth.dtype == np.uint16
    assert np.allclose(depth, np.round(xyz[:, :, 2] * 10000), atol=1)
    raw = pcl.extract_image(cloud, "z", scaling=pcl.IMAGE_SCALING_FIXED_FACTOR,
                            scaling_factor=1000)
    assert raw[0, 0] == 1000

    rgba = np.zeros((6, 8, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    coloured = pcl.PointCloud_PointXYZRGBA()
    coloured.from_organized_rgba_array(xyz, rgba)
    image = pcl.extract_image(coloured, "rgb")
    assert image.shape == (6, 8, 3)
    assert np.all(image[..., 0] == 200) and np.all(image[..., 1:] == 0)
    with pytest.raises(ValueError, match="field"):
        pcl.extract_image(cloud, "rgb")
    with pytest.raises(ValueError, match="organized"):
        pcl.extract_image(pcl.PointCloud(xyz.reshape(-1, 3)), "z")

    if pcl.have_png():
        pcl.save_png(tmp_path / "depth.png", depth)
        pcl.save_png(tmp_path / "rgb.png", image)
        assert (tmp_path / "depth.png").read_bytes()[:4] == b"\x89PNG"
        with pytest.raises(ValueError):
            pcl.save_png(tmp_path / "bad.png", depth.astype(np.float32))


def test_label_image_colours_each_label():
    labels = np.repeat(np.arange(4), 6).reshape(4, 6)
    mono = pcl.label_image(labels)
    assert mono.dtype == np.uint16 and np.array_equal(mono, labels)
    rgb = pcl.label_image(labels, pcl.IMAGE_COLORS_RGB_GLASBEY)
    assert rgb.shape == (4, 6, 3)
    colours = {tuple(rgb[i, 0]) for i in range(4)}
    assert len(colours) == 4
    assert all(len({tuple(c) for c in rgb[i]}) == 1 for i in range(4))


# --- LZF and .pclzf -------------------------------------------------------------------


def test_lzf_round_trip():
    data = (b"point cloud library " * 200) + bytes(range(256))
    packed = pcl.lzf_compress(data)
    assert len(packed) < len(data)
    assert pcl.lzf_decompress(packed, len(data)) == data
    with pytest.raises(ValueError):
        pcl.lzf_decompress(packed, len(data) - 1)
    assert pcl.lzf_compress(b"") == b""


def _write_frame(directory, stamp, depth_mm, rgb, f=50.0):
    h, w = depth_mm.shape
    base = directory / ("frame_%05d" % stamp)
    pcl.write_pclzf_depth(str(base) + "_depth.pclzf", depth_mm)
    pcl.write_pclzf_rgb(str(base) + "_rgb.pclzf", rgb)
    pcl.write_pclzf_parameters(str(base) + ".xml", f, f, w / 2.0, h / 2.0)
    return base


def test_pclzf_files_round_trip(tmp_path):
    depth = np.full((10, 12), 1500, np.uint16)
    depth[0, 0] = 0
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 1] = 99
    base = _write_frame(tmp_path, 1, depth, rgb)
    cloud = pcl.read_pclzf_depth(str(base) + "_depth.pclzf",
                                 str(base) + ".xml")
    assert (cloud.width, cloud.height) == (12, 10)
    z = cloud.to_array()[:, 2]
    assert np.isnan(z[0]) and np.allclose(z[1:], 1.5)
    both = pcl.read_pclzf(str(base) + "_depth.pclzf", str(base) + "_rgb.pclzf",
                          str(base) + ".xml")
    colour = both.to_rgb_array() if hasattr(both, "to_rgb_array") else None
    if colour is not None:
        assert np.all(colour[:, 1] == 99)
    with pytest.raises(ValueError):
        pcl.write_pclzf_depth(tmp_path / "x.pclzf", depth.astype(np.int32))


def test_image_grabber_reads_pclzf_frames(tmp_path):
    rgb = np.zeros((10, 12, 3), np.uint8)
    for i, mm in enumerate((1000, 2000, 3000)):
        _write_frame(tmp_path, i, np.full((10, 12), mm, np.uint16), rgb)
    frames = pcl.ImageGrabber(tmp_path, pclzf_mode=True)
    assert len(frames) == 3
    for i, expect in enumerate((1.0, 2.0, 3.0)):
        z = frames[i].to_array()[:, 2]
        assert np.allclose(z, expect)
    assert frames[-1].height == 10
    with pytest.raises(IndexError):
        frames[3]
    # each frame used its own XML (f = 50, cx = 6): x = (u - cx) z / f
    x = frames[0].to_array()[:, 0].reshape(10, 12)
    assert x[0, 11] - x[0, 10] == pytest.approx(1.0 / 50.0, rel=1e-3)
    # the stored intrinsics stay PCL's defaults when read by index
    assert frames.get_CameraIntrinsics()[0] == pytest.approx(525.0)
    with pytest.raises(ValueError):
        pcl.ImageGrabber(tmp_path / "missing")


@pytest.mark.skipif(not pcl.have_png(), reason="this PCL writes no PNG")
def test_image_grabber_reads_png_depth(tmp_path):
    for i, mm in enumerate((800, 1600)):
        pcl.save_png(tmp_path / ("frame_%d_depth.png" % i),
                     np.full((8, 10), mm, np.uint16))
    frames = pcl.ImageGrabber(tmp_path)
    frames.set_CameraIntrinsics(40.0, 40.0, 5.0, 4.0)
    assert len(frames) == 2
    assert np.allclose(frames[1].to_array()[:, 2], 1.6)


# --- debayering -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bilinear", "edge_aware",
                                    "edge_aware_weighted"])
def test_debayer_flat_colour(method):
    # a uniform scene: every pixel of the mosaic shows its channel's level
    bayer = np.full((16, 16), 100, np.uint8)
    rgb = pcl.debayer(bayer, method)
    assert rgb.shape == (16, 16, 3)
    assert np.all(rgb[2:-2, 2:-2] == 100)
    with pytest.raises(ValueError):
        pcl.debayer(np.zeros((15, 16), np.uint8))
    with pytest.raises(ValueError):
        pcl.debayer(bayer, "magic")


# --- VLP-16 replay ------------------------------------------------------------------------


def _vlp_capture(path, revolutions=2.2, distance_m=5.0):
    """A .pcap of VLP-16 single-return packets: 12 firings of 2 x 16
    lasers per packet, one degree per firing, every return at
    `distance_m` (2 mm units). Equal timestamps make PCL replay it
    without pausing."""
    firings = int(revolutions * 360)
    payloads = []
    for start in range(0, firings - 11, 12):
        packet = b""
        for k in range(12):
            azimuth = ((start + k) * 100) % 36000
            packet += struct.pack("<HH", 0xEEFF, azimuth)
            packet += struct.pack("<HB", int(distance_m / 0.002), 50) * 32
        packet += struct.pack("<IBB", 0, 0x37, 0x22)
        assert len(packet) == 1206
        payloads.append(packet)
    blob = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for payload in payloads:
        udp = struct.pack(">HHHH", 2368, 2368, 8 + len(payload), 0) + payload
        ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64,
                         17, 0, bytes([192, 168, 1, 201]),
                         bytes([255, 255, 255, 255])) + udp
        frame = b"\xff" * 6 + b"\x00" * 6 + b"\x08\x00" + ip
        blob += struct.pack("<IIII", 1, 0, len(frame), len(frame)) + frame
    path.write_bytes(blob)
    return path


def test_vlp_grabber_replays_a_capture(tmp_path):
    capture = _vlp_capture(tmp_path / "vlp.pcap")
    sweeps = []
    done = threading.Event()

    def on_sweep(cloud):
        sweeps.append(cloud.to_array())
        if len(sweeps) >= 2:
            done.set()

    grabber = pcl.VLPGrabber(str(capture))
    assert "VLP" in grabber.get_name()
    grabber.register_callback(on_sweep)
    grabber.start()
    done.wait(10.0)
    grabber.stop()
    assert sweeps, "no sweep was delivered"
    full = sweeps[1] if len(sweeps) > 1 else sweeps[0]
    ranges = np.linalg.norm(full, axis=1)
    assert len(full) > 300 * 16
    assert np.allclose(ranges, 5.0, atol=0.05)
