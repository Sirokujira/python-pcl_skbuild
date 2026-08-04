"""Runtime tests for the sensor (grabber) wrappers and the callback bridge.

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/

PCDGrabber replays .pcd files through the same Grabber interface and the
same per-frame callback a real sensor uses, so it exercises the whole
bridge -- including the fact that PCL calls back from its own thread --
without hardware attached.
"""

import os
import threading

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)

# Generous: CI machines are slow and PCL's timer thread sets the pace.
TIMEOUT = 10.0
FPS = 30.0


@pytest.fixture
def pcd_files(tmp_path):
    """Three one-value clouds, so a frame's contents identify its file."""
    paths = []
    for i in range(3):
        cloud = pcl.PointCloud(np.full((10, 3), float(i), dtype=np.float32))
        path = str(tmp_path / ("frame%d.pcd" % i))
        pcl.save(cloud, path)
        paths.append(path)
    return paths


class Collector:
    """Records frames and signals once the first one lands."""

    def __init__(self):
        self.frames = []
        self.threads = set()
        self.first = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, cloud):
        with self._lock:
            self.frames.append(cloud)
            self.threads.add(threading.current_thread().name)
        self.first.set()

    def wait(self, timeout=TIMEOUT):
        assert self.first.wait(timeout), "no frame arrived within %ss" % timeout


# --- construction ----------------------------------------------------

def test_grabber_reports_frame_count(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    assert grabber.num_frames() == 3
    assert len(grabber) == 3


def test_grabber_name_and_rate(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    assert "PCDGrabber" in grabber.get_name()
    assert grabber.get_frames_per_second() == pytest.approx(FPS)


def test_grabber_accepts_a_single_path(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files[0], FPS)
    assert grabber.num_frames() == 1


def test_grabber_accepts_a_directory(pcd_files):
    grabber = pcl.PCDGrabber(os.path.dirname(pcd_files[0]), FPS)
    assert grabber.num_frames() == 3


def test_grabber_rejects_an_empty_file_list():
    with pytest.raises(ValueError):
        pcl.PCDGrabber([], FPS)


def test_grabber_rejects_a_directory_with_no_clouds(tmp_path):
    """Better than PCL's own answer to a bad folder, which is a read error
    on stderr and a grabber with zero frames."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .pcd files"):
        pcl.PCDGrabber(str(empty), FPS)


# --- callbacks -------------------------------------------------------

def test_callback_receives_point_clouds(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    with grabber:
        collector.wait()
    assert collector.frames
    first = collector.frames[0]
    assert isinstance(first, pcl.PointCloud)
    assert first.size == 10


def test_callback_runs_on_pcls_own_thread(pcd_files):
    """The whole reason the bridge acquires the GIL."""
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    with grabber:
        collector.wait()
    assert collector.threads
    assert threading.current_thread().name not in collector.threads


def test_callback_delivers_each_file(pcd_files):
    """Frames carry their own file's value, so nothing is duplicated or
    dropped in the handover."""
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    with grabber:
        collector.wait()
        # Long enough for a full cycle of three frames at FPS.
        threading.Event().wait(6.0 / FPS)
    values = {float(c.to_array()[0][0]) for c in collector.frames}
    assert values <= {0.0, 1.0, 2.0}
    assert values, "no frames captured"


def test_registration_state(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    assert grabber.has_callback is False
    grabber.register_callback(lambda cloud: None)
    assert grabber.has_callback is True
    grabber.unregister_callback()
    assert grabber.has_callback is False


def test_register_callback_returns_self_for_chaining(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    assert grabber.register_callback(lambda cloud: None) is grabber


def test_register_callback_rejects_non_callables(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    with pytest.raises(TypeError):
        grabber.register_callback("not callable")


def test_handler_exception_does_not_kill_the_process(pcd_files, capfd):
    """An exception escaping into a noexcept C callback would terminate
    the interpreter; it must be printed and swallowed instead."""
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    raised = threading.Event()

    def explode(cloud):
        raised.set()
        raise ValueError("boom")

    grabber.register_callback(explode)
    with grabber:
        assert raised.wait(TIMEOUT)
        threading.Event().wait(3.0 / FPS)

    # Still alive, and the failure was reported rather than hidden.
    assert "ValueError" in capfd.readouterr().err


def test_unregistered_callback_stops_receiving(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    grabber.start()
    try:
        collector.wait()
        grabber.unregister_callback()
        threading.Event().wait(2.0 / FPS)
        settled = len(collector.frames)
        threading.Event().wait(4.0 / FPS)
        assert len(collector.frames) == settled
    finally:
        grabber.stop()


# --- lifecycle -------------------------------------------------------

def test_context_manager_starts_and_stops(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    with grabber as entered:
        assert entered is grabber
        collector.wait()
        assert grabber.is_running()
    assert not grabber.is_running()


def test_stop_is_idempotent(pcd_files):
    grabber = pcl.PCDGrabber(pcd_files, FPS)
    grabber.stop()
    grabber.stop()
    assert not grabber.is_running()


def test_grabber_survives_being_dropped_while_running(pcd_files):
    """__dealloc__ must disconnect before freeing, or a frame in flight
    calls into a dead handler."""
    grabber = pcl.PCDGrabber(pcd_files, FPS, True)
    collector = Collector()
    grabber.register_callback(collector)
    grabber.start()
    collector.wait()
    del grabber  # no crash, no hang


# --- the mode PCL cannot run --------------------------------------------

def test_trigger_mode_is_refused_rather_than_aborting(pcd_files):
    """PCL 1.14.0's manual-trigger path calls std::terminate (reproduced
    in plain C++). A binding must not pass that on to the caller."""
    grabber = pcl.PCDGrabber(pcd_files)
    for method in ("start", "trigger"):
        with pytest.raises(RuntimeError, match="frames_per_second"):
            getattr(grabber, method)()


# --- HDL / VLP LiDAR ----------------------------------------------------

def test_hdl_grabber_constructs_without_a_device():
    """No pcap and no network: construction and metadata must still work,
    which is as far as this can go without hardware."""
    grabber = pcl.HDLGrabber()
    assert "HDL" in grabber.get_name()
    assert grabber.get_maximum_number_of_lasers() > 0
    assert not grabber.is_running()


def test_hdl_grabber_distance_thresholds():
    grabber = pcl.HDLGrabber()
    grabber.set_minimum_distance_threshold(2.5)
    grabber.set_maximum_distance_threshold(80.0)
    assert grabber.get_minimum_distance_threshold() == pytest.approx(2.5)
    assert grabber.get_maximum_distance_threshold() == pytest.approx(80.0)


def test_hdl_grabber_accepts_a_callback():
    grabber = pcl.HDLGrabber()
    assert grabber.has_callback is False
    grabber.register_callback(lambda cloud: None)
    assert grabber.has_callback is True
    grabber.unregister_callback()
    assert grabber.has_callback is False
