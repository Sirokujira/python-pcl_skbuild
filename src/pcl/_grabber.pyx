# distutils: language = c++
# cython: language_level=3
"""Sensor input: pcl::Grabber wrappers and the Python callback bridge.

A grabber is PCL's streaming-sensor interface — a device (or a recording
of one) that publishes point clouds as they arrive. Consumers register a
callback and PCL invokes it, on the grabber's own thread, once per frame.

Two things make this different from every other wrapper in this package,
and both are handled here rather than pushed onto callers:

**Callbacks.** `pcl::Grabber::registerCallback` takes a `std::function`,
which Cython cannot build from a Python callable. The C++ shim at
src/pcl/compat/grabber_callback.h turns that into a plain C function
pointer plus an opaque user-data pointer — both of which Cython can
supply. `_cloud_trampoline` below is that function pointer.

**The GIL.** PCL calls back from its own thread, without the GIL. The
trampoline acquires it before touching anything Python, and a raised
exception is printed rather than allowed to unwind into C++, where it
would cross a `noexcept` boundary and terminate the process.

python-pcl's method names (`RegisterCallback` / `Start` / `Stop`) are kept
as aliases; its own grabber callback never actually delivered the cloud,
so there is no behaviour to stay compatible with.
"""

import os
import traceback

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.io.pcd_grabber cimport PCDGrabber as cPCDGrabber
from pcl.pxd.io.hdl_grabber cimport HDLGrabber as cHDLGrabber
from pcl.pxd.compat.grabber_callback cimport CloudCallback, CloudCallbackFn

from pcl._pointcloud cimport PointCloud, wrap_cloud


cdef string _tobytes(value):
    if isinstance(value, str):
        return <string> value.encode()
    return <string> value


def _expand(path):
    """A directory becomes its sorted .pcd files; anything else passes
    through as the sequence it already is."""
    if isinstance(path, (str, bytes)) and os.path.isdir(path):
        entries = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".pcd")
        )
        if not entries:
            raise ValueError("no .pcd files in %r" % path)
        return entries
    return list(path)


cdef void _deliver(shared_ptr[cPointCloud[PointXYZ]] cloud,
                   object handler) noexcept:
    """Hand one frame to the Python handler, swallowing anything it raises.

    An exception escaping here would propagate out of a `noexcept` C
    function called from a boost::signals2 slot: std::terminate, i.e. the
    interpreter dies with no traceback. Printing keeps the stream alive
    and still shows the user what broke.
    """
    try:
        handler(wrap_cloud(cloud))
    except BaseException:
        traceback.print_exc()


cdef void _cloud_trampoline(shared_ptr[cPointCloud[PointXYZ]] cloud,
                            void* user_data) noexcept nogil:
    """The C function pointer PCL ends up calling, on the grabber thread."""
    with gil:
        _deliver(cloud, <object> user_data)


cdef class PCDGrabber:
    """Replay .pcd files as a point-cloud stream (pcl::PCDGrabber).

    The grabber you can run without hardware: same interface and same
    per-frame callback as a real sensor, fed from files. *path* is a
    single .pcd file, a sequence of .pcd files, or a directory — a
    directory is expanded here, sorted, because PCL's own string
    constructor treats every path as one file and hands back a grabber
    with zero frames after printing a read error, which is a silent
    no-op to anyone who passed a folder.

    ``frames_per_second`` must be positive. PCL's other mode --
    ``frames_per_second=0``, where :meth:`trigger` publishes one cloud on
    demand -- **aborts the process** on PCL 1.14.0: its manual-trigger
    path destroys a joinable ``std::thread``, so ``std::terminate`` runs
    and there is nothing a binding can catch. That reproduces in plain
    C++ with no Python involved, so this wrapper refuses to enter it (see
    :meth:`trigger`). To step through clouds by hand, call
    :func:`pcl.load` on the files instead.
    """

    cdef cPCDGrabber[PointXYZ]* me
    cdef CloudCallback* registration
    # Holds the Python callable alive for as long as PCL may call it; the
    # shim only has a borrowed void*.
    cdef object handler
    cdef float fps

    def __cinit__(self, path, float frames_per_second=0.0, bint repeat=False):
        cdef vector[string] files
        self.me = NULL
        self.registration = NULL
        self.handler = None
        self.fps = frames_per_second

        if isinstance(path, (str, bytes)) and not os.path.isdir(path):
            self.me = new cPCDGrabber[PointXYZ](
                _tobytes(path), frames_per_second, repeat)
        else:
            for item in _expand(path):
                files.push_back(_tobytes(item))
            if files.empty():
                raise ValueError("no .pcd files given")
            self.me = new cPCDGrabber[PointXYZ](
                files, frames_per_second, repeat)
        self.registration = new CloudCallback()

    cdef _reject_trigger_mode(self, str method):
        raise RuntimeError(
            "PCDGrabber.%s() needs frames_per_second > 0: PCL's "
            "manual-trigger path calls std::terminate (verified against "
            "PCL 1.14.0 in plain C++, so it is not a binding bug and "
            "cannot be caught). Construct with a positive rate, or use "
            "pcl.load() to step through the files yourself." % method
        )

    def __dealloc__(self):
        # Disconnect before anything else: a frame arriving after `handler`
        # is gone would dereference a dangling pointer.
        if self.registration is not NULL:
            self.registration.disconnect()
            del self.registration
            self.registration = NULL
        if self.me is not NULL:
            self.me.stop()
            del self.me
            self.me = NULL

    def register_callback(self, handler):
        """Call *handler* with a :class:`PointCloud` for every frame.

        Replaces any previous registration. Returns self, so it chains.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.handler = handler
        self.registration.connect(
            self.me, <CloudCallbackFn> _cloud_trampoline,
            <void*> self.handler)
        return self

    def unregister_callback(self):
        """Stop delivering frames to the registered handler."""
        self.registration.disconnect()
        self.handler = None

    @property
    def has_callback(self):
        return self.registration.connected()

    def start(self):
        """Begin streaming; callbacks arrive on PCL's timer thread."""
        cdef cPCDGrabber[PointXYZ]* g = self.me
        if self.fps <= 0:
            self._reject_trigger_mode("start")
        with nogil:
            g.start()

    def stop(self):
        cdef cPCDGrabber[PointXYZ]* g = self.me
        with nogil:
            g.stop()

    def trigger(self):
        """Publish the next cloud once (PCL's manual-trigger mode).

        Always raises: see the class docstring. Kept as a method so the
        reason is where someone reaching for it will read it, rather than
        looking like an API this package forgot to wrap.
        """
        self._reject_trigger_mode("trigger")

    def rewind(self):
        """Go back to the first file."""
        self.me.rewind()

    def is_running(self):
        return self.me.isRunning()

    def get_name(self):
        return self.me.getName().decode()

    def get_frames_per_second(self):
        return self.me.getFramesPerSecond()

    def num_frames(self):
        return self.me.numFrames()

    def __len__(self):
        return self.me.numFrames()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # python-pcl spelling
    RegisterCallback = register_callback
    Start = start
    Stop = stop


cdef class HDLGrabber:
    """Velodyne HDL / VLP LiDAR (pcl::HDLGrabber), replaying a .pcap.

    Only the file-backed constructor is wrapped. The live-network one
    takes a ``boost::asio::ip::address``, which neither a mirror header
    nor Cython can name — that needs a shim of its own, the same way the
    callback does.
    """

    cdef cHDLGrabber* me
    cdef CloudCallback* registration
    cdef object handler

    def __cinit__(self, corrections_file="", pcap_file=""):
        self.me = new cHDLGrabber(
            _tobytes(corrections_file), _tobytes(pcap_file))
        self.registration = new CloudCallback()
        self.handler = None

    def __dealloc__(self):
        if self.registration is not NULL:
            self.registration.disconnect()
            del self.registration
            self.registration = NULL
        if self.me is not NULL:
            self.me.stop()
            del self.me
            self.me = NULL

    def register_callback(self, handler):
        """Call *handler* with a :class:`PointCloud` for every sweep."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.handler = handler
        self.registration.connect(
            self.me, <CloudCallbackFn> _cloud_trampoline,
            <void*> self.handler)
        return self

    def unregister_callback(self):
        self.registration.disconnect()
        self.handler = None

    @property
    def has_callback(self):
        return self.registration.connected()

    def start(self):
        cdef cHDLGrabber* g = self.me
        with nogil:
            g.start()

    def stop(self):
        cdef cHDLGrabber* g = self.me
        with nogil:
            g.stop()

    def is_running(self):
        return self.me.isRunning()

    def get_name(self):
        return self.me.getName().decode()

    def get_frames_per_second(self):
        return self.me.getFramesPerSecond()

    def set_minimum_distance_threshold(self, float threshold):
        self.me.setMinimumDistanceThreshold(threshold)

    def get_minimum_distance_threshold(self):
        return self.me.getMinimumDistanceThreshold()

    def set_maximum_distance_threshold(self, float threshold):
        self.me.setMaximumDistanceThreshold(threshold)

    def get_maximum_distance_threshold(self):
        return self.me.getMaximumDistanceThreshold()

    def get_maximum_number_of_lasers(self):
        return self.me.getMaximumNumberOfLasers()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # python-pcl spelling
    RegisterCallback = register_callback
    Start = start
    Stop = stop
