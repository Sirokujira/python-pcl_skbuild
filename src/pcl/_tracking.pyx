# distutils: language = c++
# cython: language_level=3
"""Tracking: a particle filter following a known object through a point
cloud stream, and the pyramidal Lucas-Kanade tracker following pixels
across organized colour frames.

    tracker = pcl.ParticleFilterTracker(particle_num=600)
    tracker.set_ReferenceCloud(model)
    for frame in frames:
        tracker.set_InputCloud(frame)
        tracker.compute()
        x, y, z, roll, pitch, yaw, weight = tracker.get_Result()

python-pcl declared `pcl_tracking_*.pxd` but never wrapped any of it (its
`pxi/Tracking/AddList.txt` is empty), so the names here follow PCL's own.

The whole chain lives in pcl/compat/tracking_args.h: PCL needs a tracker,
a cloud coherence, point coherences inside it, a search method inside
that, and an Eigen initial pose, all as shared_ptr to class templates.
That header also explains why this wraps the OMP tracker specifically --
the plain `ParticleFilterTracker` never resamples.
"""

from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointNormal, PointXYZ, PointXYZRGBA
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.compat.tracking_args cimport (
    KLTTracker, ParticleTrackerT, addHSVColorCoherence, addNormalCoherence)

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport (
    PointCloud_PointNormal, PointCloud_PointXYZRGBA, wrap_pointnormal_cloud,
    wrap_xyzrgba_cloud)


ctypedef fused _Tracker:
    ParticleTrackerT[PointXYZ]
    ParticleTrackerT[PointXYZRGBA]
    ParticleTrackerT[PointNormal]


cdef tuple _result(_Tracker* tracker):
    cdef float[7] state
    tracker.result(state)
    # A plain loop: a generator in a fused function names its closure
    # after the specialisation, which Cython 3 does not mangle.
    out = []
    cdef int i
    for i in range(7):
        out.append(float(state[i]))
    return tuple(out)


cdef object _result_transform(_Tracker* tracker):
    import numpy as np
    cdef float[16] values
    tracker.resultTransform(values)
    matrix = np.empty(16, dtype=np.float32)
    cdef Py_ssize_t i
    for i in range(16):
        matrix[i] = values[i]
    # Column-major, which is how Eigen stored it.
    return matrix.reshape(4, 4, order="F")


cdef int _KIND_NONE = -1, _KIND_XYZ = 0, _KIND_RGBA = 1, _KIND_NORMAL = 2


cdef class ParticleFilterTracker:
    """Follow a rigid object with a particle filter
    (pcl::tracking::ParticleFilterOMPTracker).

    Each particle is a candidate 6-DOF pose; every frame they are
    reweighted by how well the reference cloud placed at that pose
    explains the scene, then resampled. `particle_num` buys accuracy with
    time, `step_noise` is how far a particle may drift per frame -- set
    it near the object's real per-frame motion.

    The point type follows the reference cloud: a :class:`PointCloud`,
    a :class:`PointCloud_PointXYZRGBA` (then `add_HSVColorCoherence`
    scores colour too) or a :class:`PointCloud_PointNormal` (then
    `add_NormalCoherence` scores normals). Every particle is scored on
    distance; `exact_coherence` pairs points through a kd-tree instead of
    the octree-voxel approximation (pcl::tracking::NearestPairPointCloudCoherence
    vs ApproxNearestPairPointCloudCoherence).
    """

    cdef shared_ptr[ParticleTrackerT[PointXYZ]] xyz
    cdef shared_ptr[ParticleTrackerT[PointXYZRGBA]] rgba
    cdef shared_ptr[ParticleTrackerT[PointNormal]] normal
    cdef int kind
    cdef int particle_num_
    cdef double step_noise
    cdef double resolution
    cdef double maximum_distance
    cdef int threads
    cdef bint exact
    cdef bint kld
    cdef int max_particle_num
    cdef float bin_size
    cdef double epsilon
    cdef double delta
    cdef object hsv
    cdef object normal_weight
    cdef bint has_reference
    cdef bint has_input
    cdef object pending_input

    def __cinit__(self, *args, **kwargs):
        # Configured in __init__ and built on the first reference cloud,
        # whose type picks the point type.
        self.kind = _KIND_NONE
        self.kld = False
        self.hsv = None
        self.normal_weight = None
        self.has_reference = False
        self.has_input = False

    def __init__(self, int particle_num=400, double step_noise=0.015,
                 double resolution=0.01, double maximum_distance=0.05,
                 int threads=1, bint exact_coherence=False):
        _check_common(particle_num, step_noise, resolution, threads)
        self.particle_num_ = particle_num
        self.step_noise = step_noise
        self.resolution = resolution
        self.maximum_distance = maximum_distance
        self.threads = threads
        self.exact = exact_coherence

    def add_HSVColorCoherence(self, double h_weight=1.0, double s_weight=1.0,
                              double v_weight=0.0, double weight=1.0):
        """Also score colour agreement in HSV
        (pcl::tracking::HSVColorCoherence; PCL's default weights). Needs a
        PointCloud_PointXYZRGBA reference; call before set_ReferenceCloud."""
        if self.kind != _KIND_NONE:
            raise RuntimeError("add coherences before set_ReferenceCloud()")
        self.hsv = (h_weight, s_weight, v_weight, weight)

    def add_NormalCoherence(self, double weight=1.0):
        """Also score normal agreement (pcl::tracking::NormalCoherence);
        the tracker then turns the reference's normals with each particle.
        Needs a PointCloud_PointNormal reference; call before
        set_ReferenceCloud."""
        if self.kind != _KIND_NONE:
            raise RuntimeError("add coherences before set_ReferenceCloud()")
        self.normal_weight = weight

    cdef int _kind_of(self, cloud) except -2:
        if isinstance(cloud, PointCloud):
            return _KIND_XYZ
        if isinstance(cloud, PointCloud_PointXYZRGBA):
            return _KIND_RGBA
        if isinstance(cloud, PointCloud_PointNormal):
            return _KIND_NORMAL
        raise TypeError("expected a PointCloud, PointCloud_PointXYZRGBA or "
                        "PointCloud_PointNormal")

    cdef _build(self, int kind):
        if self.hsv is not None and kind != _KIND_RGBA:
            raise ValueError("add_HSVColorCoherence needs a "
                             "PointCloud_PointXYZRGBA reference")
        if self.normal_weight is not None and kind != _KIND_NORMAL:
            raise ValueError("add_NormalCoherence needs a "
                             "PointCloud_PointNormal reference")
        if kind == _KIND_XYZ:
            if self.kld:
                self.xyz = make_shared[ParticleTrackerT[PointXYZ]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads,
                    self.max_particle_num, self.bin_size, self.epsilon,
                    self.delta)
            else:
                self.xyz = make_shared[ParticleTrackerT[PointXYZ]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads)
            if self.exact:
                self.xyz.get().setExactCoherence(True)
        elif kind == _KIND_RGBA:
            if self.kld:
                self.rgba = make_shared[ParticleTrackerT[PointXYZRGBA]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads,
                    self.max_particle_num, self.bin_size, self.epsilon,
                    self.delta)
            else:
                self.rgba = make_shared[ParticleTrackerT[PointXYZRGBA]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads)
            if self.exact:
                self.rgba.get().setExactCoherence(True)
            if self.hsv is not None:
                h, s, v, w = self.hsv
                addHSVColorCoherence(deref(self.rgba.get()), h, s, v, w)
        else:
            if self.kld:
                self.normal = make_shared[ParticleTrackerT[PointNormal]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads,
                    self.max_particle_num, self.bin_size, self.epsilon,
                    self.delta)
            else:
                self.normal = make_shared[ParticleTrackerT[PointNormal]](
                    self.particle_num_, self.step_noise, self.resolution,
                    self.maximum_distance, self.threads)
            if self.exact:
                self.normal.get().setExactCoherence(True)
            if self.normal_weight is not None:
                addNormalCoherence(deref(self.normal.get()),
                                   self.normal_weight)
        self.kind = kind

    cdef _check_kind(self, cloud):
        cdef int kind = self._kind_of(cloud)
        if self.kind == _KIND_NONE:
            self._build(kind)
        elif kind != self.kind:
            raise TypeError("this tracker follows %s clouds"
                            % ("PointCloud", "PointCloud_PointXYZRGBA",
                               "PointCloud_PointNormal")[self.kind])

    def set_ReferenceCloud(self, reference not None):
        """The object to follow.

        PCL wants this in the object's own frame; the shim recentres it
        and keeps the centroid as the initial pose, so the poses this
        tracker reports are in the scene's coordinates.
        """
        if reference.size == 0:
            raise ValueError("reference cloud is empty")
        self._check_kind(reference)
        cdef PointCloud xyz
        cdef PointCloud_PointXYZRGBA rgba
        cdef PointCloud_PointNormal normal
        if self.kind == _KIND_XYZ:
            xyz = reference
            self.xyz.get().setReferenceCloud(xyz.thisptr_shared)
        elif self.kind == _KIND_RGBA:
            rgba = reference
            self.rgba.get().setReferenceCloud(rgba.thisptr_shared)
        else:
            normal = reference
            self.normal.get().setReferenceCloud(normal.thisptr_shared)
        self.has_reference = True
        if self.pending_input is not None:
            pending, self.pending_input = self.pending_input, None
            self.set_InputCloud(pending)

    def set_InputCloud(self, cloud not None):
        """The next frame to track through (the reference's point type).
        Set before the reference, it is held until the reference picks
        the point type."""
        if not self.has_reference:
            self._kind_of(cloud)
            self.pending_input = cloud
            return
        self._check_kind(cloud)
        cdef PointCloud xyz
        cdef PointCloud_PointXYZRGBA rgba
        cdef PointCloud_PointNormal normal
        if self.kind == _KIND_XYZ:
            xyz = cloud
            self.xyz.get().setInputCloud(xyz.thisptr_shared)
        elif self.kind == _KIND_RGBA:
            rgba = cloud
            self.rgba.get().setInputCloud(rgba.thisptr_shared)
        else:
            normal = cloud
            self.normal.get().setInputCloud(normal.thisptr_shared)
        self.has_input = True

    @property
    def particle_num(self):
        if self.kind == _KIND_XYZ:
            return self.xyz.get().particleNum()
        if self.kind == _KIND_RGBA:
            return self.rgba.get().particleNum()
        if self.kind == _KIND_NORMAL:
            return self.normal.get().particleNum()
        return self.particle_num_

    def compute(self):
        """Advance the filter by one frame."""
        if not self.has_reference:
            raise RuntimeError(
                "set_ReferenceCloud() is required before compute()")
        if not self.has_input:
            raise RuntimeError("set_InputCloud() is required before compute()")
        cdef ParticleTrackerT[PointXYZ]* a = self.xyz.get()
        cdef ParticleTrackerT[PointXYZRGBA]* b = self.rgba.get()
        cdef ParticleTrackerT[PointNormal]* c = self.normal.get()
        cdef int kind = self.kind
        with nogil:
            if kind == _KIND_XYZ:
                a.compute()
            elif kind == _KIND_RGBA:
                b.compute()
            else:
                c.compute()

    cdef _require(self):
        if self.kind == _KIND_NONE:
            raise RuntimeError("set_ReferenceCloud() is required first")

    def get_Result(self):
        """Return ``(x, y, z, roll, pitch, yaw, weight)``.

        The pose is in the scene's coordinates; `weight` is the winning
        particle's normalized likelihood.
        """
        self._require()
        if self.kind == _KIND_XYZ:
            return _result(self.xyz.get())
        if self.kind == _KIND_RGBA:
            return _result(self.rgba.get())
        return _result(self.normal.get())

    def get_ResultTransform(self):
        """The same pose as a ``(4, 4)`` float32 matrix."""
        self._require()
        if self.kind == _KIND_XYZ:
            return _result_transform(self.xyz.get())
        if self.kind == _KIND_RGBA:
            return _result_transform(self.rgba.get())
        return _result_transform(self.normal.get())

    def get_AlignedReference(self):
        """The reference cloud placed at the tracked pose, of the
        reference's type (normals turned with it)."""
        self._require()
        cdef PointCloud xyz
        cdef shared_ptr[cPointCloud[PointXYZRGBA]] rgba
        cdef shared_ptr[cPointCloud[PointNormal]] normal
        if self.kind == _KIND_XYZ:
            xyz = PointCloud()
            self.xyz.get().alignedReferenceCloud(deref(xyz.ptr()))
            return xyz
        if self.kind == _KIND_RGBA:
            rgba.reset(new cPointCloud[PointXYZRGBA]())
            self.rgba.get().alignedReferenceCloud(deref(rgba.get()))
            return wrap_xyzrgba_cloud(rgba)
        normal.reset(new cPointCloud[PointNormal]())
        self.normal.get().alignedReferenceCloud(deref(normal.get()))
        return wrap_pointnormal_cloud(normal)


cdef _check_common(int particle_num, double step_noise, double resolution,
                   int threads):
    if particle_num <= 0:
        raise ValueError("particle_num must be > 0, got %r" % particle_num)
    if step_noise <= 0:
        raise ValueError("step_noise must be > 0, got %r" % step_noise)
    if resolution <= 0:
        raise ValueError("resolution must be > 0, got %r" % resolution)
    if threads <= 0:
        raise ValueError("threads must be > 0, got %r" % threads)


cdef class KLDAdaptiveParticleFilterTracker(ParticleFilterTracker):
    """A particle filter whose particle count adapts to how certain it is
    (pcl::tracking::KLDAdaptiveParticleFilterOMPTracker, Fox 2003).

    It starts from `particle_num` and, every frame, draws particles until
    the Kullback-Leibler bound for `epsilon` (error) and `delta`
    (confidence) is met over state-space bins of `bin_size` -- up to
    `max_particle_num`. A confident track runs on few particles, a lost
    one spreads out. Same methods (and point types, and coherences) as
    :class:`ParticleFilterTracker`; `particle_num` reports the current
    count. Defaults are PCL's own tracking sample's.
    """

    def __init__(self, int particle_num=600, int max_particle_num=1000,
                 float bin_size=0.1, double epsilon=0.2, double delta=0.99,
                 double step_noise=0.015, double resolution=0.01,
                 double maximum_distance=0.05, int threads=1,
                 bint exact_coherence=False):
        _check_common(particle_num, step_noise, resolution, threads)
        if max_particle_num < particle_num:
            raise ValueError(
                "max_particle_num (%d) must be >= particle_num (%d)"
                % (max_particle_num, particle_num))
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0, got %r" % bin_size)
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0, got %r" % epsilon)
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1), got %r" % delta)
        self.particle_num_ = particle_num
        self.step_noise = step_noise
        self.resolution = resolution
        self.maximum_distance = maximum_distance
        self.threads = threads
        self.exact = exact_coherence
        self.kld = True
        self.max_particle_num = max_particle_num
        self.bin_size = bin_size
        self.epsilon = epsilon
        self.delta = delta


# --- pyramidal KLT ------------------------------------------------------------

cdef class PyramidalKLTTracker:
    """Pixels followed across ORGANIZED colour frames
    (pcl::tracking::PyramidalKLTTracker): pyramidal Lucas-Kanade on the
    intensity, plus the rigid motion the tracked points imply.

        klt = pcl.PyramidalKLTTracker()
        klt.set_InputCloud(first)
        klt.set_PointsToTrack(corners)      # (n, 2) u, v
        klt.compute()                       # the first frame is the reference
        klt.set_InputCloud(second)
        klt.compute()
        uv, status = klt.get_TrackedPoints()
        motion = klt.get_Result()           # first -> second, 4 x 4

    Tracking is frame to frame: each `compute()` moves the points from
    the previous frame and `get_Result` is that step's motion. `levels`
    is 2..5 and the window odd and at least 3 x 3 (PCL's limits, checked
    here because PCL only logs them and tracks nothing).

    PCL's own implementation barely moves a point (a 2-pixel shift reads
    as under 1, at any depth); this runs it with the three faults behind
    that corrected -- see ``CorrectedKLT`` in ``compat/tracking_args.h``.
    Points whose content leaves the frame can still lock onto the wrong
    place: there is no per-point error check.
    """

    cdef KLTTracker* me
    cdef bint has_input
    cdef bint has_points
    cdef size_t keypoints
    cdef unsigned int width
    cdef unsigned int height

    def __cinit__(self, int levels=5, int window_width=7,
                  int window_height=7):
        if not 2 <= levels <= 5:
            raise ValueError("levels must be 2..5")
        if window_width < 3 or window_height < 3 or \
                window_width % 2 == 0 or window_height % 2 == 0:
            raise ValueError("the window must be odd and at least 3 x 3")
        self.me = new KLTTracker(levels, window_width, window_height)
        self.has_input = False
        self.has_points = False
        self.keypoints = 100

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA cloud not None):
        if cloud.height <= 1:
            raise ValueError("PyramidalKLTTracker needs an organized cloud")
        self.me.setInputCloud(cloud.thisptr_shared)
        self.width = cloud.width
        self.height = cloud.height
        self.has_input = True

    def set_NumberOfKeypoints(self, size_t count):
        """At most this many points are tracked (PCL's default 100)."""
        if count == 0:
            raise ValueError("count must be >= 1")
        self.me.setNumberOfKeypoints(count)
        self.keypoints = count

    def set_PointsToTrack(self, uv):
        """``(n, 2)`` pixel coordinates (u, v) in the current input, which
        becomes the reference; only the first `set_NumberOfKeypoints` are
        kept."""
        import numpy as np
        if not self.has_input:
            raise RuntimeError("set_InputCloud() is required first")
        arr = np.asarray(uv, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            raise ValueError("expected a non-empty (n, 2) array of u, v")
        if not (np.all(arr >= 0) and np.all(arr[:, 0] < self.width)
                and np.all(arr[:, 1] < self.height)):
            raise ValueError("points must lie inside the %d x %d frame"
                             % (self.width, self.height))
        self.me.setPointsToTrack(arr.ravel().tolist())
        self.has_points = True

    def set_Accuracy(self, float accuracy):
        self.me.setAccuracy(accuracy)

    def set_Epsilon(self, float epsilon):
        self.me.setEpsilon(epsilon)

    def set_MaxIterationsNumber(self, unsigned int count):
        self.me.setMaxIterationsNumber(count)

    def set_NumberOfThreads(self, unsigned int count):
        self.me.setNumberOfThreads(count)

    def compute(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud() is required first")
        if not self.has_points:
            raise RuntimeError("set_PointsToTrack() is required first")
        with nogil:
            self.me.compute()

    @property
    def initialized(self):
        """True once a frame after the reference has been tracked."""
        return self.me.initialized()

    def get_TrackedPoints(self):
        """``(uv, status)`` for the points the last step started from:
        ``(n, 2)`` float32 positions and ``(n,)`` int32 status -- 0 for a
        point still tracked, -1 for one that left the frame, -2 for one
        whose window had too little texture. Lost rows are NaN, and the
        next step follows only the rows with status 0."""
        import numpy as np
        cdef vector[float] uv
        cdef vector[int] status
        self.me.trackedPoints(uv, status)
        codes = np.asarray(status, dtype=np.int32)
        tracked = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        if len(tracked) == len(codes):
            return tracked, codes
        # PCL keeps only the survivors, in order
        out = np.full((len(codes), 2), np.nan, dtype=np.float32)
        out[codes == 0] = tracked
        return out, codes

    def get_Result(self):
        """The rigid motion of the last step, ``(4, 4)`` float32."""
        import numpy as np
        cdef float[16] values
        self.me.result(values)
        matrix = np.empty(16, dtype=np.float32)
        cdef Py_ssize_t i
        for i in range(16):
            matrix[i] = values[i]
        return matrix.reshape(4, 4, order="F")
