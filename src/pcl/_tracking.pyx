# distutils: language = c++
# cython: language_level=3
"""Particle-filter tracking of a known object through a point cloud
stream.

    tracker = pcl.ParticleFilterTracker(particle_num=600)
    tracker.set_ReferenceCloud(model)
    for frame in frames:
        tracker.set_InputCloud(frame)
        tracker.compute()
        x, y, z, roll, pitch, yaw, weight = tracker.get_Result()

python-pcl declared `pcl_tracking_*.pxd` but never wrapped any of it (its
`pxi/Tracking/AddList.txt` is empty), so the names here follow PCL's own.

The whole chain lives in pcl/compat/tracking_args.h: PCL needs a tracker,
a cloud coherence, a point coherence inside it, a search method inside
that, and an Eigen initial pose, all as shared_ptr to class templates.
That header also explains why this wraps the OMP tracker specifically —
the plain `ParticleFilterTracker` never resamples.
"""

from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.compat.tracking_args cimport ParticleTracker

from pcl._pointcloud cimport PointCloud


cdef class ParticleFilterTracker:
    """Follow a rigid object with a particle filter
    (pcl::tracking::ParticleFilterOMPTracker).

    Each particle is a candidate 6-DOF pose; every frame they are
    reweighted by how well the reference cloud placed at that pose
    explains the scene, then resampled. `particle_num` buys accuracy with
    time, `step_noise` is how far a particle may drift per frame — set it
    near the object's real per-frame motion.
    """

    cdef shared_ptr[ParticleTracker] thisptr
    cdef bint has_reference
    cdef bint has_input

    def __cinit__(self, *args, **kwargs):
        # Built in __init__, so KLDAdaptiveParticleFilterTracker can take
        # its own constructor arguments.
        self.has_reference = False
        self.has_input = False

    def __init__(self, int particle_num=400, double step_noise=0.015,
                 double resolution=0.01, double maximum_distance=0.05,
                 int threads=1):
        _check_common(particle_num, step_noise, resolution, threads)
        self.thisptr = make_shared[ParticleTracker](
            particle_num, step_noise, resolution, maximum_distance, threads)

    cdef ParticleTracker* _tracker(self) except NULL:
        if not self.thisptr:
            raise RuntimeError("tracker was never initialised")
        return self.thisptr.get()

    def set_ReferenceCloud(self, PointCloud reference not None):
        """The object to follow.

        PCL wants this in the object's own frame; the shim recentres it
        and keeps the centroid as the initial pose, so the poses this
        tracker reports are in the scene's coordinates.
        """
        if reference.size == 0:
            raise ValueError("reference cloud is empty")
        deref(self._tracker()).setReferenceCloud(reference.thisptr_shared)
        self.has_reference = True

    def set_InputCloud(self, PointCloud cloud not None):
        """The next frame to track through."""
        deref(self._tracker()).setInputCloud(cloud.thisptr_shared)
        self.has_input = True

    @property
    def particle_num(self):
        return deref(self._tracker()).particleNum()

    def compute(self):
        """Advance the filter by one frame."""
        if not self.has_reference:
            raise RuntimeError(
                "set_ReferenceCloud() is required before compute()")
        if not self.has_input:
            raise RuntimeError("set_InputCloud() is required before compute()")
        cdef ParticleTracker* tracker = self._tracker()
        with nogil:
            tracker.compute()

    def get_Result(self):
        """Return ``(x, y, z, roll, pitch, yaw, weight)``.

        The pose is in the scene's coordinates; `weight` is the winning
        particle's normalized likelihood.
        """
        import numpy as np
        cdef float[7] state
        deref(self._tracker()).result(state)
        result = np.empty(7, dtype=np.float32)
        cdef Py_ssize_t i
        for i in range(7):
            result[i] = state[i]
        return tuple(float(v) for v in result)

    def get_ResultTransform(self):
        """The same pose as a ``(4, 4)`` float32 matrix."""
        import numpy as np
        cdef float[16] values
        deref(self._tracker()).resultTransform(values)
        matrix = np.empty(16, dtype=np.float32)
        cdef Py_ssize_t i
        for i in range(16):
            matrix[i] = values[i]
        # Column-major, which is how Eigen stored it.
        return matrix.reshape(4, 4, order="F")

    def get_AlignedReference(self):
        """The reference cloud placed at the tracked pose."""
        cdef vector[PointXYZ] points
        deref(self._tracker()).alignedReference(points)

        cdef PointCloud result = PointCloud()
        cdef Py_ssize_t n = <Py_ssize_t> points.size()
        result.resize(n)
        cdef Py_ssize_t i
        for i in range(n):
            deref(result.ptr())[<size_t> i] = points[i]
        return result


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
    (confidence) is met over state-space bins of `bin_size` — up to
    `max_particle_num`. A confident track runs on few particles, a lost
    one spreads out. Same methods as :class:`ParticleFilterTracker`;
    `particle_num` reports the current count. Defaults are PCL's own
    tracking sample's.
    """

    def __init__(self, int particle_num=600, int max_particle_num=1000,
                 float bin_size=0.1, double epsilon=0.2, double delta=0.99,
                 double step_noise=0.015, double resolution=0.01,
                 double maximum_distance=0.05, int threads=1):
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
        self.thisptr = make_shared[ParticleTracker](
            particle_num, step_noise, resolution, maximum_distance, threads,
            max_particle_num, bin_size, epsilon, delta)
