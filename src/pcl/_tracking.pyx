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

    def __cinit__(self, int particle_num=400, double step_noise=0.015,
                  double resolution=0.01, double maximum_distance=0.05,
                  int threads=1):
        if particle_num <= 0:
            raise ValueError(
                "particle_num must be > 0, got %r" % particle_num)
        if step_noise <= 0:
            raise ValueError("step_noise must be > 0, got %r" % step_noise)
        if resolution <= 0:
            raise ValueError("resolution must be > 0, got %r" % resolution)
        if threads <= 0:
            raise ValueError("threads must be > 0, got %r" % threads)

        self.thisptr = make_shared[ParticleTracker](
            particle_num, step_noise, resolution, maximum_distance, threads)
        self.has_reference = False
        self.has_input = False

    def set_ReferenceCloud(self, PointCloud reference not None):
        """The object to follow.

        PCL wants this in the object's own frame; the shim recentres it
        and keeps the centroid as the initial pose, so the poses this
        tracker reports are in the scene's coordinates.
        """
        if reference.size == 0:
            raise ValueError("reference cloud is empty")
        deref(self.thisptr).setReferenceCloud(reference.thisptr_shared)
        self.has_reference = True

    def set_InputCloud(self, PointCloud cloud not None):
        """The next frame to track through."""
        deref(self.thisptr).setInputCloud(cloud.thisptr_shared)
        self.has_input = True

    @property
    def particle_num(self):
        return deref(self.thisptr).particleNum()

    def compute(self):
        """Advance the filter by one frame."""
        if not self.has_reference:
            raise RuntimeError(
                "set_ReferenceCloud() is required before compute()")
        if not self.has_input:
            raise RuntimeError("set_InputCloud() is required before compute()")
        cdef ParticleTracker* tracker = self.thisptr.get()
        with nogil:
            tracker.compute()

    def get_Result(self):
        """Return ``(x, y, z, roll, pitch, yaw, weight)``.

        The pose is in the scene's coordinates; `weight` is the winning
        particle's normalized likelihood.
        """
        import numpy as np
        cdef float[7] state
        deref(self.thisptr).result(state)
        result = np.empty(7, dtype=np.float32)
        cdef Py_ssize_t i
        for i in range(7):
            result[i] = state[i]
        return tuple(float(v) for v in result)

    def get_ResultTransform(self):
        """The same pose as a ``(4, 4)`` float32 matrix."""
        import numpy as np
        cdef float[16] values
        deref(self.thisptr).resultTransform(values)
        matrix = np.empty(16, dtype=np.float32)
        cdef Py_ssize_t i
        for i in range(16):
            matrix[i] = values[i]
        # Column-major, which is how Eigen stored it.
        return matrix.reshape(4, 4, order="F")

    def get_AlignedReference(self):
        """The reference cloud placed at the tracked pose."""
        cdef vector[PointXYZ] points
        deref(self.thisptr).alignedReference(points)

        cdef PointCloud result = PointCloud()
        cdef Py_ssize_t n = <Py_ssize_t> points.size()
        result.resize(n)
        cdef Py_ssize_t i
        for i in range(n):
            deref(result.ptr())[<size_t> i] = points[i]
        return result
