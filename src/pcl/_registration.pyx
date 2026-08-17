# distutils: language = c++
# cython: language_level=3
"""Registration wrappers, named after sirokujira/python-pcl
(pcl/pxi/registration/).

Every class exposes the same shape as python-pcl: one call that takes
source and target and returns

    (converged, transform, estimate, fitness)

where *transform* is a 4x4 float32 array in Fortran order (Eigen is
column-major, so that is the layout PCL already has), *estimate* is the
transformed source cloud, and *fitness* is the mean squared distance to
the target.

`NormalDistributionsTransform` is here too, which python-pcl never had —
strawlab/python-pcl#265 asked for it and the repository is archived.

`getFinalTransformation` returns an `Eigen::Matrix4f`, which neither a
mirror header nor Cython can name; pcl/compat/eigen_results.h copies the
16 floats out. Same rung-4 shim pattern as the grabber callback.
"""

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.registration.icp cimport IterativeClosestPoint as cICP
from pcl.pxd.registration.icp_nl cimport (
    IterativeClosestPointNonLinear as cICPNL)
from pcl.pxd.registration.gicp cimport (
    GeneralizedIterativeClosestPoint as cGICP)
from pcl.pxd.registration.ndt cimport NormalDistributionsTransform as cNDT
from pcl.pxd.compat.eigen_results cimport finalTransformation
from pcl.pxd.compat.global_alignment_args cimport globalAlignment

from pcl._pointcloud cimport PointCloud


cdef object _as_matrix(float* buf):
    """The 16 floats the shim wrote, as a 4x4 Fortran-ordered array."""
    import numpy as np
    transf = np.empty((4, 4), dtype=np.float32, order="F")
    cdef float[::1] flat = transf.reshape(-1, order="A")
    cdef int i
    for i in range(16):
        flat[i] = buf[i]
    return transf


cdef class IterativeClosestPoint:
    """Point-to-point ICP (pcl::IterativeClosestPoint)."""

    cdef cICP[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cICP[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_EuclideanFitnessEpsilon(self, double epsilon):
        self.me.setEuclideanFitnessEpsilon(epsilon)

    def set_UseReciprocalCorrespondences(self, bint value):
        self.me.setUseReciprocalCorrespondences(value)

    def icp(self, PointCloud source not None, PointCloud target not None,
            max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class IterativeClosestPointNonLinear:
    """ICP with a Levenberg-Marquardt transformation estimator
    (pcl::IterativeClosestPointNonLinear)."""

    cdef cICPNL[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cICPNL[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def icp_nl(self, PointCloud source not None, PointCloud target not None,
               max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class GeneralizedIterativeClosestPoint:
    """Plane-to-plane ICP (pcl::GeneralizedIterativeClosestPoint)."""

    cdef cGICP[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cGICP[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_RotationEpsilon(self, double epsilon):
        self.me.setRotationEpsilon(epsilon)

    def set_CorrespondenceRandomness(self, int k):
        self.me.setCorrespondenceRandomness(k)

    def set_MaximumOptimizerIterations(self, int count):
        self.me.setMaximumOptimizerIterations(count)

    def gicp(self, PointCloud source not None, PointCloud target not None,
             max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class NormalDistributionsTransform:
    """NDT registration (pcl::NormalDistributionsTransform).

    Matches voxelised normal distributions rather than point pairs, which
    makes it the usual choice for large scans where ICP is too slow.
    Unlike the others it needs a voxel `resolution` to be useful; the PCL
    tutorial default is 1.0 with a step size of 0.1.
    """

    cdef cNDT[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cNDT[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_Resolution(self, float resolution):
        """Voxel size of the target's normal-distribution grid."""
        self.me.setResolution(resolution)

    def get_Resolution(self):
        return self.me.getResolution()

    def set_StepSize(self, double step_size):
        """More-Thuente line-search maximum step length."""
        self.me.setStepSize(step_size)

    def get_StepSize(self):
        return self.me.getStepSize()

    def set_OulierRatio(self, double outlier_ratio):
        """Expected fraction of outliers (PCL's spelling of `outlier`)."""
        self.me.setOulierRatio(outlier_ratio)

    def get_OulierRatio(self):
        return self.me.getOulierRatio()

    def get_TransformationProbability(self):
        return self.me.getTransformationProbability()

    def get_FinalNumIteration(self):
        return self.me.getFinalNumIteration()

    def ndt(self, PointCloud source not None, PointCloud target not None,
            max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class SampleConsensusPrerejective:
    """Align two clouds with no initial guess
    (pcl::SampleConsensusPrerejective).

    Every other algorithm here REFINES a pose that is already roughly
    right; point ICP at clouds metres and radians apart and it converges
    on nonsense. This one samples correspondences from FPFH descriptors,
    throws out the geometrically impossible ones with a polygon test,
    and keeps the pose with the best inlier fraction — so it needs no
    starting point at all:

        def describe(cloud):
            normals = cloud.make_NormalEstimation()
            normals.set_KSearch(15)
            fpfh = cloud.make_FPFHEstimation()
            fpfh.set_InputNormals(normals.compute_cloud())
            fpfh.set_RadiusSearch(0.12)
            return fpfh.compute()

        align = pcl.SampleConsensusPrerejective()
        align.set_MaxCorrespondenceDistance(0.05)
        converged, transform, estimate, fitness = align.align(
            source, describe(source), target, describe(target))

    The usual next step is to hand `transform` to ICP as a starting
    point, or apply it with `source.transform(transform)`.

    `max_correspondence_distance` — how close a transformed source point
    must land to count as an inlier — is the setting that decides
    whether this finds anything, and it is in the cloud's own units, so
    it has no useful default. `inlier_fraction` is how many points must
    qualify before a pose is accepted at all.
    """

    cdef int max_iterations
    cdef int number_of_samples
    cdef int correspondence_randomness
    cdef float similarity_threshold
    cdef float max_correspondence_distance
    cdef float inlier_fraction
    cdef object last_inliers

    def __cinit__(self):
        self.max_iterations = 50000
        self.number_of_samples = 3
        self.correspondence_randomness = 5
        self.similarity_threshold = 0.9
        self.max_correspondence_distance = 0.05
        self.inlier_fraction = 0.25
        self.last_inliers = None

    def set_MaximumIterations(self, int iterations):
        if iterations <= 0:
            raise ValueError(
                "max_iterations must be > 0, got %r" % iterations)
        self.max_iterations = iterations

    def get_MaximumIterations(self):
        return self.max_iterations

    def set_NumberOfSamples(self, int samples):
        """Correspondences sampled per RANSAC iteration. Three is PCL's
        floor and its default: fewer cannot determine a 6-DOF pose."""
        if samples < 3:
            raise ValueError(
                "number_of_samples must be >= 3 (a 6-DOF pose needs three "
                "correspondences), got %r" % samples)
        self.number_of_samples = samples

    def get_NumberOfSamples(self):
        return self.number_of_samples

    def set_CorrespondenceRandomness(self, int k):
        """How many nearest descriptors to pick a match from at random —
        1 is strict nearest-neighbour, higher trades precision for the
        chance to escape a wrong best match."""
        if k <= 0:
            raise ValueError(
                "correspondence_randomness must be > 0, got %r" % k)
        self.correspondence_randomness = k

    def get_CorrespondenceRandomness(self):
        return self.correspondence_randomness

    def set_SimilarityThreshold(self, float threshold):
        """Polygon-test edge-length similarity, in [0, 1). Zero disables
        the pre-rejection this algorithm is named for."""
        if not 0.0 <= threshold < 1.0:
            raise ValueError(
                "similarity_threshold must be in [0, 1), got %r" % threshold)
        self.similarity_threshold = threshold

    def get_SimilarityThreshold(self):
        return self.similarity_threshold

    def set_MaxCorrespondenceDistance(self, float distance):
        """Inlier threshold, in the cloud's own units."""
        if distance <= 0:
            raise ValueError(
                "max_correspondence_distance must be > 0, got %r" % distance)
        self.max_correspondence_distance = distance

    def get_MaxCorrespondenceDistance(self):
        return self.max_correspondence_distance

    def set_InlierFraction(self, float fraction):
        """Fraction of source points that must be inliers before a pose
        is accepted."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "inlier_fraction must be in (0, 1], got %r" % fraction)
        self.inlier_fraction = fraction

    def get_InlierFraction(self):
        return self.inlier_fraction

    def get_Inliers(self):
        """Indices into the source cloud that supported the last pose,
        or None before the first `align()`."""
        return self.last_inliers

    def align(self, PointCloud source not None, source_features,
              PointCloud target not None, target_features):
        """Align *source* onto *target* from descriptors alone.

        *source_features* and *target_features* are the ``(n, 33)``
        float32 arrays `FPFHEstimation.compute()` returns, one row per
        point of the matching cloud.

        Returns ``(converged, transform, estimate, fitness)``, the same
        shape as ICP.
        """
        import numpy as np

        src = np.ascontiguousarray(source_features, dtype=np.float32)
        tgt = np.ascontiguousarray(target_features, dtype=np.float32)
        for name, values, cloud in (("source", src, source),
                                    ("target", tgt, target)):
            if values.ndim != 2 or values.shape[1] != 33:
                raise ValueError(
                    "%s_features must be an (n, 33) FPFH array, got shape %r"
                    % (name, (values.shape,)))
            if values.shape[0] != cloud.size:
                raise ValueError(
                    "%s_features must have one row per point: cloud %d, "
                    "features %d" % (name, cloud.size, values.shape[0]))

        cdef vector[float] source_flat
        cdef vector[float] target_flat
        cdef float[::1] source_view = src.reshape(-1)
        cdef float[::1] target_view = tgt.reshape(-1)
        cdef Py_ssize_t i
        source_flat.resize(source_view.shape[0])
        for i in range(source_view.shape[0]):
            source_flat[i] = source_view[i]
        target_flat.resize(target_view.shape[0])
        for i in range(target_view.shape[0]):
            target_flat[i] = target_view[i]

        cdef float buf[16]
        cdef vector[int] inliers
        cdef double fitness = 0.0
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()
        cdef bint converged
        with nogil:
            converged = globalAlignment(
                source.thisptr_shared, source_flat,
                target.thisptr_shared, target_flat,
                self.max_iterations, self.number_of_samples,
                self.correspondence_randomness, self.similarity_threshold,
                self.max_correspondence_distance, self.inlier_fraction,
                deref(out), buf, inliers, &fitness)

        self.last_inliers = [inliers[i]
                             for i in range(<Py_ssize_t> inliers.size())]
        return (bool(converged), _as_matrix(buf), estimate, fitness)
