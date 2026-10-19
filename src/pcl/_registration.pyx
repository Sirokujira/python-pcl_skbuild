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
from libcpp.memory cimport shared_ptr

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.registration.icp cimport IterativeClosestPoint as cICP
from pcl.pxd.registration.icp_nl cimport (
    IterativeClosestPointNonLinear as cICPNL)
from pcl.pxd.registration.gicp cimport (
    GeneralizedIterativeClosestPoint as cGICP)
from pcl.pxd.registration.ndt cimport NormalDistributionsTransform as cNDT
from pcl.pxd.compat.eigen_results cimport finalTransformation
from pcl.pxd.compat.global_alignment_args cimport (
    globalAlignment, initialAlignment)
from pcl.pxd.point_types cimport PointNormal, Normal
from pcl.pxd.registration.icp cimport (
    IterativeClosestPointWithNormals as cICPWithNormals)
from pcl.pxd.registration.ia_fpcs cimport (
    FPCSInitialAlignment as cFPCSInitialAlignment)
from pcl.pxd.registration.ia_kfpcs cimport (
    KFPCSInitialAlignment as cKFPCSInitialAlignment)
from pcl.pxd.compat.registration_args cimport (
    estimateCorrespondences, rejectOneToOne, rejectDistance,
    rejectMedianDistance, rejectTrimmed, rejectSampleConsensus,
    rejectSurfaceNormal, estimateTransformation,
    validateTransformationEuclidean)
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointNormal, wrap_pointnormal_cloud)

from pcl._pointcloud cimport PointCloud
from pcl.pxd.point_types cimport PointXYZRGBA
from pcl.pxd.compat.registration_extra cimport (
    ELCHLoop, IncrementalICP, LUMGraph, MetaICP,
    correspondencesBackProjection, correspondencesNormalShooting,
    correspondencesOrganizedProjection, estimateTransformationExtra,
    estimateTransformationNormals, gicp6dAlign, jointICPAlign, ndt2dAlign,
    ppfRegister, pyramidSimilarity, rejectOrganizedBoundary, rejectPoly,
    rejectVarTrimmed)
from pcl._pointtypes cimport PointCloud_PointXYZRGBA, wrap_xyzrgba_cloud


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


cdef _matrix16(matrix, float* buf):
    """Flatten a 4x4 matrix column-major (Eigen's storage) into *buf*."""
    import numpy as np
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape != (4, 4):
        raise ValueError("expected a 4x4 matrix, got shape %s" % (m.shape,))
    cdef int row, col
    for col in range(4):
        for row in range(4):
            buf[col * 4 + row] = m[row, col]


cdef class IterativeClosestPointWithNormals:
    """Point-to-plane ICP on oriented clouds
    (pcl::IterativeClosestPointWithNormals).

    Minimises the distance from each source point to the TARGET's
    tangent plane instead of to the nearest point, which converges in far
    fewer iterations on smooth surfaces. Both clouds are
    :class:`PointCloud_PointNormal` (build one with
    `PointCloud_PointNormal().from_xyz_and_normals(cloud, normals)`).
    `set_UseSymmetricObjective(True)` uses both clouds' normals;
    `set_EnforceSameDirectionNormals` flips the source normals to agree
    with the target's.
    """

    cdef cICPWithNormals[PointNormal, PointNormal]* me

    def __cinit__(self):
        self.me = new cICPWithNormals[PointNormal, PointNormal]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud_PointNormal cloud not None):
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

    def set_UseSymmetricObjective(self, bint value):
        self.me.setUseSymmetricObjective(value)

    def get_UseSymmetricObjective(self):
        return self.me.getUseSymmetricObjective()

    def set_EnforceSameDirectionNormals(self, bint value):
        self.me.setEnforceSameDirectionNormals(value)

    def get_EnforceSameDirectionNormals(self):
        return self.me.getEnforceSameDirectionNormals()

    def icp(self, PointCloud_PointNormal source not None,
            PointCloud_PointNormal target not None, max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)`` like
        `IterativeClosestPoint.icp`, with `estimate` a
        :class:`PointCloud_PointNormal`.
        """
        if source.size == 0 or target.size == 0:
            raise ValueError("both clouds must have points")
        cdef float buf[16]
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef cPointCloud[PointNormal]* out = holder.get()
        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf),
                wrap_pointnormal_cloud(holder), self.me.getFitnessScore())


cdef class SampleConsensusInitialAlignment:
    """SAC-IA: the classic descriptor-driven initial alignment
    (pcl::SampleConsensusInitialAlignment).

    Like `SampleConsensusPrerejective` it needs no starting pose and
    takes the same FPFH ``(n, 33)`` descriptor arrays; it samples
    `number_of_samples` points at least `min_sample_distance` apart,
    matches each to one of its `correspondence_randomness` nearest
    descriptors, and keeps the pose with the lowest error. Older and
    slower than the prerejective variant, but with a different failure
    mode: no inlier-fraction gate, so it always returns its best guess.
    `max_correspondence_distance` is in the cloud's own units.
    """

    cdef float min_sample_distance
    cdef float max_correspondence_distance
    cdef int max_iterations
    cdef int number_of_samples
    cdef int correspondence_randomness

    def __cinit__(self):
        self.min_sample_distance = 0.0
        self.max_correspondence_distance = 0.05
        self.max_iterations = 500
        self.number_of_samples = 3
        self.correspondence_randomness = 10

    def set_MinSampleDistance(self, float distance):
        if distance < 0:
            raise ValueError("min_sample_distance cannot be negative")
        self.min_sample_distance = distance

    def get_MinSampleDistance(self):
        return self.min_sample_distance

    def set_MaxCorrespondenceDistance(self, float distance):
        if distance <= 0:
            raise ValueError("max_correspondence_distance must be > 0")
        self.max_correspondence_distance = distance

    def get_MaxCorrespondenceDistance(self):
        return self.max_correspondence_distance

    def set_MaximumIterations(self, int iterations):
        if iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self.max_iterations = iterations

    def get_MaximumIterations(self):
        return self.max_iterations

    def set_NumberOfSamples(self, int samples):
        if samples < 3:
            raise ValueError(
                "number_of_samples must be >= 3 (a 6-DOF pose needs three)")
        self.number_of_samples = samples

    def get_NumberOfSamples(self):
        return self.number_of_samples

    def set_CorrespondenceRandomness(self, int k):
        if k <= 0:
            raise ValueError("correspondence_randomness must be > 0")
        self.correspondence_randomness = k

    def get_CorrespondenceRandomness(self):
        return self.correspondence_randomness

    def align(self, PointCloud source not None, source_features,
              PointCloud target not None, target_features):
        """Align *source* onto *target* using the FPFH descriptors of
        each. Returns ``(converged, transform, estimate, fitness)``."""
        cdef vector[float] sf = _flat_descriptors(source_features, source.size,
                                                  "source")
        cdef vector[float] tf = _flat_descriptors(target_features, target.size,
                                                  "target")
        cdef float buf[16]
        cdef double fitness = 0.0
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()
        cdef bint converged
        with nogil:
            converged = initialAlignment(
                source.thisptr_shared, sf, target.thisptr_shared, tf,
                self.min_sample_distance, self.max_correspondence_distance,
                self.max_iterations, self.number_of_samples,
                self.correspondence_randomness, deref(out), buf, &fitness)
        return converged, _as_matrix(buf), estimate, fitness


cdef vector[float] _flat_descriptors(features, Py_ssize_t n_points,
                                     str which) except *:
    """An ``(n, 33)`` float array as the flat vector the shim takes."""
    import numpy as np
    arr = np.ascontiguousarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 33:
        raise ValueError(
            "%s descriptors must be an (n, 33) FPFH array, got shape %s"
            % (which, arr.shape))
    if arr.shape[0] != n_points:
        raise ValueError(
            "%s has %d points but %d descriptors"
            % (which, n_points, arr.shape[0]))
    cdef vector[float] flat
    cdef float[:, ::1] view = arr
    cdef Py_ssize_t i, j
    flat.reserve(arr.shape[0] * 33)
    for i in range(arr.shape[0]):
        for j in range(33):
            flat.push_back(view[i, j])
    return flat


cdef class FPCSInitialAlignment:
    """4-Points Congruent Sets (pcl::registration::FPCSInitialAlignment):
    a global registration needing no initial guess and no descriptors.

    Matches coplanar 4-point bases between the clouds. `set_Delta` is
    the tolerance in the cloud's own units (or, with ``normalize=True``,
    as a fraction of the mean point spacing), `set_ApproxOverlap` the
    expected overlap fraction (0..1). `align(source, target)` returns
    ``(converged, transform, estimate, fitness)``.
    """

    cdef cFPCSInitialAlignment[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cFPCSInitialAlignment[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_Delta(self, float delta, bint normalize=False):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.me.setDelta(delta, normalize)

    def get_Delta(self):
        return self.me.getDelta()

    def set_ApproxOverlap(self, float overlap):
        if not 0.0 < overlap <= 1.0:
            raise ValueError("approx_overlap must be in (0, 1]")
        self.me.setApproxOverlap(overlap)

    def get_ApproxOverlap(self):
        return self.me.getApproxOverlap()

    def set_ScoreThreshold(self, float threshold):
        self.me.setScoreThreshold(threshold)

    def get_ScoreThreshold(self):
        return self.me.getScoreThreshold()

    def set_NumberOfSamples(self, int samples):
        if samples <= 0:
            raise ValueError("number_of_samples must be > 0")
        self.me.setNumberOfSamples(samples)

    def set_MaxNormalDifference(self, float degrees):
        self.me.setMaxNormalDifference(degrees)

    def set_MaxComputationTime(self, int seconds):
        self.me.setMaxComputationTime(seconds)

    def set_NumberOfThreads(self, int threads):
        self.me.setNumberOfThreads(threads)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def align(self, PointCloud source not None, PointCloud target not None):
        """Returns ``(converged, transform, estimate, fitness)``."""
        if source.size < 4 or target.size < 4:
            raise ValueError("both clouds need at least four points")
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()
        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class KFPCSInitialAlignment:
    """Keypoint-based 4PCS (pcl::registration::KFPCSInitialAlignment):
    FPCS with translation bounds and a weighted candidate score, meant
    for keypoint clouds. Same settings as `FPCSInitialAlignment` plus
    `set_UpperTranslationThreshold` / `set_LowerTranslationThreshold`
    and `set_Lambda`.
    """

    cdef cKFPCSInitialAlignment[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cKFPCSInitialAlignment[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_Delta(self, float delta, bint normalize=False):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.me.setDelta(delta, normalize)

    def get_Delta(self):
        return self.me.getDelta()

    def set_ApproxOverlap(self, float overlap):
        if not 0.0 < overlap <= 1.0:
            raise ValueError("approx_overlap must be in (0, 1]")
        self.me.setApproxOverlap(overlap)

    def get_ApproxOverlap(self):
        return self.me.getApproxOverlap()

    def set_ScoreThreshold(self, float threshold):
        self.me.setScoreThreshold(threshold)

    def get_ScoreThreshold(self):
        return self.me.getScoreThreshold()

    def set_NumberOfSamples(self, int samples):
        if samples <= 0:
            raise ValueError("number_of_samples must be > 0")
        self.me.setNumberOfSamples(samples)

    def set_MaxNormalDifference(self, float degrees):
        self.me.setMaxNormalDifference(degrees)

    def set_MaxComputationTime(self, int seconds):
        self.me.setMaxComputationTime(seconds)

    def set_NumberOfThreads(self, int threads):
        self.me.setNumberOfThreads(threads)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_UpperTranslationThreshold(self, float value):
        self.me.setUpperTranslationThreshold(value)

    def get_UpperTranslationThreshold(self):
        return self.me.getUpperTranslationThreshold()

    def set_LowerTranslationThreshold(self, float value):
        self.me.setLowerTranslationThreshold(value)

    def get_LowerTranslationThreshold(self):
        return self.me.getLowerTranslationThreshold()

    def set_Lambda(self, float value):
        self.me.setLambda(value)

    def get_Lambda(self):
        return self.me.getLambda()

    def align(self, PointCloud source not None, PointCloud target not None):
        """Returns ``(converged, transform, estimate, fitness)``."""
        if source.size < 4 or target.size < 4:
            raise ValueError("both clouds need at least four points")
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()
        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


# --- the correspondence pipeline, as functions ---------------------------
#
# A correspondence set is a pair of arrays: ``pairs`` (n, 2) int32 with a
# source index and a target index per row, and ``distances`` (n,) float32
# holding the SQUARED distance between them, as PCL stores it.

cdef _pairs_in(pairs, distances, vector[int]& out_pairs,
               vector[float]& out_distances):
    import numpy as np
    p = np.ascontiguousarray(pairs, dtype=np.int32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("pairs must be an (n, 2) integer array")
    cdef int[:, ::1] pv = p
    cdef Py_ssize_t i
    out_pairs.clear()
    out_pairs.reserve(2 * p.shape[0])
    for i in range(p.shape[0]):
        out_pairs.push_back(pv[i, 0])
        out_pairs.push_back(pv[i, 1])
    out_distances.clear()
    if distances is None:
        return
    d = np.ascontiguousarray(distances, dtype=np.float32)
    if d.ndim != 1 or d.shape[0] != p.shape[0]:
        raise ValueError("distances must be an (n,) array matching pairs")
    cdef float[::1] dv = d
    out_distances.reserve(d.shape[0])
    for i in range(d.shape[0]):
        out_distances.push_back(dv[i])


cdef _pairs_out(vector[int]& pairs, vector[float]& distances):
    import numpy as np
    cdef Py_ssize_t n = <Py_ssize_t> pairs.size() // 2
    p = np.empty((n, 2), dtype=np.int32)
    d = np.empty(n, dtype=np.float32)
    cdef int[:, ::1] pv = p
    cdef float[::1] dv = d
    cdef Py_ssize_t i
    for i in range(n):
        pv[i, 0] = pairs[2 * i]
        pv[i, 1] = pairs[2 * i + 1]
        dv[i] = distances[i]
    return p, d


def estimate_correspondences(PointCloud source not None,
                             PointCloud target not None,
                             max_distance=None, bint reciprocal=False):
    """Nearest-neighbour correspondences from *source* into *target*
    (pcl::registration::CorrespondenceEstimation).

    Returns ``(pairs, distances)``: one row per source point (or fewer
    with `max_distance`, in the cloud's own units, or `reciprocal`, which
    keeps only mutual nearest neighbours). Distances are SQUARED.
    """
    cdef double limit = 0.0 if max_distance is None else float(max_distance)
    if max_distance is not None and limit <= 0:
        raise ValueError("max_distance must be positive")
    cdef vector[int] pairs
    cdef vector[float] distances
    with nogil:
        estimateCorrespondences(source.thisptr_shared, target.thisptr_shared,
                                limit, reciprocal, pairs, distances)
    return _pairs_out(pairs, distances)


def reject_one_to_one(pairs, distances):
    """Keep, for every target point, only its closest source point
    (pcl::registration::CorrespondenceRejectorOneToOne)."""
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    rejectOneToOne(p_in, d_in, p_out, d_out)
    return _pairs_out(p_out, d_out)


def reject_by_distance(pairs, distances, float max_distance):
    """Drop correspondences farther apart than `max_distance` (in the
    cloud's own units, NOT squared)
    (pcl::registration::CorrespondenceRejectorDistance)."""
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    rejectDistance(p_in, d_in, max_distance, p_out, d_out)
    return _pairs_out(p_out, d_out)


def reject_median_distance(pairs, distances, double factor=1.0):
    """Drop correspondences farther than `factor` times the median
    distance (pcl::registration::CorrespondenceRejectorMedianDistance)."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    rejectMedianDistance(p_in, d_in, factor, p_out, d_out)
    return _pairs_out(p_out, d_out)


def reject_trimmed(pairs, distances, float overlap_ratio,
                   unsigned int min_correspondences=0):
    """Keep the closest `overlap_ratio` fraction (0..1), at least
    `min_correspondences` of them
    (pcl::registration::CorrespondenceRejectorTrimmed)."""
    if not 0.0 < overlap_ratio <= 1.0:
        raise ValueError("overlap_ratio must be in (0, 1]")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    rejectTrimmed(p_in, d_in, overlap_ratio, min_correspondences, p_out, d_out)
    return _pairs_out(p_out, d_out)


def reject_sample_consensus(PointCloud source not None,
                            PointCloud target not None, pairs, distances,
                            double inlier_threshold, int max_iterations=1000,
                            bint refine=False):
    """Keep the correspondences consistent with one rigid transform,
    found by RANSAC over them
    (pcl::registration::CorrespondenceRejectorSampleConsensus).

    Returns ``(pairs, distances, transform)``: the inliers and the 4x4
    transform they agree on (identity when nothing survived).
    `inlier_threshold` is in the cloud's own units.
    """
    if inlier_threshold <= 0:
        raise ValueError("inlier_threshold must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    cdef float buf[16]
    _pairs_in(pairs, distances, p_in, d_in)
    rejectSampleConsensus(source.thisptr_shared, target.thisptr_shared,
                          p_in, d_in, inlier_threshold, max_iterations,
                          refine, p_out, d_out, buf)
    kept, dists = _pairs_out(p_out, d_out)
    return kept, dists, _as_matrix(buf)


def reject_surface_normal(PointCloud source not None,
                          PointCloud target not None,
                          PointCloud_Normal source_normals not None,
                          PointCloud_Normal target_normals not None,
                          pairs, distances, double threshold):
    """Drop correspondences whose normals disagree: kept when the cosine
    of the angle between them is at least `threshold`
    (pcl::registration::CorrespondenceRejectorSurfaceNormal)."""
    if source_normals.size != source.size:
        raise ValueError("source_normals must match the source cloud")
    if target_normals.size != target.size:
        raise ValueError("target_normals must match the target cloud")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    rejectSurfaceNormal(source.thisptr_shared, target.thisptr_shared,
                        source_normals.thisptr_shared,
                        target_normals.thisptr_shared, p_in, d_in, threshold,
                        p_out, d_out)
    return _pairs_out(p_out, d_out)


def estimate_transformation(PointCloud source not None,
                            PointCloud target not None, pairs=None,
                            str method="svd"):
    """The rigid transform taking the matched source points onto their
    target points (pcl::registration::TransformationEstimation*):
    `method` ``"svd"``, ``"lm"`` (Levenberg-Marquardt),
    ``"dual_quaternion"``, ``"svd_scale"`` (similarity: rotation,
    translation AND uniform scale), ``"2d"`` (x, y and yaw only) or
    ``"3point"`` (exactly three pairs). With no `pairs`, point i matches
    point i and the clouds must be the same size. Returns a 4x4 array.
    For point-to-plane estimation see
    `estimate_transformation_point_to_plane`.
    """
    methods = {"svd": 0, "lm": 1, "dual_quaternion": 2, "svd_scale": 3,
               "2d": 4, "3point": 5}
    if method not in methods:
        raise ValueError("method must be one of %s, got %r"
                         % (", ".join(sorted(methods)), method))
    cdef vector[int] p_in
    cdef vector[float] d_in
    if pairs is None:
        if source.size != target.size:
            raise ValueError(
                "without pairs the clouds must be the same size, got %d "
                "and %d" % (source.size, target.size))
        if source.size < 3:
            raise ValueError("at least three points are needed")
    else:
        _pairs_in(pairs, None, p_in, d_in)
        if p_in.size() < 6:
            raise ValueError("at least three pairs are needed")
    cdef int m = methods[method]
    if m == 5:
        n_pairs = (p_in.size() // 2) if pairs is not None else source.size
        if n_pairs != 3:
            raise ValueError("method '3point' takes exactly three "
                             "correspondences, got %d" % n_pairs)
    cdef float buf[16]
    if m < 2:
        estimateTransformation(m, source.thisptr_shared,
                               target.thisptr_shared, p_in, buf)
    else:
        estimateTransformationExtra(m, source.thisptr_shared,
                                    target.thisptr_shared, p_in, buf)
    return _as_matrix(buf)


def validate_transformation(PointCloud source not None,
                            PointCloud target not None, transform,
                            max_range=None):
    """Score a transform by the mean squared distance of the moved
    source to the target
    (pcl::registration::TransformationValidationEuclidean); lower is
    better, zero is a perfect overlap. `max_range` ignores target points
    farther than that from a moved source point (in the cloud's own
    units). Compare the score against your own threshold; PCL's
    `isValid` does nothing more.
    """
    cdef float buf[16]
    _matrix16(transform, buf)
    cdef double limit = 0.0 if max_range is None else float(max_range)
    if max_range is not None and limit <= 0:
        raise ValueError("max_range must be positive")
    if source.size == 0 or target.size == 0:
        raise ValueError("both clouds must have points")
    return validateTransformationEuclidean(
        source.thisptr_shared, target.thisptr_shared, buf, limit)


# --- the second registration batch ------------------------------------------

def estimate_transformation_point_to_plane(
        PointCloud_PointNormal source not None,
        PointCloud_PointNormal target not None, pairs=None,
        str method="lls", weights=None):
    """The rigid transform minimising point-to-PLANE distances: each
    source point to the tangent plane of its target point
    (pcl::registration::TransformationEstimationPointToPlane*).

    `method`: ``"lls"`` (linearised least squares, the fast one),
    ``"lm"`` (Levenberg-Marquardt), ``"symmetric"`` (both clouds'
    normals), ``"lls_weighted"`` / ``"weighted"`` (one weight per pair in
    `weights`). With no `pairs`, point i matches point i. Returns 4x4.
    """
    import numpy as np
    methods = {"lm": 0, "lls": 1, "lls_weighted": 2, "symmetric": 3,
               "weighted": 4}
    if method not in methods:
        raise ValueError("method must be one of %s, got %r"
                         % (", ".join(sorted(methods)), method))
    cdef vector[int] p_in
    cdef vector[float] d_in
    cdef Py_ssize_t n_pairs
    if pairs is None:
        if source.size != target.size:
            raise ValueError("without pairs the clouds must be the same size")
        n_pairs = source.size
    else:
        _pairs_in(pairs, None, p_in, d_in)
        n_pairs = p_in.size() // 2
    if n_pairs < 3:
        raise ValueError("at least three correspondences are needed")
    cdef vector[double] w
    if methods[method] in (2, 4):
        if weights is None:
            raise ValueError("method %r needs weights, one per pair" % method)
        warr = np.asarray(weights, dtype=np.float64).ravel()
        if warr.shape[0] != n_pairs:
            raise ValueError("weights must have one value per pair (%d), got "
                             "%d" % (n_pairs, warr.shape[0]))
        w = warr.tolist()
    elif weights is not None:
        raise ValueError("weights only apply to 'lls_weighted' and 'weighted'")
    cdef float buf[16]
    estimateTransformationNormals(methods[method], source.thisptr_shared,
                                  target.thisptr_shared, p_in, w, buf)
    return _as_matrix(buf)


cdef double _max_distance(max_distance) except? -1:
    if max_distance is None:
        return 0.0
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    return max_distance


def estimate_correspondences_normal_shooting(
        PointCloud source not None, PointCloud_Normal source_normals not None,
        PointCloud target not None, int k=10, max_distance=None):
    """Correspondences along the SOURCE normals
    (pcl::registration::CorrespondenceEstimationNormalShooting): of each
    source point's `k` nearest target points, the one closest to the line
    through the point along its normal. Same ``(pairs, distances)`` shape
    as `estimate_correspondences`."""
    if source_normals.size != source.size:
        raise ValueError("source_normals must match the source cloud")
    if k < 1:
        raise ValueError("k must be >= 1")
    cdef double limit = _max_distance(max_distance)
    cdef vector[int] pairs
    cdef vector[float] distances
    with nogil:
        correspondencesNormalShooting(source.thisptr_shared,
                                      source_normals.thisptr_shared,
                                      target.thisptr_shared, k, limit, pairs,
                                      distances)
    return _pairs_out(pairs, distances)


def estimate_correspondences_back_projection(
        PointCloud source not None, PointCloud_Normal source_normals not None,
        PointCloud target not None, PointCloud_Normal target_normals not None,
        int k=10, max_distance=None):
    """Back-projection correspondences
    (pcl::registration::CorrespondenceEstimationBackProjection): of each
    source point's `k` nearest target points, the one whose normal agrees
    best with the source normal."""
    if source_normals.size != source.size:
        raise ValueError("source_normals must match the source cloud")
    if target_normals.size != target.size:
        raise ValueError("target_normals must match the target cloud")
    if k < 1:
        raise ValueError("k must be >= 1")
    cdef double limit = _max_distance(max_distance)
    cdef vector[int] pairs
    cdef vector[float] distances
    with nogil:
        correspondencesBackProjection(source.thisptr_shared,
                                      source_normals.thisptr_shared,
                                      target.thisptr_shared,
                                      target_normals.thisptr_shared, k, limit,
                                      pairs, distances)
    return _pairs_out(pairs, distances)


def estimate_correspondences_organized_projection(
        PointCloud source not None, PointCloud target not None, float fx,
        float fy, float cx, float cy, float depth_threshold=0.1,
        max_distance=None):
    """Projective data association
    (pcl::registration::CorrespondenceEstimationOrganizedProjection): each
    source point is projected through the pinhole camera ``(fx, fy, cx,
    cy)`` into the ORGANIZED target, and matched to the target pixel it
    lands on when their depths differ by less than `depth_threshold`. No
    search at all, the reason Kinect fusion pipelines use it.

    PCL truncates the projected coordinate, so pixel ``(u, v)`` covers
    ``[u, u + 1)``: a target built from pixel corners (``x = (u - cx) *
    z / fx``) sits on the boundary and float rounding sends some points
    one pixel to the left. Build it from pixel centres (``u + 0.5``)."""
    if target.ptr().height < 2:
        raise ValueError("the target must be an organized cloud")
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    cdef double limit = _max_distance(max_distance)
    cdef vector[int] pairs
    cdef vector[float] distances
    with nogil:
        correspondencesOrganizedProjection(source.thisptr_shared,
                                           target.thisptr_shared, fx, fy, cx,
                                           cy, depth_threshold, limit, pairs,
                                           distances)
    return _pairs_out(pairs, distances)


def reject_var_trimmed(PointCloud source not None, PointCloud target not None,
                       pairs, distances, double min_ratio=0.05,
                       double max_ratio=0.95):
    """Variable trimming (pcl::registration::CorrespondenceRejectorVarTrimmed,
    Chetverikov's FRMSD): picks the overlap fraction between `min_ratio`
    and `max_ratio` that minimises the trimmed error, and keeps that
    fraction of the closest pairs. Returns ``(pairs, distances,
    trimmed_distance, trim_factor)``."""
    if not 0.0 < min_ratio <= max_ratio <= 1.0:
        raise ValueError("need 0 < min_ratio <= max_ratio <= 1")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    cdef double trimmed = 0.0, factor = 0.0
    _pairs_in(pairs, distances, p_in, d_in)
    with nogil:
        rejectVarTrimmed(source.thisptr_shared, target.thisptr_shared, p_in,
                         d_in, min_ratio, max_ratio, p_out, d_out, &trimmed,
                         &factor)
    kept, dists = _pairs_out(p_out, d_out)
    return kept, dists, trimmed, factor


def reject_poly(PointCloud source not None, PointCloud target not None,
                pairs, distances, int cardinality=3,
                float similarity_threshold=0.75, int iterations=10000):
    """Polygonal consistency (pcl::registration::CorrespondenceRejectorPoly):
    random `cardinality`-gons of matched points must have edge lengths
    that agree between the clouds to within `similarity_threshold` (0..1);
    pairs that never take part in a consistent polygon are dropped."""
    if cardinality < 2:
        raise ValueError("cardinality must be >= 2")
    if not 0.0 < similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be in (0, 1)")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    with nogil:
        rejectPoly(source.thisptr_shared, target.thisptr_shared, p_in, d_in,
                   cardinality, similarity_threshold, iterations, p_out, d_out)
    return _pairs_out(p_out, d_out)


def reject_organized_boundary(PointCloud source not None,
                              PointCloud target not None, pairs, distances,
                              int boundary_nans=8):
    """Drop pairs whose TARGET point lies on a depth boundary of the
    organized target cloud
    (pcl::registration::CorrespondenceRejectionOrganizedBoundary): a point
    counts as boundary when the 5 x 5 window around it holds at least
    `boundary_nans` invalid pixels or depth jumps. The window is clipped
    at the image border without being rescaled, so a pixel on the top or
    bottom row next to a hole sees fewer invalid neighbours and may
    survive where the rows below it do not."""
    if target.ptr().height < 2:
        raise ValueError("the target must be an organized cloud")
    cdef vector[int] p_in, p_out
    cdef vector[float] d_in, d_out
    _pairs_in(pairs, distances, p_in, d_in)
    with nogil:
        rejectOrganizedBoundary(source.thisptr_shared, target.thisptr_shared,
                                p_in, d_in, boundary_nans, p_out, d_out)
    return _pairs_out(p_out, d_out)


_PYRAMID_MAX_BINS = 1 << 24


def compare_pyramid_feature_histograms(features_a, features_b,
                                       dimension_range=None,
                                       target_range=None):
    """Similarity of two SETS of low-dimensional features
    (pcl::PyramidFeatureHistogram, Grauman and Darrell's pyramid match):
    1 for identical distributions, towards 0 for unrelated ones.

    Takes ``(n, d)`` and ``(m, d)`` arrays, ``1 <= d <= 16`` -- PPF's four
    values are what PCL's own test feeds it. The histogram is DENSE: each
    pyramid level allocates the product of its per-dimension bin counts,
    so FPFH's 33 dimensions would need 2^33 counters at two bins per axis
    (reduce such descriptors first, e.g. with PCA).

    `dimension_range` is one ``(min, max)`` per dimension that the values
    are mapped from (default: the span of both sets, padded so the
    maximum is inside it); every value must lie in ``[min, max)``,
    because PCL indexes its histogram without a bounds check.
    `target_range` is the ``(min, max)`` each dimension is mapped to, and
    sets the resolution: level 0 has ``ceil(extent / sqrt(d))`` bins per
    dimension, each further level half as many (default: about 2^20
    level-0 bins in total, at most 64 per dimension).
    """
    import numpy as np
    a = np.ascontiguousarray(features_a, dtype=np.float32)
    b = np.ascontiguousarray(features_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected two (n, d) arrays with the same d")
    cdef int d = a.shape[1]
    if d < 1 or d > 16:
        raise ValueError(
            "PyramidFeatureHistogram is dense, so d must be 1..16; this "
            "is %d -- reduce the descriptor's dimension first" % d)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("both feature sets must be non-empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("features must be finite")
    if dimension_range is None:
        both = np.vstack([a, b])
        lo = both.min(axis=0)
        span = both.max(axis=0) - lo
        hi = lo + np.maximum(span * 1.001, 1e-3)
        source = np.column_stack([lo, hi]).astype(np.float32)
    else:
        source = np.asarray(dimension_range, dtype=np.float32)
        if source.shape != (d, 2) or np.any(source[:, 1] <= source[:, 0]):
            raise ValueError("dimension_range must be %d (min, max) pairs "
                             "with max > min" % d)
        for arr in (a, b):
            if np.any(arr < source[:, 0]) or np.any(arr >= source[:, 1]):
                raise ValueError(
                    "every value must lie in its dimension_range [min, max): "
                    "PCL would index past the end of its histogram")
    if target_range is None:
        per_dim = min(64, int(2 ** (20.0 / d)))
        extent = max(per_dim, 2) * np.sqrt(d)
        target = np.tile([0.0, extent], (d, 1)).astype(np.float32)
    else:
        target = np.asarray(target_range, dtype=np.float32)
        if target.shape != (d, 2) or np.any(target[:, 1] <= target[:, 0]):
            raise ValueError("target_range must be %d (min, max) pairs "
                             "with max > min" % d)
    extents = (target[:, 1] - target[:, 0]).astype(np.float64)
    if np.sqrt(np.sum(extents ** 2)) <= 2.0:
        raise ValueError("target_range is too small for even one pyramid "
                         "level (its diagonal must exceed 2)")
    bins = np.prod(np.ceil(extents / np.sqrt(d)))
    if bins > _PYRAMID_MAX_BINS:
        raise ValueError(
            "target_range gives %.3g level-0 bins (limit %d): the pyramid "
            "is a dense array" % (bins, _PYRAMID_MAX_BINS))
    cdef vector[float] va = a.ravel().tolist()
    cdef vector[float] vb = b.ravel().tolist()
    cdef vector[float] vs = source.ravel().tolist()
    cdef vector[float] vt = target.ravel().tolist()
    cdef float similarity
    with nogil:
        similarity = pyramidSimilarity(d, va, vb, vs, vt)
    return float(similarity)


cdef class _AlignSettings:
    """The ICP-style settings the align-function wrappers share."""

    cdef int max_iterations
    cdef double max_correspondence_distance
    cdef double transformation_epsilon

    def __cinit__(self, *args, **kwargs):
        self.max_iterations = 10
        self.max_correspondence_distance = 0.05
        self.transformation_epsilon = 1e-8

    def set_MaximumIterations(self, int count):
        if count < 1:
            raise ValueError("count must be >= 1")
        self.max_iterations = count

    def get_MaximumIterations(self):
        return self.max_iterations

    def set_MaxCorrespondenceDistance(self, double distance):
        if distance <= 0:
            raise ValueError("distance must be positive")
        self.max_correspondence_distance = distance

    def get_MaxCorrespondenceDistance(self):
        return self.max_correspondence_distance

    def set_TransformationEpsilon(self, double epsilon):
        self.transformation_epsilon = epsilon


cdef class GeneralizedIterativeClosestPoint6D(_AlignSettings):
    """Generalized ICP in position AND colour
    (pcl::GeneralizedIterativeClosestPoint6D): correspondences are found
    in a 6-D space of xyz plus CIELab colour scaled by `lab_weight`, then
    aligned by GICP. On :class:`PointCloud_PointXYZRGBA`; helps when the
    geometry alone is ambiguous (a flat wall with a poster on it)."""

    cdef float lab_weight

    def __cinit__(self, float lab_weight=0.032):
        if lab_weight < 0:
            raise ValueError("lab_weight must be >= 0")
        self.lab_weight = lab_weight
        self.max_iterations = 50

    def gicp(self, PointCloud_PointXYZRGBA source not None,
             PointCloud_PointXYZRGBA target not None, max_iter=None):
        """Returns ``(converged, transform, estimate, fitness)`` like
        `GeneralizedIterativeClosestPoint.gicp`, with `estimate` a
        :class:`PointCloud_PointXYZRGBA`."""
        if source.size == 0 or target.size == 0:
            raise ValueError("both clouds must have points")
        cdef int iterations = self.max_iterations if max_iter is None else max_iter
        cdef shared_ptr[cPointCloud[PointXYZRGBA]] holder
        holder.reset(new cPointCloud[PointXYZRGBA]())
        cdef float buf[16]
        cdef double fitness = 0.0
        cdef bint converged
        with nogil:
            converged = gicp6dAlign(source.thisptr_shared, target.thisptr_shared,
                                    self.lab_weight, iterations,
                                    self.max_correspondence_distance,
                                    self.transformation_epsilon,
                                    deref(holder.get()), buf, &fitness)
        return converged, _as_matrix(buf), wrap_xyzrgba_cloud(holder), fitness


cdef class NormalDistributionsTransform2D(_AlignSettings):
    """2-D NDT (pcl::NormalDistributionsTransform2D): registers the clouds'
    x, y projections on a grid of Gaussians -- planar laser scans, a
    robot on a floor. The result rotates only about z.

    `set_Grid(centre, extent, step)` defines the grid in x, y (default
    centred on the origin, 20 x 20 with 1-unit cells);
    `set_OptimizationStepSize(x, y, theta)` scales each Newton step
    (default 1, PCL's own).

    The basin is the width of the cell Gaussians: a target with no
    thickness across a structure -- noise-free walls, say -- gives
    near-singular cells whose likelihood is zero a few millimetres off
    the line, so the gradient vanishes and the result stays at the
    initial pose with `has_converged` set.
    """

    cdef float centre[2]
    cdef float extent[2]
    cdef float step[2]
    cdef double opt_step[3]

    def __cinit__(self):
        self.centre[0] = 0.0
        self.centre[1] = 0.0
        self.extent[0] = 20.0
        self.extent[1] = 20.0
        self.step[0] = 1.0
        self.step[1] = 1.0
        self.opt_step[0] = 1.0
        self.opt_step[1] = 1.0
        self.opt_step[2] = 1.0
        self.max_iterations = 35
        self.transformation_epsilon = 1e-6

    def set_Grid(self, centre, extent, step):
        cx, cy = centre
        ex, ey = extent
        sx, sy = step
        if ex <= 0 or ey <= 0 or sx <= 0 or sy <= 0:
            raise ValueError("extent and step must be positive")
        self.centre[0], self.centre[1] = cx, cy
        self.extent[0], self.extent[1] = ex, ey
        self.step[0], self.step[1] = sx, sy

    def set_OptimizationStepSize(self, double x, double y, double theta):
        self.opt_step[0], self.opt_step[1], self.opt_step[2] = x, y, theta

    def ndt(self, PointCloud source not None, PointCloud target not None,
            max_iter=None):
        """Returns ``(converged, transform, estimate, fitness)``."""
        if source.size == 0 or target.size == 0:
            raise ValueError("both clouds must have points")
        cdef int iterations = self.max_iterations if max_iter is None else max_iter
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()
        cdef float buf[16]
        cdef double fitness = 0.0
        cdef bint converged
        with nogil:
            converged = ndt2dAlign(source.thisptr_shared, target.thisptr_shared,
                                   self.centre, self.extent, self.step,
                                   self.opt_step, iterations,
                                   self.transformation_epsilon, deref(out),
                                   buf, &fitness)
        return converged, _as_matrix(buf), estimate, fitness


cdef class JointIterativeClosestPoint(_AlignSettings):
    """ICP over several source/target pairs at once, solving ONE rigid
    transform that fits them all (pcl::JointIterativeClosestPoint) --
    e.g. the same motion seen by several sensors.

        transform = pcl.JointIterativeClosestPoint().icp(
            [src_a, src_b], [tgt_a, tgt_b])[1]
    """

    def icp(self, sources, targets, max_iter=None):
        """Returns ``(converged, transform, fitness)``."""
        sources = list(sources)
        targets = list(targets)
        if not sources or len(sources) != len(targets):
            raise ValueError("give one target per source, at least one pair")
        cdef vector[shared_ptr[cPointCloud[PointXYZ]]] src, tgt
        for cloud in sources:
            if not isinstance(cloud, PointCloud) or (<PointCloud> cloud).size == 0:
                raise ValueError("sources must be non-empty PointClouds")
            src.push_back((<PointCloud> cloud).thisptr_shared)
        for cloud in targets:
            if not isinstance(cloud, PointCloud) or (<PointCloud> cloud).size == 0:
                raise ValueError("targets must be non-empty PointClouds")
            tgt.push_back((<PointCloud> cloud).thisptr_shared)
        cdef int iterations = self.max_iterations if max_iter is None else max_iter
        cdef float buf[16]
        cdef double fitness = 0.0
        cdef bint converged
        with nogil:
            converged = jointICPAlign(src, tgt, iterations,
                                      self.max_correspondence_distance,
                                      self.transformation_epsilon, buf,
                                      &fitness)
        return converged, _as_matrix(buf), fitness


cdef class PPFRegistration:
    """Point Pair Feature registration (pcl::PPFRegistration, Drost et
    al.): the model's point pair features go into a hash table, every
    scene reference point votes for model poses, and the most voted pose
    cluster wins. Needs no initial guess; model and scene are
    :class:`PointCloud_PointNormal`, downsampled first (PPF is quadratic
    in the model).

    `angle_step` (radians) and `distance_step` (cloud units) discretise
    the hash table; `scene_sampling_rate` keeps every n-th scene point as
    a reference point; the clustering thresholds merge nearby poses.
    """

    cdef float angle_step
    cdef float distance_step
    cdef unsigned int sampling_rate
    cdef float position_threshold
    cdef float rotation_threshold

    def __cinit__(self, float angle_step=0.20943951, float distance_step=0.01):
        if angle_step <= 0 or distance_step <= 0:
            raise ValueError("steps must be positive")
        self.angle_step = angle_step
        self.distance_step = distance_step
        self.sampling_rate = 10
        self.position_threshold = 0.2
        self.rotation_threshold = 0.52359878

    def set_SceneReferencePointSamplingRate(self, unsigned int rate):
        if rate < 1:
            raise ValueError("rate must be >= 1")
        self.sampling_rate = rate

    def set_PositionClusteringThreshold(self, float threshold):
        self.position_threshold = threshold

    def set_RotationClusteringThreshold(self, float radians):
        self.rotation_threshold = radians

    def align(self, PointCloud_PointNormal model not None,
              PointCloud_PointNormal scene not None):
        """Returns ``(converged, transform, estimate, fitness)``; the
        transform places the model in the scene."""
        if model.size < 2 or scene.size < 2:
            raise ValueError("model and scene need at least two points")
        cdef shared_ptr[cPointCloud[PointNormal]] holder
        holder.reset(new cPointCloud[PointNormal]())
        cdef float buf[16]
        cdef double fitness = 0.0
        cdef bint converged
        with nogil:
            converged = ppfRegister(model.thisptr_shared, scene.thisptr_shared,
                                    self.angle_step, self.distance_step,
                                    self.sampling_rate, self.position_threshold,
                                    self.rotation_threshold,
                                    deref(holder.get()), buf, &fitness)
        return (converged, _as_matrix(buf), wrap_pointnormal_cloud(holder),
                fitness)


cdef class IncrementalRegistration:
    """Chain frame-to-frame ICP into an absolute pose
    (pcl::registration::IncrementalRegistration): each `register_Cloud`
    aligns the new frame to the previous one and composes the result.

        reg = pcl.IncrementalRegistration(max_correspondence_distance=0.1)
        for frame in frames:
            reg.register_Cloud(frame)
            pose = reg.get_AbsoluteTransform()
    """

    cdef IncrementalICP* me

    def __cinit__(self, double max_correspondence_distance=0.05,
                  int max_iterations=50, double transformation_epsilon=1e-8):
        if max_correspondence_distance <= 0 or max_iterations < 1:
            raise ValueError("distance must be positive, iterations >= 1")
        self.me = new IncrementalICP(max_correspondence_distance,
                                     max_iterations, transformation_epsilon)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def register_Cloud(self, PointCloud cloud not None, delta_estimate=None):
        """Align *cloud* to the previous frame (the first one only sets
        the origin). Returns whether ICP converged."""
        cdef float buf[16]
        cdef float* guess = NULL
        if delta_estimate is not None:
            _matrix16(delta_estimate, buf)
            guess = buf
        return self.me.registerCloud(cloud.thisptr_shared, guess)

    def get_DeltaTransform(self):
        cdef float buf[16]
        self.me.deltaTransform(buf)
        return _as_matrix(buf)

    def get_AbsoluteTransform(self):
        cdef float buf[16]
        self.me.absoluteTransform(buf)
        return _as_matrix(buf)

    def reset(self):
        self.me.reset()


cdef class MetaRegistration:
    """Register every new frame against the ACCUMULATED map of all
    previous ones (pcl::registration::MetaRegistration) -- slower than
    frame-to-frame, but it does not drift the same way.
    `get_MetaCloud()` returns the map."""

    cdef MetaICP* me

    def __cinit__(self, double max_correspondence_distance=0.05,
                  int max_iterations=50, double transformation_epsilon=1e-8):
        if max_correspondence_distance <= 0 or max_iterations < 1:
            raise ValueError("distance must be positive, iterations >= 1")
        self.me = new MetaICP(max_correspondence_distance, max_iterations,
                              transformation_epsilon)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def register_Cloud(self, PointCloud cloud not None, delta_estimate=None):
        cdef float buf[16]
        cdef float* guess = NULL
        if delta_estimate is not None:
            _matrix16(delta_estimate, buf)
            guess = buf
        return self.me.registerCloud(cloud.thisptr_shared, guess)

    def get_AbsoluteTransform(self):
        cdef float buf[16]
        self.me.absoluteTransform(buf)
        return _as_matrix(buf)

    def get_MetaCloud(self):
        cdef PointCloud out = PointCloud()
        self.me.metaCloud(deref(out.ptr()))
        return out

    def reset(self):
        self.me.reset()


cdef class LUM:
    """Lu and Milios' globally consistent scan matching
    (pcl::registration::LUM): a graph of clouds, each with a pose, and
    correspondences between overlapping ones; `compute()` moves all poses
    at once to minimise the correspondence error. The first cloud is the
    fixed reference.

        lum = pcl.LUM()
        a = lum.add_PointCloud(cloud_a)
        b = lum.add_PointCloud(cloud_b, pose=(x, y, z, roll, pitch, yaw))
        lum.set_Correspondences(b, a, pairs)     # (m, 2): index in b, in a
        lum.compute()
        transform_b = lum.get_Transformation(b)
    """

    cdef LUMGraph* me

    def __cinit__(self):
        self.me = new LUMGraph()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_PointCloud(self, PointCloud cloud not None, pose=None):
        """Add a cloud (in its own frame) with an initial pose ``(x, y, z,
        roll, pitch, yaw)``; returns its vertex index."""
        cdef float p[6]
        cdef float* pp = NULL
        if pose is not None:
            values = [float(v) for v in pose]
            if len(values) != 6:
                raise ValueError("pose is (x, y, z, roll, pitch, yaw)")
            for i in range(6):
                p[i] = values[i]
            pp = p
        return self.me.addCloud(cloud.thisptr_shared, pp)

    def get_NumVertices(self):
        return self.me.size()

    cdef int _vertex(self, int vertex) except -1:
        if not 0 <= vertex < self.me.size():
            raise IndexError("vertex %d out of range" % vertex)
        return vertex

    def set_Correspondences(self, int source, int target, pairs):
        """``(m, 2)`` index pairs: a point of cloud *source*, its match in
        cloud *target*."""
        self._vertex(source)
        self._vertex(target)
        if source == target:
            raise ValueError("source and target must differ")
        cdef vector[int] p
        cdef vector[float] d
        _pairs_in(pairs, None, p, d)
        if p.size() < 6:
            raise ValueError("at least three correspondences are needed")
        self.me.setCorrespondences(source, target, p)

    def set_MaxIterations(self, int count):
        if count < 1:
            raise ValueError("count must be >= 1")
        self.me.setMaxIterations(count)

    def set_ConvergenceThreshold(self, float threshold):
        self.me.setConvergenceThreshold(threshold)

    def compute(self):
        if self.me.size() < 2:
            raise RuntimeError("add at least two clouds first")
        with nogil:
            self.me.compute()

    def get_Pose(self, int vertex):
        import numpy as np
        cdef float p[6]
        self.me.pose(self._vertex(vertex), p)
        return np.array([p[i] for i in range(6)], dtype=np.float32)

    def get_Transformation(self, int vertex):
        cdef float buf[16]
        self.me.transformation(self._vertex(vertex), buf)
        return _as_matrix(buf)

    def get_TransformedCloud(self, int vertex):
        cdef PointCloud out = PointCloud()
        self.me.transformedCloud(self._vertex(vertex), deref(out.ptr()))
        return out

    def get_ConcatenatedCloud(self):
        cdef PointCloud out = PointCloud()
        self.me.concatenatedCloud(deref(out.ptr()))
        return out


cdef class ELCH:
    """Explicit Loop Closing Heuristic (pcl::registration::ELCH): a chain
    of clouds that comes back to where it started; the loop-closing error
    between `loop_start` and `loop_end` is spread back along the chain.
    The closure itself comes from ICP, or from `set_LoopTransform`.

        elch = pcl.ELCH()
        for cloud in chain:            # already roughly aligned
            elch.add_PointCloud(cloud)
        elch.set_LoopStart(0)
        elch.set_LoopEnd(len(chain) - 1)
        elch.compute()
        corrected = [elch.get_PointCloud(i) for i in range(len(chain))]
    """

    cdef ELCHLoop* me
    cdef int loop_start
    cdef int loop_end

    def __cinit__(self, double max_correspondence_distance=0.1,
                  int max_iterations=50, double transformation_epsilon=1e-8):
        self.me = new ELCHLoop(max_correspondence_distance, max_iterations,
                               transformation_epsilon)
        self.loop_start = -1
        self.loop_end = -1

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_PointCloud(self, PointCloud cloud not None):
        """Append a cloud to the chain (copied); returns its index."""
        return self.me.addCloud(cloud.thisptr_shared)

    cdef int _vertex(self, int vertex) except -1:
        if not 0 <= vertex < self.me.size():
            raise IndexError("vertex %d out of range" % vertex)
        return vertex

    def set_LoopStart(self, int vertex):
        self.me.setLoopStart(self._vertex(vertex))
        self.loop_start = vertex

    def set_LoopEnd(self, int vertex):
        self.me.setLoopEnd(self._vertex(vertex))
        self.loop_end = vertex

    def set_LoopTransform(self, matrix):
        """The transform that closes the loop, instead of running ICP
        between the end clouds."""
        cdef float buf[16]
        _matrix16(matrix, buf)
        self.me.setLoopTransform(buf)

    def compute(self):
        if self.me.size() < 3:
            raise RuntimeError("a loop needs at least three clouds")
        if self.loop_start < 0 or self.loop_end < 0:
            raise RuntimeError("set_LoopStart and set_LoopEnd first")
        if self.loop_start == self.loop_end:
            raise ValueError("the loop start and end must differ")
        with nogil:
            self.me.compute()

    def get_PointCloud(self, int vertex):
        """Cloud *vertex* as the loop correction left it."""
        cdef PointCloud out = PointCloud()
        self.me.cloud(self._vertex(vertex), deref(out.ptr()))
        return out
