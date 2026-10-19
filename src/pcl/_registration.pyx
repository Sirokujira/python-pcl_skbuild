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
    target points: `method` ``"svd"``
    (pcl::registration::TransformationEstimationSVD) or ``"lm"``
    (TransformationEstimationLM). With no `pairs`, point i matches
    point i and the clouds must be the same size. Returns a 4x4 array.
    """
    methods = {"svd": 0, "lm": 1}
    if method not in methods:
        raise ValueError("method must be 'svd' or 'lm', got %r" % method)
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
    cdef float buf[16]
    estimateTransformation(methods[method], source.thisptr_shared,
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
