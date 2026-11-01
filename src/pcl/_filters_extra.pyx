# distutils: language = c++
# cython: language_level=3
"""The rest of pcl/filters: farthest-point and covariance sampling, the
OpenMP fast bilateral filter, normal refinement, the label / covariance
/ occlusion voxel grids, plane and box clipping, 3-D and 2-D
convolution, and the morphological operator.

Those whose PCL API takes Eigen planes, boxes or kernels, a nested enum,
or a point type nothing else uses go through pcl/compat/filter_args.h.
"""

from cython.operator cimport dereference as deref

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport Normal, PointXYZ, PointXYZRGB
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.filters.covariance_sampling cimport (
    CovarianceSampling as cCovarianceSampling)
from pcl.pxd.filters.fast_bilateral_omp cimport (
    FastBilateralFilterOMP as cFastBilateralFilterOMP)
from pcl.pxd.compat.filter_args cimport (
    FarthestPointSampler, clipBox, clipPlane, convolve2D, gaussianConvolution3D,
    gaussianConvolution3DRGB, morphologicalOperator, occludedVoxels,
    occlusionStates, refineNormals, voxelGridCovariance, voxelGridLabel)
cimport pcl.pxd.compat.filter_args as fa

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointXYZRGB, wrap_normal_cloud,
    wrap_xyzrgb_cloud)


# filters::Convolution border handling and the morphological operators,
# re-exported from the headers through the shim.
BORDERS_POLICY_IGNORE = fa.BORDERS_POLICY_IGNORE
BORDERS_POLICY_MIRROR = fa.BORDERS_POLICY_MIRROR
BORDERS_POLICY_DUPLICATE = fa.BORDERS_POLICY_DUPLICATE
MORPH_OPEN = fa.MORPH_OPEN
MORPH_CLOSE = fa.MORPH_CLOSE
MORPH_DILATE = fa.MORPH_DILATE
MORPH_ERODE = fa.MORPH_ERODE


cdef object _int_array(vector[int]& values):
    import numpy as np
    result = np.empty(values.size(), dtype=np.int32)
    cdef int[::1] view = result
    cdef Py_ssize_t i
    for i in range(<Py_ssize_t> values.size()):
        view[i] = values[i]
    return result


cdef object _leaf(leaf_size):
    if isinstance(leaf_size, (int, float)):
        values = (float(leaf_size),) * 3
    else:
        values = tuple(float(v) for v in leaf_size)
    if len(values) != 3 or min(values) <= 0:
        raise ValueError("leaf size must be positive (one value or three)")
    return values


cdef void _triple(values, float* out) except *:
    a, b, c = values
    out[0] = a
    out[1] = b
    out[2] = c


# --- sampling ----------------------------------------------------------------

cdef class FarthestPointSampling:
    """Farthest point sampling (pcl::FarthestPointSampling): start from a
    random point and keep adding the point farthest from everything
    picked so far -- an even cover of the cloud in `sample` points.
    `filter()` returns the sampled :class:`PointCloud`,
    `filter_indices()` their indices. PCL 1.13+: on an older PCL both
    raise RuntimeError."""

    cdef FarthestPointSampler* me
    cdef Py_ssize_t n_input
    cdef bint has_sample

    def __cinit__(self, PointCloud pc=None, sample=None, seed=None):
        self.me = new FarthestPointSampler()
        self.n_input = -1
        self.has_sample = False
        if pc is not None:
            self.set_InputCloud(pc)
        if sample is not None:
            self.set_Sample(sample)
        if seed is not None:
            self.set_Seed(seed)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_Sample(self, Py_ssize_t count):
        if count < 1:
            raise ValueError("sample must be >= 1")
        self.me.setSample(count)
        self.has_sample = True

    def get_Sample(self):
        return self.me.getSample()

    def set_Seed(self, unsigned int seed):
        """Fix the random start (PCL seeds from std::random_device)."""
        self.me.setSeed(seed)

    def get_Seed(self):
        return self.me.getSeed()

    cdef _check(self):
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud(...) is required first")
        if not self.has_sample:
            raise RuntimeError("set_Sample(...) is required first")
        if self.me.getSample() > self.n_input:
            raise ValueError("sample (%d) exceeds the cloud (%d points)"
                             % (self.me.getSample(), self.n_input))

    def filter(self):
        self._check()
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.filter(deref(out))
        return pc

    def filter_indices(self):
        self._check()
        cdef vector[int] indices
        with nogil:
            self.me.filter(indices)
        return _int_array(indices)


cdef class CovarianceSampling:
    """Covariance sampling (pcl::CovarianceSampling, Gelfand et al.):
    keeps the `number_of_samples` points that best constrain a
    point-to-plane ICP, so registration on the sample stays as well
    conditioned as on the whole cloud. Needs normals.
    `compute_ConditionNumber()` reports how well conditioned the input
    is (lower is better)."""

    cdef cCovarianceSampling[PointXYZ, Normal]* me
    cdef Py_ssize_t n_input
    cdef bint has_normals
    cdef bint has_samples

    def __cinit__(self, PointCloud pc=None):
        self.me = new cCovarianceSampling[PointXYZ, Normal]()
        self.n_input = -1
        self.has_normals = False
        self.has_samples = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self.n_input = pc.size

    def set_Normals(self, PointCloud_Normal normals not None):
        if self.n_input >= 0 and normals.size != self.n_input:
            raise ValueError("normals must match the cloud")
        self.me.setNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_NumberOfSamples(self, unsigned int count):
        if count < 1:
            raise ValueError("count must be >= 1")
        self.me.setNumberOfSamples(count)
        self.has_samples = True

    def get_NumberOfSamples(self):
        return self.me.getNumberOfSamples()

    cdef _check(self):
        if self.n_input < 0:
            raise RuntimeError("set_InputCloud(...) is required first")
        if not self.has_normals:
            raise RuntimeError("set_Normals(...) is required first")

    def compute_ConditionNumber(self):
        self._check()
        return self.me.computeConditionNumber()

    def filter(self):
        self._check()
        if not self.has_samples:
            raise RuntimeError("set_NumberOfSamples(...) is required first")
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.filter(deref(out))
        return pc

    def filter_indices(self):
        self._check()
        if not self.has_samples:
            raise RuntimeError("set_NumberOfSamples(...) is required first")
        cdef vector[int] indices
        with nogil:
            self.me.filter(indices)
        return _int_array(indices)


cdef class FastBilateralFilterOMP:
    """`FastBilateralFilter` over several threads
    (pcl::FastBilateralFilterOMP); organized clouds only."""

    cdef cFastBilateralFilterOMP[PointXYZ]* me
    cdef bint has_input

    def __cinit__(self, PointCloud pc=None, unsigned int threads=0):
        self.me = new cFastBilateralFilterOMP[PointXYZ](threads)
        self.has_input = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        if pc.ptr().height < 2:
            raise ValueError("FastBilateralFilterOMP needs an organized cloud "
                             "(height > 1)")
        self.me.setInputCloud(pc.thisptr_shared)
        self.has_input = True

    def set_sigma_s(self, float sigma_s):
        """Spatial window, in pixels."""
        self.me.setSigmaS(sigma_s)

    def get_sigma_s(self):
        return self.me.getSigmaS()

    def set_sigma_r(self, float sigma_r):
        """Depth difference still averaged, in the cloud's units."""
        self.me.setSigmaR(sigma_r)

    def get_sigma_r(self):
        return self.me.getSigmaR()

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def filter(self):
        if not self.has_input:
            raise RuntimeError("set_InputCloud(...) is required first")
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.filter(deref(out))
        return pc


# --- normals -----------------------------------------------------------------

cdef class NormalRefinement:
    """Iteratively smooth normals towards their neighbours'
    (pcl::NormalRefinement): each normal becomes the distance-weighted
    mean of its `k` nearest neighbours' until it moves less than
    `convergence_threshold`, for at most `max_iterations` rounds.

        refined = pcl.NormalRefinement(k=10).filter(cloud, normals)
    """

    cdef int k
    cdef unsigned int max_iterations
    cdef float convergence

    def __cinit__(self, int k=10, unsigned int max_iterations=15,
                  float convergence_threshold=1e-5):
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.max_iterations = max_iterations
        self.convergence = convergence_threshold

    def set_MaxIterations(self, unsigned int count):
        self.max_iterations = count

    def get_MaxIterations(self):
        return self.max_iterations

    def set_ConvergenceThreshold(self, float threshold):
        self.convergence = threshold

    def get_ConvergenceThreshold(self):
        return self.convergence

    def filter(self, PointCloud cloud not None,
               PointCloud_Normal normals not None):
        """Return the refined :class:`PointCloud_Normal`."""
        if normals.size != cloud.size:
            raise ValueError("normals must match the cloud")
        if cloud.size <= self.k:
            raise ValueError("the cloud needs more than k points")
        cdef shared_ptr[cPointCloud[Normal]] holder
        holder.reset(new cPointCloud[Normal]())
        with nogil:
            refineNormals(cloud.thisptr_shared, normals.thisptr_shared,
                          self.k, self.max_iterations, self.convergence,
                          deref(holder.get()))
        return wrap_normal_cloud(holder)


# --- voxel grids ---------------------------------------------------------------

cdef class VoxelGridLabel:
    """A voxel grid that keeps a label per voxel (pcl::VoxelGridLabel):
    the centroid of each voxel together with the label most of its
    points carry.

        points, labels = pcl.VoxelGridLabel(0.05).filter(cloud, labels)
    """

    cdef float lx, ly, lz

    def __cinit__(self, leaf_size=0.01):
        self.lx, self.ly, self.lz = _leaf(leaf_size)

    def set_LeafSize(self, float lx, float ly, float lz):
        self.lx, self.ly, self.lz = _leaf((lx, ly, lz))

    def filter(self, PointCloud cloud not None, labels):
        """Return ``(points, labels)``: the voxel centroids as a
        :class:`PointCloud` and a uint32 label per centroid."""
        import numpy as np
        arr = np.ascontiguousarray(labels)
        if arr.ndim != 1 or arr.shape[0] != cloud.size:
            raise ValueError("labels must be an (n,) array, one per point")
        if arr.size and arr.min() < 0:
            raise ValueError("labels must be non-negative")
        cdef vector[unsigned int] in_labels = arr.astype(np.uint32).tolist()
        cdef vector[unsigned int] out_labels
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* src = cloud.ptr()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            voxelGridLabel(deref(src), in_labels, self.lx, self.ly, self.lz,
                           deref(out), out_labels)
        result = np.empty(out_labels.size(), dtype=np.uint32)
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> out_labels.size()):
            result[i] = out_labels[i]
        return pc, result


cdef class VoxelGridCovariance:
    """A voxel grid with per-voxel statistics (pcl::VoxelGridCovariance):
    the mean, 3x3 covariance and point count of every voxel with at least
    `min_points_per_voxel` points -- the cell model NDT registers against.

        means, covariances, counts = pcl.VoxelGridCovariance(0.5).compute(cloud)

    `cov_eig_value_inflation_ratio` floors each covariance's smaller
    eigenvalues at that fraction of the largest, so flat cells stay
    invertible.
    """

    cdef float lx, ly, lz
    cdef int min_points
    cdef double inflation

    def __cinit__(self, leaf_size=0.5, int min_points_per_voxel=6,
                  double cov_eig_value_inflation_ratio=0.01):
        self.lx, self.ly, self.lz = _leaf(leaf_size)
        self.set_MinPointPerVoxel(min_points_per_voxel)
        self.inflation = cov_eig_value_inflation_ratio

    def set_LeafSize(self, float lx, float ly, float lz):
        self.lx, self.ly, self.lz = _leaf((lx, ly, lz))

    def set_MinPointPerVoxel(self, int count):
        """Points a voxel needs for a covariance (PCL's floor is 3)."""
        if count < 3:
            raise ValueError("a covariance needs at least 3 points per voxel")
        self.min_points = count

    def get_MinPointPerVoxel(self):
        return self.min_points

    def set_CovEigValueInflationRatio(self, double ratio):
        self.inflation = ratio

    def compute(self, PointCloud cloud not None):
        """Return ``(means (k, 3), covariances (k, 3, 3), counts (k,))``."""
        import numpy as np
        cdef vector[float] means, covariances
        cdef vector[int] counts
        with nogil:
            voxelGridCovariance(cloud.thisptr_shared, self.lx, self.ly,
                                self.lz, self.min_points, self.inflation,
                                means, covariances, counts)
        k = counts.size()
        m = np.array(means, dtype=np.float32).reshape(k, 3)
        c = np.array(covariances, dtype=np.float32).reshape(k, 3, 3)
        return m, c, _int_array(counts)


cdef class VoxelGridOcclusionEstimation:
    """Which voxels are hidden behind others, seen from a sensor
    (pcl::VoxelGridOcclusionEstimation): a ray is cast from `origin` to
    every voxel of the cloud's bounding grid, and a voxel is occluded
    when an occupied one lies on the way.

        occl = pcl.VoxelGridOcclusionEstimation(0.1, origin=(0, 0, 0))
        ijk, centers = occl.occluded_voxels(cloud)
        states = occl.occlusion_states(cloud, queries)
    """

    cdef float lx, ly, lz
    cdef float origin[3]

    def __cinit__(self, leaf_size=0.1, origin=(0.0, 0.0, 0.0)):
        self.lx, self.ly, self.lz = _leaf(leaf_size)
        _triple(origin, self.origin)

    def set_LeafSize(self, float lx, float ly, float lz):
        self.lx, self.ly, self.lz = _leaf((lx, ly, lz))

    def set_SensorOrigin(self, origin):
        _triple(origin, self.origin)

    def occluded_voxels(self, PointCloud cloud not None):
        """Return ``(ijk (k, 3) int32, centers (k, 3) float32)`` of every
        occluded voxel."""
        import numpy as np
        if cloud.size == 0:
            raise ValueError("the cloud is empty")
        cdef vector[int] ijk
        cdef vector[float] centers
        with nogil:
            occludedVoxels(cloud.thisptr_shared, self.origin, self.lx,
                           self.ly, self.lz, ijk, centers)
        k = ijk.size() // 3
        return (np.array(ijk, dtype=np.int32).reshape(k, 3),
                np.array(centers, dtype=np.float32).reshape(k, 3))

    def occlusion_states(self, PointCloud cloud not None,
                         PointCloud queries not None):
        """State of the voxel holding each query point: 0 free, 1
        occluded, -1 outside the grid."""
        if cloud.size == 0:
            raise ValueError("the cloud is empty")
        cdef vector[int] states
        cdef cPointCloud[PointXYZ]* q = queries.ptr()
        with nogil:
            occlusionStates(cloud.thisptr_shared, self.origin, self.lx,
                            self.ly, self.lz, deref(q), states)
        return _int_array(states)


# --- clipping --------------------------------------------------------------------

cdef class PlaneClipper3D:
    """Keep the points on the positive side of a plane
    (pcl::PlaneClipper3D): ``a*x + b*y + c*z + d >= 0``.

        indices = pcl.PlaneClipper3D(0, 0, 1, -0.5).clip(cloud)  # z >= 0.5
    """

    cdef float a, b, c, d

    def __cinit__(self, float a, float b, float c, float d):
        if a == 0 and b == 0 and c == 0:
            raise ValueError("(a, b, c) must not be the zero vector")
        self.a, self.b, self.c, self.d = a, b, c, d

    def set_PlaneParameters(self, float a, float b, float c, float d):
        if a == 0 and b == 0 and c == 0:
            raise ValueError("(a, b, c) must not be the zero vector")
        self.a, self.b, self.c, self.d = a, b, c, d

    def clip(self, PointCloud cloud not None):
        """Return the int32 indices of the points kept."""
        cdef vector[int] kept
        cdef cPointCloud[PointXYZ]* src = cloud.ptr()
        with nogil:
            clipPlane(deref(src), self.a, self.b, self.c, self.d, kept)
        return _int_array(kept)


cdef class BoxClipper3D:
    """Keep the points inside an oriented box (pcl::BoxClipper3D): a box
    of `size` (full edge lengths), rotated by the Rodrigues vector
    `rodrigues` (axis times angle in radians) and centred at `center`.

        indices = pcl.BoxClipper3D(center=(0, 0, 0), size=(1, 1, 1)).clip(cloud)

    Unlike `CropBox` the box can have any orientation; this is the
    clipper `ExtractPolygonalPrismData` and the octree visualizers build
    on.
    """

    cdef float center[3]
    cdef float size[3]
    cdef float rodrigues[3]

    def __cinit__(self, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0),
                  rodrigues=(0.0, 0.0, 0.0)):
        self.set_Box(center, size, rodrigues)

    def set_Box(self, center, size, rodrigues=(0.0, 0.0, 0.0)):
        _triple(center, self.center)
        _triple(size, self.size)
        _triple(rodrigues, self.rodrigues)
        if min(self.size[0], self.size[1], self.size[2]) <= 0:
            raise ValueError("box size must be positive on every axis")

    def clip(self, PointCloud cloud not None):
        """Return the int32 indices of the points inside the box."""
        cdef vector[int] kept
        cdef cPointCloud[PointXYZ]* src = cloud.ptr()
        with nogil:
            clipBox(deref(src), self.rodrigues, self.center, self.size, kept)
        return _int_array(kept)


# --- convolution ---------------------------------------------------------------------

cdef class Convolution3D:
    """Gaussian smoothing over 3-D neighbourhoods
    (pcl::filters::Convolution3D with GaussianKernel): each point becomes
    the Gaussian-weighted mean of its neighbours within `radius`. On a
    :class:`PointCloud_PointXYZRGB` the colour is smoothed too
    (GaussianKernelRGB). Neighbours beyond
    `threshold_relative_to_sigma` sigmas are ignored.

        smoothed = pcl.Convolution3D(sigma=0.02, radius=0.05).convolve(cloud)
    """

    cdef float sigma
    cdef float threshold
    cdef double radius
    cdef unsigned int threads

    def __cinit__(self, float sigma=0.01, double radius=0.03,
                  float threshold_relative_to_sigma=3.0,
                  unsigned int threads=0):
        if sigma <= 0 or radius <= 0 or threshold_relative_to_sigma <= 0:
            raise ValueError("sigma, radius and threshold must be positive")
        self.sigma = sigma
        self.radius = radius
        self.threshold = threshold_relative_to_sigma
        self.threads = threads

    def set_Sigma(self, float sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma

    def set_RadiusSearch(self, double radius):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = radius

    def set_ThresholdRelativeToSigma(self, float coefficient):
        if coefficient <= 0:
            raise ValueError("coefficient must be positive")
        self.threshold = coefficient

    def set_NumberOfThreads(self, unsigned int threads):
        self.threads = threads

    def convolve(self, cloud not None):
        """Return the smoothed cloud, of the same type as *cloud*."""
        cdef PointCloud out
        cdef PointCloud src
        cdef PointCloud_PointXYZRGB rgb
        cdef shared_ptr[cPointCloud[PointXYZRGB]] rgb_out
        if isinstance(cloud, PointCloud_PointXYZRGB):
            rgb = cloud
            rgb_out.reset(new cPointCloud[PointXYZRGB]())
            with nogil:
                gaussianConvolution3DRGB(rgb.thisptr_shared, self.sigma,
                                         self.threshold, self.radius,
                                         self.threads, deref(rgb_out.get()))
            return wrap_xyzrgb_cloud(rgb_out)
        if not isinstance(cloud, PointCloud):
            raise TypeError("expected a PointCloud or PointCloud_PointXYZRGB, "
                            "got %s" % type(cloud).__name__)
        src = cloud
        out = PointCloud()
        cdef cPointCloud[PointXYZ]* dst = out.ptr()
        with nogil:
            gaussianConvolution3D(src.thisptr_shared, self.sigma,
                                  self.threshold, self.radius, self.threads,
                                  deref(dst))
        return out


cdef class Convolution:
    """Separable 2-D convolution of an organized cloud's x, y, z
    (pcl::filters::Convolution): rows with `h_kernel`, then columns with
    `v_kernel`; either may be omitted. `borders` is one of
    ``pcl.BORDERS_POLICY_IGNORE`` (borders become NaN), ``MIRROR``
    (default) or ``DUPLICATE``.

        kernel = [0.25, 0.5, 0.25]
        smoothed = pcl.Convolution(kernel, kernel).convolve(cloud)
    """

    cdef vector[float] h_kernel
    cdef vector[float] v_kernel
    cdef int borders

    def __cinit__(self, h_kernel=None, v_kernel=None,
                  int borders=fa.BORDERS_POLICY_MIRROR):
        self.set_Kernels(h_kernel, v_kernel)
        self.set_BordersPolicy(borders)

    def set_Kernels(self, h_kernel=None, v_kernel=None):
        for kernel in (h_kernel, v_kernel):
            if kernel is not None:
                if len(kernel) == 0 or len(kernel) % 2 == 0:
                    raise ValueError("kernels must have an odd length")
        self.h_kernel = [] if h_kernel is None else [float(v) for v in h_kernel]
        self.v_kernel = [] if v_kernel is None else [float(v) for v in v_kernel]

    def set_BordersPolicy(self, int policy):
        if policy not in (fa.BORDERS_POLICY_IGNORE, fa.BORDERS_POLICY_MIRROR,
                          fa.BORDERS_POLICY_DUPLICATE):
            raise ValueError("unknown borders policy %r" % policy)
        self.borders = policy

    def convolve(self, PointCloud cloud not None):
        """Return the convolved organized :class:`PointCloud`."""
        if cloud.ptr().height < 2:
            raise ValueError("Convolution needs an organized cloud "
                             "(height > 1)")
        if self.h_kernel.empty() and self.v_kernel.empty():
            raise RuntimeError("set at least one kernel first")
        cdef Py_ssize_t limit = min(cloud.ptr().width, cloud.ptr().height)
        if <Py_ssize_t> max(self.h_kernel.size(), self.v_kernel.size()) > limit:
            raise ValueError("a kernel is longer than the image is wide/high")
        cdef PointCloud out = PointCloud()
        cdef cPointCloud[PointXYZ]* dst = out.ptr()
        with nogil:
            convolve2D(cloud.thisptr_shared, self.h_kernel, self.v_kernel,
                       self.borders, deref(dst))
        return out


# --- morphology ---------------------------------------------------------------------

def apply_morphological_operator(PointCloud cloud not None, float resolution,
                                 int operator):
    """Grey-scale morphology on z over a grid of `resolution`
    (pcl::applyMorphologicalOperator): ``pcl.MORPH_DILATE`` raises every
    point to the highest z in its cell neighbourhood, ``MORPH_ERODE``
    lowers it to the lowest, ``MORPH_OPEN`` / ``MORPH_CLOSE`` combine
    the two. The building block of the progressive morphological ground
    filter. Returns a new :class:`PointCloud` of the same points."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if operator not in (fa.MORPH_OPEN, fa.MORPH_CLOSE, fa.MORPH_DILATE,
                        fa.MORPH_ERODE):
        raise ValueError("operator must be one of pcl.MORPH_*")
    if cloud.size == 0:
        raise ValueError("the cloud is empty")
    cdef PointCloud out = PointCloud()
    cdef cPointCloud[PointXYZ]* dst = out.ptr()
    with nogil:
        morphologicalOperator(cloud.thisptr_shared, resolution, operator,
                              deref(dst))
    return out
