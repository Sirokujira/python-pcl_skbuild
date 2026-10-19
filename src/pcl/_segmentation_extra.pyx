# distutils: language = c++
# cython: language_level=3
"""The rest of pcl/segmentation: organized connected components and the
comparators that drive it, labelled euclidean clusters, GrabCut, the
FPFH unary classifier and the random walker.

Organized segmentation is a comparator plus a segmenter:

    normals = pcl.IntegralImageNormalEstimation(cloud).compute_cloud()
    comparator = pcl.PlaneCoefficientComparator(normals)
    seg = pcl.OrganizedConnectedComponentSegmentation(comparator)
    seg.set_InputCloud(cloud)
    labels, components = seg.segment()   # (h, w) int64, index lists

Every comparator here is a settings holder: PCL's comparators take
shared_ptr-to-const clouds, per-point plane equations and raw pointers,
so the whole comparator is built and run in compat/segmentation_args.h
when `segment()` is called.
"""

import math

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport Normal, PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.point_indices cimport PointIndices
from pcl.pxd.compat.segmentation_args cimport (
    INVALID_LABEL, comparatorExcludesLabels, grabCut, labeledEuclideanClusters,
    labeledRegionBoundary, organizedConnectedComponents, randomWalkerSegment,
    unarySegment, unaryTrain, unaryTrainWithLabels)
cimport pcl.pxd.compat.segmentation_args as sa

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_Normal, PointCloud_PointXYZRGBA


cdef list _index_lists(vector[PointIndices]& clusters):
    """Plain loops on purpose: a comprehension over a C++ reference
    builds a closure that segfaults (see .claude/rules/pipeline.md)."""
    cdef list result = []
    cdef list one
    cdef size_t i, j
    for i in range(clusters.size()):
        one = []
        for j in range(clusters[i].indices.size()):
            one.append(clusters[i].indices[j])
        result.append(one)
    return result


cdef vector[unsigned int] _labels_in(labels, Py_ssize_t n) except *:
    """One label per point; negative values (no label) become PCL's
    invalid label."""
    import numpy as np
    arr = np.asarray(labels).ravel()
    if arr.shape[0] != n:
        raise ValueError("expected %d labels, got %d" % (n, arr.shape[0]))
    if arr.dtype.kind not in "iu":
        raise ValueError("labels must be integers")
    arr = np.where(arr < 0, INVALID_LABEL, arr).astype(np.uint32)
    return arr.tolist()


cdef object _label_image(vector[unsigned int]& labels, unsigned int width,
                         unsigned int height):
    """(height, width) int64, -1 where PCL assigned no label."""
    import numpy as np
    out = np.empty(labels.size(), dtype=np.int64)
    cdef long long[::1] view = out
    cdef size_t i
    for i in range(labels.size()):
        view[i] = -1 if labels[i] == INVALID_LABEL else labels[i]
    return out.reshape(height, width)


# --- comparators --------------------------------------------------------------

cdef class _Comparator:
    """What `OrganizedConnectedComponentSegmentation` needs to build one
    PCL comparator: its kind, its parameters, and the per-point inputs."""

    cdef int kind
    cdef vector[float] params
    cdef PointCloud_Normal normals
    cdef vector[unsigned int] labels
    cdef vector[unsigned int] exclude
    cdef vector[float] distance_map
    cdef bint needs_colour

    def __cinit__(self, *args, **kwargs):
        self.needs_colour = False


cdef class EuclideanClusterComparator(_Comparator):
    """Neighbouring pixels join when closer than `distance_threshold`
    (pcl::EuclideanClusterComparator); with `depth_dependent` the
    threshold scales by z^2.

    `set_Labels(labels)` with `set_ExcludeLabels(values)` keeps pixels
    whose label is excluded from joining anything -- the classic use is
    clustering the objects left after `segment_and_refine()` removed the
    planes. PCL 1.11+ reads its set the other way round (only the listed
    labels may join); the wrapper detects which way the installed PCL
    reads it and passes the set that excludes.
    """

    def __init__(self, float distance_threshold=0.01,
                 bint depth_dependent=False):
        self.kind = sa.COMPARATOR_EUCLIDEAN_CLUSTER
        self.set_DistanceThreshold(distance_threshold, depth_dependent)

    def set_DistanceThreshold(self, float distance, bint depth_dependent=False):
        if distance <= 0:
            raise ValueError("distance_threshold must be positive")
        self.params = [distance, 1.0 if depth_dependent else 0.0]

    def get_DistanceThreshold(self):
        return self.params[0]

    def get_DepthDependent(self):
        return self.params[1] != 0.0

    def set_Labels(self, labels):
        """One integer label per pixel (negative: none), e.g. the
        ``labels`` of `OrganizedMultiPlaneSegmentation.segment_and_refine()`."""
        import numpy as np
        arr = np.asarray(labels).ravel()
        self.labels = _labels_in(arr, arr.shape[0])

    def set_ExcludeLabels(self, labels):
        self.exclude = sorted(set(int(v) for v in labels))

    def get_ExcludeLabels(self):
        return list(self.exclude)


def comparator_reads_exclude_set_as_exclusion():
    """How the installed PCL's EuclideanClusterComparator reads its
    exclude set: True for an exclusion, False when it keeps ONLY the
    listed labels (PCL 1.11 through at least 1.14). The wrapper adapts
    either way; this is for tests and the curious."""
    return comparatorExcludesLabels()


cdef class _PlaneComparator(_Comparator):

    cdef _set_plane(self, PointCloud_Normal normals, float angular_threshold,
                    float distance_threshold, bint depth_dependent):
        if normals is None:
            raise ValueError("normals are required")
        if not 0 < angular_threshold < math.pi:
            raise ValueError("angular_threshold is in radians, in (0, pi)")
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        self.normals = normals
        self.params = [angular_threshold, distance_threshold,
                       1.0 if depth_dependent else 0.0]

    def get_AngularThreshold(self):
        return self.params[0]

    def get_DistanceThreshold(self):
        return self.params[1]


cdef class PlaneCoefficientComparator(_PlaneComparator):
    """Neighbouring pixels join when their normals agree within
    `angular_threshold` (radians) and their tangent planes' offsets
    (``p . n``) within `distance_threshold`, scaled by z^2 with
    `depth_dependent` (pcl::PlaneCoefficientComparator). Defaults are
    PCL's -- except that PCL's constructor stores the 2-degree default
    as an angle where `compare()` expects its cosine (accepting normals
    up to ~88 degrees apart); here the angle always goes through the
    setter that converts it."""

    def __init__(self, PointCloud_Normal normals not None,
                 float angular_threshold=math.radians(2.0),
                 float distance_threshold=0.02, bint depth_dependent=True):
        self.kind = sa.COMPARATOR_PLANE_COEFFICIENT
        self._set_plane(normals, angular_threshold, distance_threshold,
                        depth_dependent)


cdef class EuclideanPlaneCoefficientComparator(_PlaneComparator):
    """Normals within `angular_threshold` (radians) and the points
    themselves closer than `distance_threshold`
    (pcl::EuclideanPlaneCoefficientComparator); PCL ignores
    `depth_dependent` for this one."""

    def __init__(self, PointCloud_Normal normals not None,
                 float angular_threshold=math.radians(2.0),
                 float distance_threshold=0.02, bint depth_dependent=True):
        self.kind = sa.COMPARATOR_EUCLIDEAN_PLANE_COEFFICIENT
        self._set_plane(normals, angular_threshold, distance_threshold,
                        depth_dependent)


cdef class EdgeAwarePlaneComparator(_PlaneComparator):
    """`PlaneCoefficientComparator` that also refuses to grow across
    depth edges (pcl::EdgeAwarePlaneComparator): a pixel closer than
    `distance_map_threshold` pixels to a depth change, or with curvature
    above `curvature_threshold`, joins nothing, and neighbours must be
    within `euclidean_distance_threshold` of each other.

    `distance_map` is ``(height, width)``: the distance in pixels to the
    nearest depth change, as `IntegralImageNormalEstimation.get_DistanceMap()`
    returns. PCL reads it unconditionally, so it is required."""

    def __init__(self, PointCloud_Normal normals not None, distance_map,
                 float angular_threshold=math.radians(2.0),
                 float distance_threshold=0.02, bint depth_dependent=True,
                 float curvature_threshold=0.04,
                 float distance_map_threshold=5.0,
                 float euclidean_distance_threshold=0.04):
        import numpy as np
        self.kind = sa.COMPARATOR_EDGE_AWARE_PLANE
        self._set_plane(normals, angular_threshold, distance_threshold,
                        depth_dependent)
        self.params.push_back(curvature_threshold)
        self.params.push_back(distance_map_threshold)
        self.params.push_back(euclidean_distance_threshold)
        dm = np.ascontiguousarray(distance_map, dtype=np.float32).ravel()
        if dm.shape[0] != normals.size:
            raise ValueError("distance_map needs one value per pixel (%d), "
                             "got %d" % (normals.size, dm.shape[0]))
        self.distance_map = dm.tolist()


cdef class GroundPlaneComparator(_PlaneComparator):
    """Pixels whose normal is within `ground_angular_threshold` of
    `expected_ground_normal` (default -y, a camera's "up" when y points
    down) and agrees with its neighbour's within `angular_threshold`
    (pcl::GroundPlaneComparator) -- a drivable-surface detector. PCL's
    compare() ignores the distance threshold."""

    def __init__(self, PointCloud_Normal normals not None,
                 float angular_threshold=math.radians(2.0),
                 float ground_angular_threshold=math.radians(10.0),
                 expected_ground_normal=(0.0, -1.0, 0.0)):
        self.kind = sa.COMPARATOR_GROUND_PLANE
        self._set_plane(normals, angular_threshold, 0.1, True)
        if not 0 < ground_angular_threshold < math.pi:
            raise ValueError("ground_angular_threshold is in radians, in "
                             "(0, pi)")
        gx, gy, gz = (float(v) for v in expected_ground_normal)
        length = math.sqrt(gx * gx + gy * gy + gz * gz)
        if length == 0:
            raise ValueError("expected_ground_normal must be non-zero")
        self.params.push_back(ground_angular_threshold)
        self.params.push_back(gx / length)
        self.params.push_back(gy / length)
        self.params.push_back(gz / length)


cdef class RGBPlaneCoefficientComparator(_PlaneComparator):
    """Normals within `angular_threshold`, the points themselves closer
    than `distance_threshold` (a euclidean distance here, as in
    `EuclideanPlaneCoefficientComparator`, not a plane offset) and their
    colours within `color_threshold` (euclidean in 0..255 RGB)
    (pcl::RGBPlaneCoefficientComparator). Needs a
    :class:`PointCloud_PointXYZRGBA` input."""

    def __init__(self, PointCloud_Normal normals not None,
                 float color_threshold=50.0,
                 float angular_threshold=math.radians(2.0),
                 float distance_threshold=0.02, bint depth_dependent=True):
        self.kind = sa.COMPARATOR_RGB_PLANE_COEFFICIENT
        self._set_plane(normals, angular_threshold, distance_threshold,
                        depth_dependent)
        if color_threshold <= 0:
            raise ValueError("color_threshold must be positive")
        self.params.push_back(color_threshold)
        self.needs_colour = True


# --- organized connected components ------------------------------------------

cdef class OrganizedConnectedComponentSegmentation:
    """Connected components of an ORGANIZED cloud under a comparator
    (pcl::OrganizedConnectedComponentSegmentation): two neighbouring
    pixels share a component when the comparator says they match.

    `segment()` returns ``(labels, components)``: a ``(height, width)``
    int64 label image (-1 for invalid pixels) and one index list per
    component, singletons included. Takes a :class:`PointCloud` or a
    :class:`PointCloud_PointXYZRGBA` (the colour comparator needs the
    latter).
    """

    cdef _Comparator comparator
    cdef object cloud
    cdef unsigned int width
    cdef unsigned int height

    def __cinit__(self, _Comparator comparator=None):
        self.comparator = comparator

    def set_Comparator(self, _Comparator comparator not None):
        self.comparator = comparator

    def get_Comparator(self):
        return self.comparator

    def set_InputCloud(self, cloud):
        if not isinstance(cloud, (PointCloud, PointCloud_PointXYZRGBA)):
            raise TypeError("expected a PointCloud or PointCloud_PointXYZRGBA")
        if cloud.height <= 1:
            raise ValueError(
                "OrganizedConnectedComponentSegmentation needs an organized "
                "cloud (height > 1); this one is %d x %d"
                % (cloud.width, cloud.height))
        self.cloud = cloud
        self.width = cloud.width
        self.height = cloud.height

    def segment(self):
        if self.cloud is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        if self.comparator is None:
            raise RuntimeError("no comparator: call set_Comparator() first")
        cdef _Comparator c = self.comparator
        cdef Py_ssize_t n = self.width * self.height
        if c.normals is not None and c.normals.size != n:
            raise ValueError("the comparator's normals (%d) do not match the "
                             "cloud (%d points)" % (c.normals.size, n))
        if c.labels.size() and <Py_ssize_t> c.labels.size() != n:
            raise ValueError("the comparator's labels (%d) do not match the "
                             "cloud (%d points)" % (c.labels.size(), n))
        if c.distance_map.size() and <Py_ssize_t> c.distance_map.size() != n:
            raise ValueError("the comparator's distance_map does not match "
                             "the cloud")
        if c.needs_colour and not isinstance(self.cloud,
                                             PointCloud_PointXYZRGBA):
            raise ValueError("%s needs a PointCloud_PointXYZRGBA input"
                             % type(c).__name__)
        cdef shared_ptr[cPointCloud[Normal]] normals
        if c.normals is not None:
            normals = c.normals.thisptr_shared
        cdef vector[unsigned int] labels
        cdef vector[PointIndices] components
        cdef PointCloud xyz
        cdef PointCloud_PointXYZRGBA rgba
        if isinstance(self.cloud, PointCloud):
            xyz = self.cloud
            with nogil:
                organizedConnectedComponents(
                    xyz.thisptr_shared, c.kind, c.params, normals, c.labels,
                    c.exclude, c.distance_map, labels, components)
        else:
            rgba = self.cloud
            with nogil:
                organizedConnectedComponents(
                    rgba.thisptr_shared, c.kind, c.params, normals, c.labels,
                    c.exclude, c.distance_map, labels, components)
        return (_label_image(labels, self.width, self.height),
                _index_lists(components))

    @staticmethod
    def find_LabeledRegionBoundary(labels, int start_index):
        """The outer contour of the region containing `start_index` in a
        ``(height, width)`` label image, as pixel indices in tracing
        order. `start_index` must lie ON the region's edge (the last
        index of a component in raster order always does); an interior
        pixel gives an empty list."""
        import numpy as np
        arr = np.asarray(labels)
        if arr.ndim != 2:
            raise ValueError("labels must be a (height, width) array")
        cdef unsigned int h = arr.shape[0]
        cdef unsigned int w = arr.shape[1]
        if not 0 <= start_index < h * w:
            raise IndexError("start_index out of range")
        cdef vector[unsigned int] flat = _labels_in(arr, h * w)
        cdef vector[int] boundary
        with nogil:
            labeledRegionBoundary(start_index, flat, w, h, boundary)
        return [boundary[i] for i in range(boundary.size())]


# --- labelled euclidean clusters ----------------------------------------------

cdef class LabeledEuclideanClusterExtraction:
    """Euclidean clusters that never cross a label boundary
    (pcl::LabeledEuclideanClusterExtraction): points join only
    neighbours within `set_ClusterTolerance` that carry the same label.
    `Extract()` returns ``{label: [cluster, ...]}``, largest cluster
    first within each label.

    PCL writes each cluster to ``output[label]`` without sizing the
    output (a crash for any non-empty cloud, in plain C++ too); the
    wrapper sizes it first.
    """

    cdef PointCloud _input
    cdef vector[unsigned int] labels
    cdef bint has_labels
    cdef double tolerance
    cdef int min_size
    cdef int max_size

    def __cinit__(self, PointCloud pc=None, labels=None):
        self.tolerance = 0.0
        self.min_size = 1
        self.max_size = 2147483647
        self.has_labels = False
        if pc is not None:
            self.set_InputCloud(pc)
        if labels is not None:
            self.set_Labels(labels)

    def set_InputCloud(self, PointCloud pc not None):
        self._input = pc
        if self.has_labels and <Py_ssize_t> self.labels.size() != pc.size:
            self.has_labels = False

    def set_Labels(self, labels):
        """One non-negative integer label per point."""
        import numpy as np
        arr = np.asarray(labels).ravel()
        if arr.dtype.kind not in "iu" or (arr.size and arr.min() < 0):
            raise ValueError("labels must be non-negative integers")
        if arr.size and arr.max() > 10000000:
            raise ValueError("labels above 10^7 would need an output slot "
                             "per value; renumber them first")
        self.labels = arr.astype(np.uint32).tolist()
        self.has_labels = True

    def set_ClusterTolerance(self, double tolerance):
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = tolerance

    def get_ClusterTolerance(self):
        return self.tolerance

    def set_MinClusterSize(self, int size):
        self.min_size = size

    def set_MaxClusterSize(self, int size):
        self.max_size = size

    def Extract(self):
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        if not self.has_labels or <Py_ssize_t> self.labels.size() != self._input.size:
            raise RuntimeError("set_Labels() with one label per point is "
                               "required")
        if self.tolerance <= 0:
            raise RuntimeError("set_ClusterTolerance() is required")
        cdef vector[unsigned int] out_labels
        cdef vector[PointIndices] clusters
        cdef cPointCloud[PointXYZ]* cloud = self._input.ptr()
        with nogil:
            labeledEuclideanClusters(cloud[0], self.labels, self.tolerance,
                                     self.min_size, self.max_size, out_labels,
                                     clusters)
        cdef list lists = _index_lists(clusters)
        cdef dict result = {}
        cdef size_t i
        for i in range(out_labels.size()):
            result.setdefault(out_labels[i], []).append(lists[i])
        return result


# --- GrabCut ------------------------------------------------------------------

cdef class GrabCut:
    """GrabCut (pcl::GrabCut) on a :class:`PointCloud_PointXYZRGBA`:
    colour GMMs for object and background plus a graph cut over the
    neighbourhood (the image grid for an organized cloud, `k`-nearest
    neighbours otherwise), iterated to convergence.

    `set_ForegroundRegion(indices)` gives the pixels that MAY hold the
    object -- GrabCut's user rectangle; everything else is fixed as
    background. PCL's own name for that call is
    setBackgroundPointsIndices, and its comment says the indices are the
    background, but the implementation marks them as possible foreground
    (`apps/src/grabcut_2d.cpp` passes the rectangle's interior). `extract()`
    returns the foreground indices.
    """

    cdef PointCloud_PointXYZRGBA _input
    cdef vector[int] region
    cdef unsigned int k
    cdef float lambda_
    cdef int neighbours

    def __cinit__(self, unsigned int k=5, float lambda_=50.0):
        self.k = k
        self.lambda_ = lambda_
        self.neighbours = 9

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self._input = pc

    def set_K(self, unsigned int k):
        """Gaussian components per colour model (PCL default 5)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def get_K(self):
        return self.k

    def set_Lambda(self, float value):
        """Weight of the smoothness term (PCL default 50)."""
        self.lambda_ = value

    def get_Lambda(self):
        return self.lambda_

    def set_NumberOfNeighbours(self, int count):
        """Neighbours per point on an unorganized cloud (PCL default 9)."""
        if count < 1:
            raise ValueError("count must be >= 1")
        self.neighbours = count

    def get_NumberOfNeighbours(self):
        return self.neighbours

    def set_ForegroundRegion(self, indices):
        import numpy as np
        arr = np.asarray(indices, dtype=np.int64).ravel()
        self.region = arr.astype(np.int32).tolist()

    def extract(self):
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        cdef Py_ssize_t n = self._input.size
        if self.region.size() == 0:
            raise RuntimeError("set_ForegroundRegion() is required")
        cdef size_t i
        for i in range(self.region.size()):
            if not 0 <= self.region[i] < n:
                raise IndexError("region index %d out of range"
                                 % self.region[i])
        cdef vector[int] foreground
        with nogil:
            grabCut(self._input.thisptr_shared, self.region, self.k,
                    self.lambda_, self.neighbours, foreground)
        return [foreground[i] for i in range(foreground.size())]


# --- unary classifier ---------------------------------------------------------

def _repair_centres(centres):
    """Replace the NaN centres PCL's k-means leaves for empty clusters
    with a copy of a finite one of the same set."""
    import numpy as np
    bad = ~np.all(np.isfinite(centres), axis=1)
    if np.all(bad):
        raise RuntimeError("k-means produced no finite centre; check the "
                           "radii (FPFH needs neighbours within them)")
    centres[bad] = centres[np.flatnonzero(~bad)[0]]
    return centres


cdef class UnaryClassifier:
    """FPFH bag-of-words classifier (pcl::UnaryClassifier): `train()`
    reduces a cloud's FPFH descriptors to `set_ClusterSize` k-means
    centres; `train_with_labels()` does it per label;
    `set_TrainedFeatures()` + `segment()` then assigns each point of a
    new cloud to the trained set whose nearest centre (chi-square) is
    within `set_FeatureThreshold`, -1 when none is.

    Every trained set must hold the same number of centres: PCL strides
    through the concatenated sets by the first one's size. PCL's k-means
    divides by a cluster's size without checking it, so a cluster left
    empty becomes a NaN centre, and a NaN centre poisons the chi-square
    nearest-neighbour search; the training calls replace any such row
    with a copy of another centre of the same set (which changes no
    nearest-centre answer), and `set_TrainedFeatures` refuses NaN.
    `train_with_labels` prints "cluster numbers: ..." to stdout -- PCL
    writes it with std::cout.
    """

    cdef unsigned int k
    cdef float normal_radius
    cdef float fpfh_radius
    cdef float threshold
    cdef vector[float] trained
    cdef int sets

    def __cinit__(self):
        self.k = 0
        self.normal_radius = 0.01
        self.fpfh_radius = 0.05
        self.threshold = 5.0
        self.sets = 0

    def set_ClusterSize(self, unsigned int k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def get_ClusterSize(self):
        return self.k

    def set_NormalRadiusSearch(self, float radius):
        self.normal_radius = radius

    def set_FPFHRadiusSearch(self, float radius):
        self.fpfh_radius = radius

    def set_FeatureThreshold(self, float threshold):
        self.threshold = threshold

    cdef _check_k(self, PointCloud pc):
        if self.k < 1:
            raise RuntimeError("set_ClusterSize() is required")
        if pc.size < self.k:
            raise ValueError("%d points cannot give %d centres"
                             % (pc.size, self.k))

    def train(self, PointCloud pc not None):
        """``(k, 33)`` float32 FPFH centres of the whole cloud."""
        import numpy as np
        self._check_k(pc)
        cdef vector[float] centres
        with nogil:
            unaryTrain(pc.thisptr_shared, self.k, self.normal_radius,
                       self.fpfh_radius, centres)
        return _repair_centres(
            np.asarray(centres, dtype=np.float32).reshape(-1, 33))

    def train_with_labels(self, PointCloud pc not None, labels):
        """``{label: (k, 33) float32}``, one set of centres per label."""
        import numpy as np
        self._check_k(pc)
        arr = np.asarray(labels).ravel()
        if arr.shape[0] != pc.size or arr.dtype.kind not in "iu" or \
                (arr.size and arr.min() < 0):
            raise ValueError("expected one non-negative integer label per "
                             "point")
        values, counts = np.unique(arr, return_counts=True)
        if np.any(counts < self.k):
            raise ValueError("every label needs at least k = %d points"
                             % self.k)
        cdef vector[unsigned int] vlabels = arr.astype(np.uint32).tolist()
        cdef vector[unsigned int] out_labels
        cdef vector[int] out_counts
        cdef vector[float] centres
        cdef cPointCloud[PointXYZ]* cloud = pc.ptr()
        with nogil:
            unaryTrainWithLabels(cloud[0], vlabels, self.k,
                                 self.normal_radius, self.fpfh_radius,
                                 out_labels, out_counts, centres)
        flat = np.asarray(centres, dtype=np.float32).reshape(-1, 33)
        cdef dict result = {}
        cdef Py_ssize_t start = 0
        cdef size_t i
        for i in range(out_labels.size()):
            result[out_labels[i]] = _repair_centres(
                flat[start:start + out_counts[i]].copy())
            start += out_counts[i]
        return result

    def set_TrainedFeatures(self, features):
        """A list of ``(k, 33)`` centre arrays, all with the same k;
        `segment()` reports a point's position in this list."""
        import numpy as np
        arrays = [np.ascontiguousarray(f, dtype=np.float32) for f in features]
        if not arrays:
            raise ValueError("at least one trained set is required")
        shape = arrays[0].shape
        if len(shape) != 2 or shape[1] != 33 or shape[0] == 0:
            raise ValueError("each trained set must be a (k, 33) array")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("trained centres must be finite")
        if any(a.shape != shape for a in arrays):
            raise ValueError("every trained set must hold the same number "
                             "of centres (PCL strides by the first one)")
        self.trained = np.concatenate(arrays).ravel().tolist()
        self.sets = len(arrays)

    def segment(self, PointCloud pc not None):
        """``(n,)`` int32: for each point, the index of the trained set it
        matches, or -1."""
        import numpy as np
        if self.sets == 0:
            raise RuntimeError("set_TrainedFeatures() is required")
        cdef vector[int] out
        with nogil:
            unarySegment(pc.thisptr_shared, self.trained, self.sets,
                         self.threshold, self.normal_radius,
                         self.fpfh_radius, out)
        return np.asarray(out, dtype=np.int32)


# --- random walker --------------------------------------------------------------

def random_walker(edges, weights, seeds):
    """Random walker segmentation (pcl::segmentation::randomWalker,
    Grady 2006) on an undirected weighted graph -- build it from a
    point cloud's neighbourhoods (e.g. k nearest neighbours weighted by
    ``exp(-d^2 / sigma^2)``) or from anything else.

    `edges` is ``(m, 2)`` vertex pairs, `weights` ``(m,)`` positive
    values, `seeds` one integer per vertex: 0 for unseeded, otherwise
    the seed's colour. Returns ``(colours, potentials, columns)``:
    every vertex's colour as ``(n,)`` uint32 (0 only for an unseeded
    vertex with no edges), ``(n, k)`` float32 probabilities that a walk
    from each vertex first reaches a seed of colour ``columns[j]``.
    """
    import numpy as np
    s = np.asarray(seeds).ravel()
    if s.dtype.kind not in "iu" or (s.size and s.min() < 0):
        raise ValueError("seeds must be non-negative integers, 0 = unseeded")
    e = np.asarray(edges)
    if e.size == 0:
        e = e.reshape(0, 2)
    if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in "iu":
        raise ValueError("edges must be an (m, 2) integer array")
    w = np.asarray(weights, dtype=np.float32).ravel()
    if w.shape[0] != e.shape[0]:
        raise ValueError("expected one weight per edge")
    if w.size and not np.all(w > 0):
        raise ValueError("weights must be positive")
    if e.size and (e.min() < 0 or e.max() >= s.shape[0]):
        raise ValueError("edge endpoints must be vertices 0..n-1")
    if not np.any(s > 0):
        raise ValueError("at least one vertex must be seeded")
    cdef vector[int] vedges = e.astype(np.int32).ravel().tolist()
    cdef vector[float] vweights = w.tolist()
    cdef vector[unsigned int] colours = s.astype(np.uint32).tolist()
    cdef vector[unsigned int] columns
    cdef vector[float] potentials
    cdef bint solved
    with nogil:
        solved = randomWalkerSegment(vedges, vweights, colours, columns,
                                     potentials)
    if not solved:
        raise RuntimeError("the random walker's linear solve failed")
    k = <Py_ssize_t> columns.size()
    return (np.asarray(colours, dtype=np.uint32),
            np.asarray(potentials, dtype=np.float32).reshape(-1, k)
            if k else np.zeros((len(s), 0), np.float32),
            [columns[i] for i in range(columns.size())])
