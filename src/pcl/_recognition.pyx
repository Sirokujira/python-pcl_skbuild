# distutils: language = c++
# cython: language_level=3
"""Correspondence grouping: find instances of a model inside a scene.

This is the second half of PCL's 3-D object recognition pipeline. The
first half — describe keypoints and match the descriptors — produces a
list of (model index, scene index) correspondences; grouping turns those
into poses, discarding the matches that no rigid transform can explain:

    grouping = pcl.GeometricConsistencyGrouping(model, scene)
    grouping.set_GCSize(0.01)
    grouping.set_GCThreshold(5)
    for transform, correspondences in grouping.recognize(pairs):
        ...

python-pcl declared `pcl_Recognition_*.pxd` but never wrapped any of it
(its `pxi/Recognition/AddList.txt` is empty), so the names here follow
PCL's own rather than an existing Python API.

Everything goes through pcl/compat/recognition_args.h — see that header
for why correspondences and Eigen-allocator result vectors cannot be
stated in a pxd.

The last stage, hypothesis verification, decides which of the
recognized instances to believe: `GreedyVerification` and `PapazovHV`
each take the scene plus every candidate model already moved into it
(`model.transform(pose)`), and return one keep/reject flag per
candidate:

    hv = pcl.PapazovHV()
    hv.set_SceneCloud(scene)
    hv.add_Models([model.transform(t) for t, _ in instances])
    keep = hv.verify()

They go through pcl/compat/hypotheses_args.h, which also sets the
"complete" model list PCL leaves to the caller (without it, PapazovHV
keeps nothing). GlobalHypothesesVerification is not wrapped: PCL's
keeps every hypothesis it is given (see the README).
"""

from libcpp.vector cimport vector

from pcl.pxd.compat.recognition_args cimport (
    geometricConsistencyGrouping, hough3DGrouping)
from pcl.pxd.compat.matching_args cimport (
    matchFpfhDescriptors, matchShotDescriptors)
from pcl.pxd.compat.hypotheses_args cimport (
    greedyVerification, papazovVerification)
from libcpp.memory cimport shared_ptr
from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud

from pcl._pointcloud cimport PointCloud


cdef object _instances(vector[float]& transforms, vector[int]& counts,
                       vector[int]& pairs):
    """Split the shim's flat output into (transform, correspondences).

    Plain loops on purpose — see the Cython traps section of
    .claude/rules/pipeline.md: a comprehension over a C++ reference makes
    Cython build a closure holding it, and that segfaults.
    """
    import numpy as np

    cdef Py_ssize_t n = <Py_ssize_t> counts.size()
    cdef Py_ssize_t i, j, k, offset = 0
    cdef Py_ssize_t count

    result = []
    for i in range(n):
        matrix = np.empty(16, dtype=np.float32)
        for j in range(16):
            matrix[j] = transforms[i * 16 + j]

        count = <Py_ssize_t> counts[i]
        correspondences = []
        for k in range(count):
            correspondences.append(
                (pairs[(offset + k) * 2], pairs[(offset + k) * 2 + 1]))
        offset += count

        # Column-major, which is how Eigen stored it.
        result.append((matrix.reshape(4, 4, order="F"), correspondences))
    return result


cdef _split_pairs(correspondences, vector[int]& model_indices,
                  vector[int]& scene_indices, vector[float]& distances):
    """Take (model, scene) or (model, scene, distance) tuples apart."""
    for entry in correspondences:
        if len(entry) == 2:
            model_index, scene_index = entry
            distance = 0.0
        elif len(entry) == 3:
            model_index, scene_index, distance = entry
        else:
            raise ValueError(
                "each correspondence must be (model_index, scene_index) or "
                "(model_index, scene_index, distance), got %r" % (entry,))
        model_indices.push_back(<int> model_index)
        scene_indices.push_back(<int> scene_index)
        distances.push_back(<float> distance)


cdef class _CorrespondenceGrouping:
    """What both grouping algorithms share: the two clouds."""

    cdef PointCloud model
    cdef PointCloud scene

    def set_InputCloud(self, PointCloud model not None):
        """The model to look for."""
        self.model = model

    def set_SceneCloud(self, PointCloud scene not None):
        """The scene to look in."""
        self.scene = scene

    cdef _check_ready(self):
        if self.model is None:
            raise RuntimeError("set_InputCloud() is required before recognize()")
        if self.scene is None:
            raise RuntimeError("set_SceneCloud() is required before recognize()")


cdef class GeometricConsistencyGrouping(_CorrespondenceGrouping):
    """Group correspondences by geometric consistency
    (pcl::GeometricConsistencyGrouping).

    Two correspondences are consistent when the distance between the two
    model points matches the distance between the two scene points to
    within `GCSize`; a cluster of at least `GCThreshold` mutually
    consistent correspondences becomes one recognized instance.
    """

    cdef double gc_size
    cdef int gc_threshold

    def __cinit__(self, PointCloud model=None, PointCloud scene=None):
        self.gc_size = 1.0
        self.gc_threshold = 3
        if model is not None:
            self.set_InputCloud(model)
        if scene is not None:
            self.set_SceneCloud(scene)

    def set_GCSize(self, double gc_size):
        """Consensus set resolution, in the cloud's own units."""
        if gc_size <= 0:
            raise ValueError("gc_size must be > 0, got %r" % gc_size)
        self.gc_size = gc_size

    def get_GCSize(self):
        return self.gc_size

    def set_GCThreshold(self, int threshold):
        """Smallest cluster that counts as an instance. PCL's floor is 3:
        a 6-DOF pose needs three correspondences."""
        if threshold < 3:
            raise ValueError(
                "gc_threshold must be >= 3 (a 6-DOF pose needs three "
                "correspondences), got %r" % threshold)
        self.gc_threshold = threshold

    def get_GCThreshold(self):
        return self.gc_threshold

    def recognize(self, correspondences):
        """Return a list of ``(transform, correspondences)``.

        *correspondences* is an iterable of ``(model_index, scene_index)``
        or ``(model_index, scene_index, distance)``. Each transform is a
        ``(4, 4)`` float32 matrix taking the model into the scene.
        """
        self._check_ready()

        cdef vector[int] model_indices
        cdef vector[int] scene_indices
        cdef vector[float] distances
        _split_pairs(correspondences, model_indices, scene_indices, distances)

        cdef vector[float] transforms
        cdef vector[int] counts
        cdef vector[int] pairs
        cdef double gc_size = self.gc_size
        cdef int gc_threshold = self.gc_threshold
        with nogil:
            geometricConsistencyGrouping(
                self.model.thisptr_shared, self.scene.thisptr_shared,
                model_indices, scene_indices, distances,
                gc_size, gc_threshold, transforms, counts, pairs)
        return _instances(transforms, counts, pairs)


cdef class Hough3DGrouping(_CorrespondenceGrouping):
    """Group correspondences by Hough voting (pcl::Hough3DGrouping).

    Each correspondence votes for the object centroid in a 3-D Hough
    space; peaks become instances. Votes are cast in local reference
    frames, which the shim estimates with BOARD — so unlike geometric
    consistency this one needs a support radius.
    """

    cdef float rf_radius
    cdef double bin_size
    cdef double threshold

    def __cinit__(self, PointCloud model=None, PointCloud scene=None):
        self.rf_radius = 0.015
        self.bin_size = 0.01
        self.threshold = 5.0
        if model is not None:
            self.set_InputCloud(model)
        if scene is not None:
            self.set_SceneCloud(scene)

    def set_RFRadius(self, float radius):
        """Support radius of the local reference frames."""
        if radius <= 0:
            raise ValueError("rf_radius must be > 0, got %r" % radius)
        self.rf_radius = radius

    def get_RFRadius(self):
        return self.rf_radius

    def set_HoughBinSize(self, double bin_size):
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0, got %r" % bin_size)
        self.bin_size = bin_size

    def get_HoughBinSize(self):
        return self.bin_size

    def set_HoughThreshold(self, double threshold):
        """Votes a bin needs before it counts as an instance."""
        self.threshold = threshold

    def get_HoughThreshold(self):
        return self.threshold

    def recognize(self, correspondences):
        """Same contract as GeometricConsistencyGrouping.recognize()."""
        self._check_ready()

        cdef vector[int] model_indices
        cdef vector[int] scene_indices
        cdef vector[float] distances
        _split_pairs(correspondences, model_indices, scene_indices, distances)

        cdef vector[float] transforms
        cdef vector[int] counts
        cdef vector[int] pairs
        cdef float rf_radius = self.rf_radius
        cdef double bin_size = self.bin_size
        cdef double threshold = self.threshold
        with nogil:
            hough3DGrouping(
                self.model.thisptr_shared, self.scene.thisptr_shared,
                model_indices, scene_indices, distances,
                rf_radius, bin_size, threshold, transforms, counts, pairs)
        return _instances(transforms, counts, pairs)


def match_descriptors(model_descriptors, scene_descriptors,
                      double max_distance=-1.0):
    """Match every scene descriptor to its nearest model descriptor.

    The step between FPFH/SHOT estimation and correspondence grouping:

        pairs = pcl.match_descriptors(model_desc, scene_desc)
        instances = grouping.recognize(pairs)

    *model_descriptors* and *scene_descriptors* are the ``(n, 33)`` /
    ``(n, 352)`` float32 arrays the estimators return; both must have the
    same width. Matching runs on a FLANN kd-tree (pcl::KdTreeFLANN), so
    it scales where a numpy broadcast cannot.

    *max_distance* drops matches farther than that in descriptor space
    (PCL's grouping tutorial uses 0.25 for unit-length SHOT); negative
    means unbounded. Rows containing NaN — SHOT's "no descriptor here" —
    are skipped on both sides.

    Returns a list of ``(model_index, scene_index, distance)``, the form
    ``recognize()`` accepts directly.
    """
    import numpy as np

    model = np.ascontiguousarray(model_descriptors, dtype=np.float32)
    scene = np.ascontiguousarray(scene_descriptors, dtype=np.float32)
    if model.ndim != 2 or scene.ndim != 2:
        raise ValueError(
            "descriptors must be 2-D (n, dim) arrays, got shapes %r and %r"
            % (model.shape, scene.shape))
    if model.shape[1] != scene.shape[1]:
        raise ValueError(
            "model and scene descriptors differ in width: %d vs %d"
            % (model.shape[1], scene.shape[1]))

    cdef int dim = <int> model.shape[1]
    if dim not in (33, 352):
        raise ValueError(
            "unsupported descriptor width %d (wrapped: 33 for FPFH, "
            "352 for SHOT)" % dim)

    cdef vector[float] model_flat
    cdef vector[float] scene_flat
    cdef float[::1] model_view = model.reshape(-1)
    cdef float[::1] scene_view = scene.reshape(-1)
    cdef Py_ssize_t i
    model_flat.resize(model_view.shape[0])
    for i in range(model_view.shape[0]):
        model_flat[i] = model_view[i]
    scene_flat.resize(scene_view.shape[0])
    for i in range(scene_view.shape[0]):
        scene_flat[i] = scene_view[i]

    cdef vector[int] model_indices
    cdef vector[int] scene_indices
    cdef vector[float] distances
    cdef float bound = <float> max_distance
    if dim == 33:
        with nogil:
            matchFpfhDescriptors(model_flat, scene_flat, bound,
                                 model_indices, scene_indices, distances)
    else:
        with nogil:
            matchShotDescriptors(model_flat, scene_flat, bound,
                                 model_indices, scene_indices, distances)

    result = []
    for i in range(<Py_ssize_t> scene_indices.size()):
        result.append((model_indices[i], scene_indices[i],
                       float(distances[i])))
    return result


cdef class _HypothesisVerification:
    """What every verifier shares (pcl::HypothesisVerification): the
    scene, the candidate models, and three distances in the cloud's own
    units — `resolution` (both clouds are voxel-downsampled to it first),
    `inlier_threshold` (how close a model point must be to the scene to
    be explained by it) and `occlusion_threshold`. PCL's defaults are
    0.005 for all three: millimetre-scale data in metres."""

    cdef PointCloud scene
    cdef list models
    cdef float resolution
    cdef float inlier_threshold
    cdef float occlusion_threshold
    cdef object mask

    def __cinit__(self, *args, **kwargs):
        self.scene = None
        self.models = []
        self.resolution = 0.005
        self.inlier_threshold = 0.005
        self.occlusion_threshold = 0.005
        self.mask = None

    def set_SceneCloud(self, PointCloud scene not None):
        if scene.size == 0:
            raise ValueError("the scene cloud is empty")
        self.scene = scene

    def add_Models(self, models):
        """Replace the candidates with *models*: PointClouds already
        placed in the scene's frame."""
        checked = []
        for model in models:
            if not isinstance(model, PointCloud):
                raise TypeError("models must be PointClouds, got %s"
                                % type(model).__name__)
            if (<PointCloud> model).size == 0:
                raise ValueError("model %d is empty" % len(checked))
            checked.append(model)
        self.models = checked
        self.mask = None

    def set_Resolution(self, float resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = resolution

    def get_Resolution(self):
        return self.resolution

    def set_InlierThreshold(self, float threshold):
        if threshold <= 0:
            raise ValueError("inlier threshold must be positive")
        self.inlier_threshold = threshold

    def get_InlierThreshold(self):
        return self.inlier_threshold

    def set_OcclusionThreshold(self, float threshold):
        if threshold <= 0:
            raise ValueError("occlusion threshold must be positive")
        self.occlusion_threshold = threshold

    def get_OcclusionThreshold(self):
        return self.occlusion_threshold

    cdef void _run(self, shared_ptr[cPointCloud[PointXYZ]] scene,
                   vector[shared_ptr[cPointCloud[PointXYZ]]]& models,
                   vector[int]& mask) except *:
        raise NotImplementedError

    def verify(self):
        """Return one bool per model, in `add_Models` order: True for a
        hypothesis the verifier keeps."""
        if self.scene is None:
            raise RuntimeError("set_SceneCloud(...) is required first")
        if not self.models:
            raise RuntimeError("add_Models(...) is required first")
        cdef vector[shared_ptr[cPointCloud[PointXYZ]]] models
        for model in self.models:
            models.push_back((<PointCloud> model).thisptr_shared)
        cdef vector[int] mask
        self._run(self.scene.thisptr_shared, models, mask)
        result = []
        cdef size_t i
        for i in range(mask.size()):
            result.append(bool(mask[i]))
        self.mask = result
        return list(result)

    def get_Mask(self):
        """The result of the last `verify()`, or None before one."""
        return None if self.mask is None else list(self.mask)


cdef class GreedyVerification(_HypothesisVerification):
    """Accept hypotheses one at a time, best first, as long as each
    explains more scene points than `lambda_` times the points it
    leaves unexplained (pcl::GreedyVerification). Fast; overlapping
    hypotheses are handled by the order."""

    cdef float lambda_

    def __cinit__(self, float lambda_=1.5):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.lambda_ = lambda_

    def get_Lambda(self):
        return self.lambda_

    cdef void _run(self, shared_ptr[cPointCloud[PointXYZ]] scene,
                   vector[shared_ptr[cPointCloud[PointXYZ]]]& models,
                   vector[int]& mask) except *:
        with nogil:
            greedyVerification(scene, models, self.resolution,
                               self.inlier_threshold, self.occlusion_threshold,
                               self.lambda_, mask)


cdef class PapazovHV(_HypothesisVerification):
    """Papazov and Burschka's conflict-graph verification
    (pcl::PapazovHV): a hypothesis needs `support_threshold` of its
    points explained and at most `penalty_threshold` of them in free
    space; of two hypotheses sharing more than `conflict_threshold` of
    their points, the better supported one wins."""

    cdef float conflict_threshold
    cdef float support_threshold
    cdef float penalty_threshold

    def __cinit__(self):
        # PCL's own defaults (hv_papazov.h).
        self.conflict_threshold = 0.02
        self.support_threshold = 0.1
        self.penalty_threshold = 0.1

    def set_ConflictThreshold(self, float fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("conflict threshold must be in [0, 1]")
        self.conflict_threshold = fraction

    def set_SupportThreshold(self, float fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("support threshold must be in [0, 1]")
        self.support_threshold = fraction

    def set_PenaltyThreshold(self, float fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("penalty threshold must be in [0, 1]")
        self.penalty_threshold = fraction

    cdef void _run(self, shared_ptr[cPointCloud[PointXYZ]] scene,
                   vector[shared_ptr[cPointCloud[PointXYZ]]]& models,
                   vector[int]& mask) except *:
        with nogil:
            papazovVerification(
                scene, models, self.resolution, self.inlier_threshold,
                self.occlusion_threshold, self.conflict_threshold,
                self.support_threshold, self.penalty_threshold, mask)
