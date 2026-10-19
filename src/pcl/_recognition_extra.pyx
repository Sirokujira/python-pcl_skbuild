# distutils: language = c++
# cython: language_level=3
"""The rest of pcl/recognition: CRH alignment, trimmed ICP, occlusion
filtering against an organized scene, ObjRecRANSAC, the implicit shape
model, and template matching (LINEMOD through LineRGBD, and DOTMOD).

    rec = pcl.ObjRecRANSAC(pair_width=0.1, voxel_size=0.01)
    rec.add_Model(model, model_normals, "mug")
    for name, pose, confidence in rec.recognize(scene, scene_normals):
        ...

Everything goes through pcl/compat/recognition_extra.h, which also lists
the upstream faults each wrapper works around.
"""

from cython.operator cimport dereference as deref
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.compat.recognition_extra cimport (
    DotMatcher, ObjectRecognizer, ShapeModel, TemplateMatcher, crhAlign,
    occlusionFilter, trimmedICP)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.point_types cimport Normal, PointXYZ

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_Normal, PointCloud_PointXYZRGBA


cdef object _matrices(vector[float]& flat):
    """Column-major 4x4 blocks -> a list of float32 matrices (plain loops:
    no comprehension over a C++ reference, see pipeline.md)."""
    import numpy as np
    cdef Py_ssize_t n = <Py_ssize_t> flat.size() // 16
    cdef Py_ssize_t i, j
    out = []
    for i in range(n):
        m = np.empty(16, dtype=np.float32)
        for j in range(16):
            m[j] = flat[i * 16 + j]
        out.append(m.reshape(4, 4, order="F"))
    return out


cdef _check_pair(PointCloud cloud, PointCloud_Normal normals, str what):
    import numpy as np
    if cloud.size == 0:
        raise ValueError("%s cloud is empty" % what)
    if normals.size != cloud.size:
        raise ValueError("%s: %d points but %d normals"
                         % (what, cloud.size, normals.size))
    if not np.isfinite(cloud.to_array()).all():
        raise ValueError("%s cloud has non-finite points; "
                         "remove_nan() first" % what)


cdef vector[unsigned char] _mask_bytes(mask, unsigned int width,
                                       unsigned int height) except *:
    import numpy as np
    arr = np.asarray(mask)
    if arr.shape != (height, width):
        raise ValueError("mask must have the frame's shape (%d, %d), got %r"
                         % (height, width, arr.shape))
    flat = np.ascontiguousarray(arr != 0, dtype=np.uint8).ravel()
    cdef vector[unsigned char] out
    cdef const unsigned char[::1] view = flat
    out.assign(&view[0], &view[0] + view.shape[0])
    return out


cdef _check_frame(PointCloud_PointXYZRGBA frame):
    if frame.height <= 1:
        raise ValueError("expected an organized PointCloud_PointXYZRGBA")


# --- CRH alignment -------------------------------------------------------------


cdef class CRHAlignment:
    """Roll candidates between two views of an object from their camera
    roll histograms (pcl::CRHAlignment).

    CVFH matches a view up to a roll about the line from the camera to
    its centroid; the CRH of each view (`CRHEstimation.compute()`) pins
    that roll down. `align` returns up to five candidate 4 x 4
    transforms taking the input view onto the target, best first, at the
    histogram's 2-degree resolution.
    """

    cdef object transforms

    def __cinit__(self):
        self.transforms = []

    def align(self, input_crh, target_crh, input_centroid, target_centroid):
        import numpy as np
        a = np.asarray(input_crh, dtype=np.float32).ravel()
        b = np.asarray(target_crh, dtype=np.float32).ravel()
        if a.shape != (90,) or b.shape != (90,):
            raise ValueError("CRH histograms have 90 values")
        ca = np.ascontiguousarray(input_centroid, dtype=np.float32).ravel()
        cb = np.ascontiguousarray(target_centroid, dtype=np.float32).ravel()
        if ca.shape != (3,) or cb.shape != (3,):
            raise ValueError("centroids are (x, y, z)")
        cdef vector[float] va = a.tolist()
        cdef vector[float] vb = b.tolist()
        cdef float[::1] pa = ca
        cdef float[::1] pb = cb
        cdef vector[float] out
        crhAlign(va, vb, &pa[0], &pb[0], out)
        self.transforms = _matrices(out)
        return list(self.transforms)

    def get_Transforms(self):
        return list(self.transforms)


# --- trimmed ICP ----------------------------------------------------------------


cdef class TrimmedICP:
    """ICP that pairs only the source points closest to the target
    (pcl::recognition::TrimmedICP), so a partial overlap is not pulled
    off by the rest.

        icp = pcl.TrimmedICP()
        icp.init(target)
        pose = icp.align(source, num_source_points_to_use=300, guess=pose)

    It iterates while a round lowers the summed squared distance below
    `new_to_old_energy_ratio` times the previous one (0.99; PCL clamps
    1 or more to that).
    """

    cdef PointCloud target
    cdef float ratio

    def __cinit__(self):
        self.ratio = 0.99

    def init(self, PointCloud target not None):
        if target.size == 0:
            raise ValueError("target cloud is empty")
        self.target = target

    def set_NewToOldEnergyRatio(self, float ratio):
        if not 0.0 < ratio:
            raise ValueError("ratio must be > 0")
        self.ratio = ratio if ratio < 1.0 else 0.99

    def align(self, PointCloud source not None, int num_source_points_to_use,
              guess=None):
        """Return the refined source -> target 4 x 4. Using every point
        is plain ICP; PCL prints a warning for it, so 1 <= n < size."""
        import numpy as np
        if self.target is None:
            raise RuntimeError("init(target) is required first")
        if not 1 <= num_source_points_to_use < source.size:
            raise ValueError("num_source_points_to_use must be in 1..%d"
                             % (source.size - 1))
        m = np.eye(4, dtype=np.float32) if guess is None else \
            np.asarray(guess, dtype=np.float32)
        if m.shape != (4, 4):
            raise ValueError("guess must be 4 x 4")
        flat = np.ascontiguousarray(m.ravel(order="F"))
        cdef float[::1] view = flat
        with nogil:
            trimmedICP(source.thisptr_shared, self.target.thisptr_shared,
                       num_source_points_to_use, self.ratio, &view[0])
        return flat.reshape(4, 4, order="F")


# --- occlusion reasoning ----------------------------------------------------------


cdef PointCloud _occlusion(PointCloud scene, PointCloud cloud, float f,
                           float threshold, bint check, bint occluded):
    if scene.height <= 1:
        raise ValueError("the scene must be an organized cloud")
    if not f > 0:
        raise ValueError("focal_length must be > 0")
    cdef PointCloud out = PointCloud()
    cdef cPointCloud[PointXYZ]* result = out.ptr()
    with nogil:
        occlusionFilter(scene.thisptr_shared, cloud.thisptr_shared, f,
                        threshold, check, occluded, deref(result))
    return out


def occlusion_filter(PointCloud scene not None, PointCloud cloud not None,
                     float focal_length, float threshold=0.01,
                     bint check_invalid_depth=True):
    """The points of `cloud` that an organized `scene` does not hide
    (pcl::occlusion_reasoning::filter).

    Each point is projected with `focal_length` about the image centre
    and kept when it is no more than `threshold` behind the scene's depth
    at that pixel. Points outside the image are dropped, as are points
    over a scene pixel with no depth unless `check_invalid_depth` is
    False. PCL's ZBuffering, which builds the depth image from an
    unorganized scene, is not wrapped (see the README)."""
    return _occlusion(scene, cloud, focal_length, threshold,
                      check_invalid_depth, False)


def get_occluded_cloud(PointCloud scene not None, PointCloud cloud not None,
                       float focal_length, float threshold=0.01,
                       bint check_invalid_depth=True):
    """The points of `cloud` more than `threshold` behind an organized
    `scene` (pcl::occlusion_reasoning::getOccludedCloud); the
    complement of `occlusion_filter` among the points that project
    inside the image."""
    return _occlusion(scene, cloud, focal_length, threshold,
                      check_invalid_depth, True)


# --- ObjRecRANSAC -------------------------------------------------------------------


cdef class ObjRecRANSAC:
    """Recognise known objects in a scene from oriented point pairs
    (pcl::recognition::ObjRecRANSAC, Papazov & Burschka).

    Every pair of model points `pair_width` apart goes into a hash table;
    recognition samples such pairs in the scene, votes for the poses
    they imply, and accepts the poses whose transformed model matches
    the scene. `pair_width` should be a sizeable fraction of the
    objects (a tenth to a half of their extent), `voxel_size` a few times
    the point spacing. Scenes are single views seen along +z: hypotheses
    are tested against the scene projected onto the xy-plane.

    Recognition is randomised; PCL's build prints progress to stdout on
    every call (OBJ_REC_RANSAC_VERBOSE is defined in its header).
    """

    cdef ObjectRecognizer* me
    cdef object names

    def __cinit__(self, float pair_width, float voxel_size):
        if not pair_width > 0 or not voxel_size > 0:
            raise ValueError("pair_width and voxel_size must be > 0")
        self.me = new ObjectRecognizer(pair_width, voxel_size)
        self.names = set()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_Model(self, PointCloud points not None,
                  PointCloud_Normal normals not None, str name):
        _check_pair(points, normals, "model")
        if name in self.names:
            raise ValueError("a model named %r was already added" % name)
        cdef string cname = name.encode()
        cdef cbool ok
        cdef cPointCloud[PointXYZ]* p = points.ptr()
        cdef cPointCloud[Normal]* n = normals.ptr()
        with nogil:
            ok = self.me.addModel(deref(p), deref(n), cname)
        if not ok:
            raise RuntimeError("PCL refused model %r" % name)
        self.names.add(name)

    def set_MaxCoplanarityAngleDegrees(self, float degrees):
        self.me.setMaxCoplanarityAngleDegrees(degrees)

    def set_SceneBoundsEnlargementFactor(self, float factor):
        self.me.setSceneBoundsEnlargementFactor(factor)

    def set_IgnoreCoplanarPointPairs(self, bint on):
        self.me.ignoreCoplanarPointPairs(on)

    def set_IcpHypothesesRefinement(self, bint on):
        self.me.icpHypothesesRefinement(on)

    def recognize(self, PointCloud scene not None,
                  PointCloud_Normal normals not None,
                  double success_probability=0.99):
        """A list of ``(name, pose, confidence)``: the model -> scene 4 x 4
        and the fraction of the model's voxels the scene explains."""
        if not self.names:
            raise RuntimeError("add_Model() is required first")
        _check_pair(scene, normals, "scene")
        if not 0.0 < success_probability < 1.0:
            raise ValueError("success_probability must be in (0, 1)")
        cdef vector[string] names
        cdef vector[float] transforms
        cdef vector[float] confidences
        cdef cPointCloud[PointXYZ]* p = scene.ptr()
        cdef cPointCloud[Normal]* n = normals.ptr()
        with nogil:
            self.me.recognize(deref(p), deref(n), success_probability, names,
                              transforms, confidences)
        poses = _matrices(transforms)
        out = []
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> names.size()):
            out.append((names[i].decode(), poses[i], float(confidences[i])))
        return out


# --- implicit shape model ------------------------------------------------------------


cdef class ImplicitShapeModelEstimation:
    """Class-level object detection by voting
    (pcl::ism::ImplicitShapeModelEstimation, Knopp et al.).

        ism = pcl.ImplicitShapeModelEstimation(feature_radius=0.05)
        ism.add_TrainingCloud(cup, cup_normals, 0)
        ism.add_TrainingCloud(box, box_normals, 1)
        ism.train()
        peaks = ism.find_Objects(scene, scene_normals, 0)  # x, y, z, density

    Training clusters FPFH descriptors of the downsampled clouds into
    visual words, each remembering where the object's centre was;
    detection lets every scene word vote for centres and returns the
    strongest peaks of the votes. The words are k-means clusters from
    random starts, so results vary a little between trainings; with few
    training objects, use more clusters than PCL's default 184 (with
    one object per class, 184 sometimes left a class no word of its own,
    and so no votes). The feature radius is not stored in a saved model
    -- use the same one when loading.

    PCL needs several things set right that it does not check: the
    feature estimator, finite and not axis-aligned normals, and a sensible
    sigma. compat/recognition_extra.h handles each.
    """

    cdef ShapeModel* me
    cdef bint trained

    def __cinit__(self, double feature_radius=0.05, float sampling_size=0.02,
                  unsigned int number_of_clusters=184, bint n_vot=True):
        if not feature_radius > 0 or not sampling_size > 0:
            raise ValueError("feature_radius and sampling_size must be > 0")
        if number_of_clusters < 1:
            raise ValueError("number_of_clusters must be >= 1")
        self.me = new ShapeModel()
        self.me.setFeatureRadius(feature_radius)
        self.me.setSamplingSize(sampling_size)
        self.me.setNumberOfClusters(number_of_clusters)
        self.me.setNVotState(n_vot)
        self.trained = False

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_TrainingCloud(self, PointCloud points not None,
                          PointCloud_Normal normals not None,
                          unsigned int class_id):
        """One example of class `class_id` (0, 1, ...); points or normals
        that are not finite are left out."""
        if normals.size != points.size:
            raise ValueError("%d points but %d normals"
                             % (points.size, normals.size))
        cdef size_t kept = self.me.addTrainingCloud(
            deref(points.ptr()), deref(normals.ptr()), class_id)
        if kept == 0:
            raise ValueError("no finite point with a finite normal")
        return kept

    def train(self, sigmas=None):
        """Build the model. `sigmas` (one per class) is the vote kernel's
        width; by default a tenth of each class's mean diameter."""
        if self.me.numberOfTrainingClouds() == 0:
            raise RuntimeError("add_TrainingCloud() is required first")
        cdef vector[float] s
        if sigmas is not None:
            for v in sigmas:
                if not v > 0:
                    raise ValueError("sigmas must be > 0")
                s.push_back(v)
        cdef cbool ok
        with nogil:
            ok = self.me.train(s)
        if not ok:
            raise RuntimeError("PCL could not train the model")
        self.trained = True

    @property
    def number_of_classes(self):
        return self.me.numberOfClasses()

    @property
    def number_of_visual_words(self):
        return self.me.numberOfVisualWords()

    @property
    def sigmas(self):
        return list(self.me.sigmas())

    def find_Objects(self, PointCloud scene not None,
                     PointCloud_Normal normals not None, int class_id,
                     non_maxima_radius=None, sigma=None):
        """``(n, 4)`` float32 peaks -- x, y, z of a likely centre and its
        vote density -- strongest first. `non_maxima_radius` (default ten
        sigmas, as in PCL's tutorial) merges peaks closer than that."""
        import numpy as np
        if not self.trained:
            raise RuntimeError("train() or load_Model() is required first")
        if not 0 <= class_id < self.me.numberOfClasses():
            raise ValueError("class_id must be in 0..%d"
                             % (self.me.numberOfClasses() - 1))
        if normals.size != scene.size:
            raise ValueError("%d points but %d normals"
                             % (scene.size, normals.size))
        cdef double s = 0.0 if sigma is None else sigma
        if sigma is not None and not s > 0:
            raise ValueError("sigma must be > 0")
        cdef double r = (10.0 * (s if s > 0 else self.me.sigmas()[class_id])
                         if non_maxima_radius is None else non_maxima_radius)
        cdef vector[float] peaks
        cdef cPointCloud[PointXYZ]* p = scene.ptr()
        cdef cPointCloud[Normal]* n = normals.ptr()
        with nogil:
            self.me.findObjects(deref(p), deref(n), class_id, r, s, peaks)
        return np.asarray(peaks, dtype=np.float32).reshape(-1, 4)

    def save_Model(self, path):
        import os
        if not self.trained:
            raise RuntimeError("nothing trained to save")
        if not self.me.save(os.fsencode(os.fspath(path))):
            raise IOError("could not write %s" % path)

    def load_Model(self, path):
        import os
        if not self.me.load(os.fsencode(os.fspath(path))):
            raise IOError("could not read an ISM model from %s" % path)
        self.trained = True


# --- template matching ----------------------------------------------------------------


cdef class LineRGBD:
    """Template matching on organized colour + depth frames: LINEMOD on
    colour gradients and surface normals (pcl::LineRGBD).

        matcher = pcl.LineRGBD()
        matcher.add_Template(frame, mask, object_id=3)  # mask: (h, w) bool
        for d in matcher.detect(other_frame):
            d["object_id"], d["region"], d["response"]

    A template is the object's pixels in one frame (`mask`); its 3-D box
    is taken from those points, and a detection reports it re-centred on
    the scene points under the detected region. Detections are on
    LINEMOD's 8-pixel grid refined by its spreading, so expect the region
    within a few pixels.
    """

    cdef TemplateMatcher* me
    cdef int templates

    def __cinit__(self):
        self.me = new TemplateMatcher()
        self.templates = 0

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def add_Template(self, PointCloud_PointXYZRGBA frame not None, mask,
                     size_t object_id=0, region=None):
        """Return the template id. `region` (x, y, width, height) defaults
        to the mask's bounding rectangle."""
        import numpy as np
        _check_frame(frame)
        cdef vector[unsigned char] bytes_ = _mask_bytes(mask, frame.width,
                                                        frame.height)
        on = np.argwhere(np.asarray(mask) != 0)
        if len(on) == 0:
            raise ValueError("the mask selects no pixel")
        if region is None:
            (y0, x0), (y1, x1) = on.min(axis=0), on.max(axis=0)
            region = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        cdef int x, y, w, h
        x, y, w, h = [int(v) for v in region]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.width or \
                y + h > frame.height:
            raise ValueError("region must lie inside the frame")
        cdef int tid
        with nogil:
            tid = self.me.addTemplate(frame.thisptr_shared, bytes_, x, y, w, h,
                                      object_id)
        self.templates += 1
        return tid

    def set_DetectionThreshold(self, float threshold):
        """Minimum response, 0..1 (PCL's default 0.75)."""
        self.me.setDetectionThreshold(threshold)

    def set_GradientMagnitudeThreshold(self, float threshold):
        self.me.setGradientMagnitudeThreshold(threshold)

    def set_IntersectionVolumeThreshold(self, float threshold):
        """Overlapping detections whose 3-D boxes share more than this
        fraction are merged (PCL's default 1.0)."""
        self.me.setIntersectionVolumeThreshold(threshold)

    def detect(self, PointCloud_PointXYZRGBA frame not None):
        return self._detect(frame, False, 0, 0, 0)

    def detect_semi_scale_invariant(self, PointCloud_PointXYZRGBA frame not None,
                                    float min_scale=0.6944444,
                                    float max_scale=1.44,
                                    float scale_multiplier=1.2):
        """`detect` over a range of template scales."""
        if not 0 < min_scale <= max_scale or not scale_multiplier > 1:
            raise ValueError("need 0 < min_scale <= max_scale and "
                             "scale_multiplier > 1")
        return self._detect(frame, True, min_scale, max_scale,
                            scale_multiplier)

    cdef _detect(self, PointCloud_PointXYZRGBA frame, bint semi, float lo,
                 float hi, float step):
        _check_frame(frame)
        if self.templates == 0:
            raise RuntimeError("add_Template() is required first")
        cdef vector[int] tids
        cdef vector[int] oids
        cdef vector[float] responses
        cdef vector[int] regions
        cdef vector[float] boxes
        with nogil:
            self.me.detect(frame.thisptr_shared, semi, lo, hi, step, tids,
                           oids, responses, regions, boxes)
        out = []
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> tids.size()):
            out.append({
                "template_id": tids[i],
                "object_id": oids[i],
                "response": responses[i],
                "region": (regions[4 * i], regions[4 * i + 1],
                           regions[4 * i + 2], regions[4 * i + 3]),
                "bounding_box": (boxes[6 * i], boxes[6 * i + 1],
                                 boxes[6 * i + 2], boxes[6 * i + 3],
                                 boxes[6 * i + 4], boxes[6 * i + 5]),
            })
        return out


cdef class DOTMOD:
    """Dominant-orientation template matching on colour gradients
    (pcl::DOTMOD with ColorGradientDOTModality): each `bin_size` square
    keeps the orientations of its strongest gradients, and a window
    scores the fraction of bins agreeing with the template's.

        dot = pcl.DOTMOD(48, 48, bin_size=8)
        dot.add_Template(frame, x, y)
        for x, y, template_id, score in dot.detect(other_frame, 0.8):
            ...

    Sizes and positions are pixels (PCL's own are pixels for the
    template and bins for detections; these are all pixels). The window
    plus half a bin must fit the frame, since PCL reads that far past it
    unchecked.
    """

    cdef DotMatcher* me
    cdef size_t width
    cdef size_t height
    cdef size_t bin

    def __cinit__(self, size_t template_width, size_t template_height,
                  size_t bin_size=8):
        if bin_size < 2:
            raise ValueError("bin_size must be >= 2")
        if template_width < bin_size or template_height < bin_size:
            raise ValueError("the template must span at least one bin")
        self.me = new DotMatcher(template_width, template_height, bin_size)
        self.width = template_width
        self.height = template_height
        self.bin = bin_size

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_GradientMagnitudeThreshold(self, float threshold):
        self.me.setGradientMagnitudeThreshold(threshold)

    def add_Template(self, PointCloud_PointXYZRGBA frame not None, size_t x,
                     size_t y, mask=None):
        """The template window with top-left pixel (x, y); `mask` (the
        frame's shape) limits it to the object, default all of it."""
        import numpy as np
        _check_frame(frame)
        if mask is None:
            mask = np.zeros((frame.height, frame.width), np.uint8)
            mask[y:y + self.height, x:x + self.width] = 1
        cdef vector[unsigned char] bytes_ = _mask_bytes(mask, frame.width,
                                                        frame.height)
        cdef size_t tid = 0
        cdef cbool ok
        with nogil:
            ok = self.me.addTemplate(frame.thisptr_shared, bytes_, x, y, tid)
        if not ok:
            raise ValueError(
                "the %d x %d window at (%d, %d) plus half a bin does not fit "
                "the %d x %d frame" % (self.width, self.height, x, y,
                                       frame.width, frame.height))
        return tid

    def detect(self, PointCloud_PointXYZRGBA frame not None,
               float threshold=0.8):
        """``(x, y, template_id, score)`` for every window scoring above
        `threshold`, best first."""
        _check_frame(frame)
        if self.me.numberOfTemplates() == 0:
            raise RuntimeError("add_Template() is required first")
        cdef vector[int] xs
        cdef vector[int] ys
        cdef vector[int] tids
        cdef vector[float] scores
        with nogil:
            self.me.detect(frame.thisptr_shared, threshold, xs, ys, tids, scores)
        out = []
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> xs.size()):
            out.append((xs[i], ys[i], tids[i], float(scores[i])))
        out.sort(key=lambda d: -d[3])
        return out
