# distutils: language = c++
# cython: language_level=3
"""pcl/common algorithms: PCA, line / plane intersections, the histogram
norms, TransformationFromCorrespondences, projection-matrix
estimation and PosesFromMatches.

The rest of pcl/common -- centroids, covariance, min / max, distances,
angles, random clouds, colour tables -- is a line of numpy each and is
not repeated here (see the README). Everything goes through
pcl/compat/common_args.h.
"""

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from pcl.pxd.compat.common_args cimport (
    PrincipalComponents, estimateProjectionMatrix, histogramNorm,
    lineWithLine, planeWithPlane, posesFromMatches, threePlanes,
    transformationFromCorrespondences)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.point_types cimport PointXYZ

from pcl._pointcloud cimport PointCloud


cdef class PCA:
    """Principal component analysis of a cloud (pcl::PCA).

    `eigenvectors` are the columns of a 3 x 3 matrix, largest variance
    first, the third made right-handed; `eigenvalues` are the variances
    along them (sample covariance, n - 1). `project` gives a cloud's
    coordinates in that frame about the mean, `reconstruct` maps them
    back.
    """

    cdef PrincipalComponents* me
    cdef PointCloud cloud

    def __cinit__(self, PointCloud cloud not None):
        if cloud.size < 3:
            raise ValueError("PCA needs at least 3 points")
        self.me = new PrincipalComponents()
        self.cloud = cloud
        self.me.setInputCloud(cloud.thisptr_shared)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    @property
    def mean(self):
        import numpy as np
        out = np.empty(3, dtype=np.float32)
        cdef float[::1] v = out
        self.me.mean(&v[0])
        return out

    @property
    def eigenvectors(self):
        import numpy as np
        out = np.empty(9, dtype=np.float32)
        cdef float[::1] v = out
        self.me.eigenVectors(&v[0])
        return out.reshape(3, 3, order="F")

    @property
    def eigenvalues(self):
        import numpy as np
        out = np.empty(3, dtype=np.float32)
        cdef float[::1] v = out
        self.me.eigenValues(&v[0])
        return out

    def project(self, PointCloud cloud not None):
        cdef PointCloud out = PointCloud()
        self.me.project(deref(cloud.ptr()), deref(out.ptr()))
        return out

    def reconstruct(self, PointCloud projected not None):
        cdef PointCloud out = PointCloud()
        self.me.reconstruct(deref(projected.ptr()), deref(out.ptr()))
        return out


cdef _vector(values, int n, str what):
    import numpy as np
    arr = np.ascontiguousarray(values, dtype=np.float32).ravel()
    if arr.shape != (n,):
        raise ValueError("%s must have %d values" % (what, n))
    return arr


def line_line_intersection(line_a, line_b, double sqr_eps=1e-4):
    """The point where two lines meet (pcl::lineWithLineIntersection),
    each given as (point, direction) -- six values, as
    SampleConsensusModelLine returns them. It is the midpoint of their
    closest approach; None when they are parallel or pass further apart
    than sqrt(`sqr_eps`)."""
    a = _vector(line_a, 6, "line_a")
    b = _vector(line_b, 6, "line_b")
    import numpy as np
    out = np.empty(3, dtype=np.float32)
    cdef float[::1] va = a
    cdef float[::1] vb = b
    cdef float[::1] vo = out
    if not lineWithLine(&va[0], &vb[0], &vo[0], sqr_eps):
        return None
    return out


def plane_plane_intersection(plane_a, plane_b, double angular_tolerance=0.1):
    """The line two planes ``ax + by + cz + d = 0`` share
    (pcl::planeWithPlaneIntersection) as (point, direction), or None when
    they are within `angular_tolerance` radians of parallel."""
    a = _vector(plane_a, 4, "plane_a")
    b = _vector(plane_b, 4, "plane_b")
    import numpy as np
    out = np.empty(6, dtype=np.float32)
    cdef float[::1] va = a
    cdef float[::1] vb = b
    cdef float[::1] vo = out
    if not planeWithPlane(&va[0], &vb[0], &vo[0], angular_tolerance):
        return None
    return out[:3], out[3:]


def three_planes_intersection(plane_a, plane_b, plane_c,
                              double determinant_tolerance=1e-6):
    """The point three planes share (pcl::threePlanesIntersection), or
    None when two of them are (nearly) parallel."""
    a = _vector(plane_a, 4, "plane_a")
    b = _vector(plane_b, 4, "plane_b")
    c = _vector(plane_c, 4, "plane_c")
    import numpy as np
    out = np.empty(3, dtype=np.float32)
    cdef float[::1] va = a
    cdef float[::1] vb = b
    cdef float[::1] vc = c
    cdef float[::1] vo = out
    if not threePlanes(&va[0], &vb[0], &vc[0], &vo[0], determinant_tolerance):
        return None
    return out


_NORMS = {"L1": 0, "L2_SQR": 1, "L2": 2, "LINF": 3, "JM": 4, "B": 5,
          "SUBLINEAR": 6, "CS": 7, "DIV": 8, "PF": 9, "K": 10, "KL": 11,
          "HIK": 12}


def histogram_distance(a, b, str norm="L2", float p1=1.0, float p2=1.0):
    """One of pcl/common/norms.h's distances between two histograms of
    one length: "L1", "L2_SQR", "L2", "LINF", "JM" (Jeffries-Matusita),
    "B" (Bhattacharyya), "SUBLINEAR", "CS" (chi-square), "DIV"
    (divergence), "PF" (patch-fitting; `p1`, `p2`), "K" (`p1`, `p2`), "KL"
    (Kullback-Leibler) and "HIK" (histogram intersection kernel). Several
    take logarithms or square roots of the bins, so they want
    non-negative values (KL wants positive ones)."""
    import numpy as np
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.shape != vb.shape or va.size == 0:
        raise ValueError("the histograms must have one, non-zero length")
    if norm not in _NORMS:
        raise ValueError("norm must be one of %s" % ", ".join(_NORMS))
    cdef vector[float] ca = va.tolist()
    cdef vector[float] cb = vb.tolist()
    return histogramNorm(ca, cb, _NORMS[norm], p1, p2)


def transformation_from_correspondences(PointCloud source not None,
                                        PointCloud target not None,
                                        weights=None):
    """The weighted least-squares rigid motion taking `source` onto
    `target`, point i onto point i (pcl::TransformationFromCorrespondences),
    as a 4 x 4 float32 matrix."""
    import numpy as np
    if source.size != target.size or source.size < 3:
        raise ValueError("need two clouds of one size, at least 3 points")
    cdef vector[float] w
    if weights is not None:
        warr = np.asarray(weights, dtype=np.float32).ravel()
        if warr.shape != (source.size,) or (warr < 0).any():
            raise ValueError("one non-negative weight per point")
        w = warr.tolist()
    out = np.empty(16, dtype=np.float32)
    cdef float[::1] vo = out
    transformationFromCorrespondences(deref(source.ptr()), deref(target.ptr()),
                                      w, &vo[0])
    return out.reshape(4, 4, order="F")


def estimate_projection_matrix(PointCloud cloud not None):
    """``(P, K, residual)`` for an organized cloud
    (pcl::estimateProjectionMatrix): the 3 x 4 matrix taking each point
    to its pixel, the camera matrix
    pcl::getCameraMatrixFromProjectionMatrix reads off it, and the
    least-squares residual (near 0 for a cloud a pinhole camera could
    have produced). P is the unit-norm least-squares solution, signed so
    P[0, 0] > 0 -- K [R | t] up to that scale; K is upper triangular with
    K[2, 2] = 1."""
    import numpy as np
    if cloud.height <= 1:
        raise ValueError("estimate_projection_matrix needs an organized cloud")
    p = np.empty(12, dtype=np.float32)
    k = np.empty(9, dtype=np.float32)
    cdef float[::1] vp = p
    cdef float[::1] vk = k
    residual = estimateProjectionMatrix(cloud.thisptr_shared, &vp[0], &vk[0])
    return p.reshape(3, 4), k.reshape(3, 3), float(residual)


def poses_from_matches(points1, points2, poses, distances, int size=3,
                       int max_tested_combinations=1000, int max_results=-1,
                       double max_distance_error=0.2):
    """Pose hypotheses from 6-DoF matches (pcl::PosesFromMatches), e.g.
    NARF features matched between a scene and a model.

    Match i pairs ``points1[i]`` (scene) with ``points2[i]`` (model) and
    carries ``poses[i]``, the 4 x 4 transform taking model coordinates to
    scene coordinates that the match alone implies; ``distances[i]`` is
    its descriptor distance. PCL expects the matches sorted best first.
    `size` is how many matches each hypothesis is built from:

    * 1 -- every match's own pose, as given;
    * 2 -- pairs whose point distances agree within
      `max_distance_error` (a fraction) in both clouds, also along each
      match's own axes, fitted to all of that;
    * 3 -- triples whose three pairwise distances agree, fitted to the
      three point pairs (the poses are not used).

    Pairs and triples are tried best first, at most
    `max_tested_combinations` of them; `max_results` < 0 keeps every
    hypothesis. Returns a list of ``(pose, score, matches)``: the 4 x 4
    float32 transform, the mean distance of the matches used (PCL calls
    it a score, but lower is better) and their indices."""
    import numpy as np
    p1 = np.ascontiguousarray(points1, dtype=np.float32)
    p2 = np.ascontiguousarray(points2, dtype=np.float32)
    t = np.asarray(poses, dtype=np.float32)
    d = np.ascontiguousarray(distances, dtype=np.float32).ravel()
    n = len(d)
    if p1.shape != (n, 3) or p2.shape != (n, 3) or t.shape != (n, 4, 4):
        raise ValueError("need (n, 3) points1 and points2, (n, 4, 4) poses "
                         "and n distances")
    if size not in (1, 2, 3):
        raise ValueError("size must be 1, 2 or 3")
    if max_tested_combinations < 1 or max_distance_error < 0:
        raise ValueError("max_tested_combinations must be >= 1 and "
                         "max_distance_error >= 0")
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()
            and np.isfinite(t).all() and np.isfinite(d).all()):
        raise ValueError("matches must be finite")
    cdef vector[float] v1 = p1.ravel().tolist()
    cdef vector[float] v2 = p2.ravel().tolist()
    cdef vector[float] vt = np.transpose(t, (0, 2, 1)).ravel().tolist()
    cdef vector[float] vd = d.tolist()
    cdef vector[float] out_poses, out_scores
    cdef vector[int] out_indices
    with nogil:
        posesFromMatches(v1, v2, vt, vd, size, max_tested_combinations,
                         max_results, max_distance_error, out_poses,
                         out_scores, out_indices)
    all_poses = np.asarray(out_poses, dtype=np.float32).reshape(-1, 4, 4)
    result = []
    cdef Py_ssize_t i, k
    for i in range(<Py_ssize_t> out_scores.size()):
        used = []
        for k in range(size):
            used.append(out_indices[i * size + k])
        result.append((all_poses[i].T.copy(), float(out_scores[i]), used))
    return result
