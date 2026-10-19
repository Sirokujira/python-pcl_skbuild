# distutils: language = c++
# cython: language_level=3
"""pcl/common algorithms: PCA, line / plane intersections, the histogram
norms, TransformationFromCorrespondences and projection-matrix
estimation.

The rest of pcl/common -- centroids, covariance, min / max, distances,
angles, random clouds, colour tables -- is a line of numpy each and is
not repeated here (see the README). Everything goes through
pcl/compat/common_args.h.
"""

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from pcl.pxd.compat.common_args cimport (
    PrincipalComponents, estimateProjectionMatrix, histogramNorm,
    lineWithLine, planeWithPlane, threePlanes,
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
