# distutils: language = c++
# cython: language_level=3
"""Segmentation wrappers, named after sirokujira/python-pcl
(pcl/pxi/Segmentation/).

Reach them through `PointCloud.make_segmenter()` and
`PointCloud.make_EuclideanClusterExtraction()`.

The SAC method and model constants are re-exported here so callers can
write `pcl.SACMODEL_PLANE` / `pcl.SAC_RANSAC` exactly as in python-pcl;
they come from the generated
src/pcl/pxd/sample_consensus/{method,model}_types.pxd.
"""

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_indices cimport PointIndices
from pcl.pxd.model_coefficients cimport ModelCoefficients
from pcl.pxd.segmentation.sac_segmentation cimport (
    SACSegmentation as cSACSegmentation)
from pcl.pxd.segmentation.extract_clusters cimport (
    EuclideanClusterExtraction as cEuclideanClusterExtraction)
cimport pcl.pxd.sample_consensus.method_types as method_types
cimport pcl.pxd.sample_consensus.model_types as model_types

from pcl._pointcloud cimport PointCloud


cdef class Segmentation:
    """Fit a model to a cloud with RANSAC and friends
    (pcl::SACSegmentation)."""

    cdef cSACSegmentation[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSACSegmentation[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_optimize_coefficients(self, bint b):
        self.me.setOptimizeCoefficients(b)

    def set_model_type(self, int m):
        """Model to fit, e.g. ``pcl.SACMODEL_PLANE``."""
        self.me.setModelType(m)

    def set_method_type(self, int m):
        """Estimation method, e.g. ``pcl.SAC_RANSAC``."""
        self.me.setMethodType(m)

    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold(d)

    def set_MaxIterations(self, int count):
        self.me.setMaxIterations(count)

    def set_probability(self, double probability):
        self.me.setProbability(probability)

    def set_radius_limits(self, double min_radius, double max_radius):
        self.me.setRadiusLimits(min_radius, max_radius)

    def segment(self):
        """Run the segmentation.

        Returns ``(inlier_indices, model_coefficients)`` as plain Python
        lists, matching python-pcl.
        """
        cdef PointIndices inliers
        cdef ModelCoefficients coefficients
        with nogil:
            self.me.segment(inliers, coefficients)
        return [inliers.indices[i] for i in range(inliers.indices.size())], \
               [coefficients.values[i]
                for i in range(coefficients.values.size())]


cdef class EuclideanClusterExtraction:
    """Split a cloud into clusters by euclidean distance
    (pcl::EuclideanClusterExtraction)."""

    cdef cEuclideanClusterExtraction[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cEuclideanClusterExtraction[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_ClusterTolerance(self, double b):
        self.me.setClusterTolerance(b)

    def set_MinClusterSize(self, int min_size):
        self.me.setMinClusterSize(min_size)

    def set_MaxClusterSize(self, int max_size):
        self.me.setMaxClusterSize(max_size)

    def Extract(self):
        """Return the clusters as a list of index lists."""
        cdef vector[PointIndices] clusters
        with nogil:
            self.me.extract(clusters)
        return [
            [clusters[i].indices[j] for j in range(clusters[i].indices.size())]
            for i in range(clusters.size())
        ]


# python-pcl exposes these as module-level constants; keep the same names.
SAC_RANSAC = method_types.SAC_RANSAC
SAC_LMEDS = method_types.SAC_LMEDS
SAC_MSAC = method_types.SAC_MSAC
SAC_RRANSAC = method_types.SAC_RRANSAC
SAC_RMSAC = method_types.SAC_RMSAC
SAC_MLESAC = method_types.SAC_MLESAC
SAC_PROSAC = method_types.SAC_PROSAC

SACMODEL_PLANE = model_types.SACMODEL_PLANE
SACMODEL_LINE = model_types.SACMODEL_LINE
SACMODEL_CIRCLE2D = model_types.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = model_types.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = model_types.SACMODEL_SPHERE
SACMODEL_CYLINDER = model_types.SACMODEL_CYLINDER
SACMODEL_CONE = model_types.SACMODEL_CONE
SACMODEL_PARALLEL_LINE = model_types.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = model_types.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_PLANE = model_types.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PLANE = model_types.SACMODEL_NORMAL_PLANE
SACMODEL_NORMAL_SPHERE = model_types.SACMODEL_NORMAL_SPHERE
SACMODEL_STICK = model_types.SACMODEL_STICK
