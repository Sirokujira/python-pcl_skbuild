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

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_indices cimport PointIndices
from pcl.pxd.model_coefficients cimport ModelCoefficients
from pcl.pxd.segmentation.sac_segmentation cimport (
    SACSegmentation as cSACSegmentation)
from pcl.pxd.segmentation.extract_clusters cimport (
    EuclideanClusterExtraction as cEuclideanClusterExtraction)
from pcl.pxd.segmentation.sac_segmentation_normals cimport (
    SACSegmentationFromNormals as cSACSegmentationFromNormals)
from pcl.pxd.segmentation.progressive_morphological_filter cimport (
    ProgressiveMorphologicalFilter as cProgressiveMorphologicalFilter)
from pcl.pxd.point_types cimport Normal
from pcl.pxd.compat.eigen_args cimport setSegmentationAxis
cimport pcl.pxd.sample_consensus.method_types as method_types
cimport pcl.pxd.sample_consensus.model_types as model_types

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_Normal


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

    def set_axis(self, float x, float y, float z):
        """Axis a parallel/perpendicular model is measured against."""
        setSegmentationAxis(deref(self.me), x, y, z)

    def set_eps_angle(self, double angle):
        self.me.setEpsAngle(angle)

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
SACMODEL_TORUS = model_types.SACMODEL_TORUS
SACMODEL_PARALLEL_LINES = model_types.SACMODEL_PARALLEL_LINES
SACMODEL_REGISTRATION = model_types.SACMODEL_REGISTRATION
SACMODEL_REGISTRATION_2D = model_types.SACMODEL_REGISTRATION_2D
SACMODEL_NORMAL_PARALLEL_PLANE = model_types.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_ELLIPSE3D = model_types.SACMODEL_ELLIPSE3D


cdef class SegmentationNormal:
    """Model fitting that also uses surface normals
    (pcl::SACSegmentationFromNormals).

    What `SACMODEL_CYLINDER` and the `NORMAL_*` models need — they score
    a candidate on normal agreement as well as distance, so
    `set_InputNormals` is required.

        normals = cloud.make_NormalEstimation()
        normals.set_KSearch(50)
        seg = cloud.make_segmenter_normals()
        seg.set_InputNormals(normals.compute_cloud())
    """

    cdef cSACSegmentationFromNormals[PointXYZ, Normal]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSACSegmentationFromNormals[PointXYZ, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)

    def set_optimize_coefficients(self, bint b):
        self.me.setOptimizeCoefficients(b)

    def set_model_type(self, int m):
        self.me.setModelType(m)

    def set_method_type(self, int m):
        self.me.setMethodType(m)

    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold(d)

    def set_max_iterations(self, int count):
        self.me.setMaxIterations(count)

    def set_MaxIterations(self, int count):
        """python-pcl spelling of `set_max_iterations`."""
        self.me.setMaxIterations(count)

    def set_normal_distance_weight(self, double weight):
        """How much normal agreement counts against distance, 0..1."""
        self.me.setNormalDistanceWeight(weight)

    def set_radius_limits(self, double min_radius, double max_radius):
        """Radius range for the cylinder and sphere models."""
        self.me.setRadiusLimits(min_radius, max_radius)

    def set_axis(self, float x, float y, float z):
        setSegmentationAxis(deref(self.me), x, y, z)

    def set_eps_angle(self, double angle):
        self.me.setEpsAngle(angle)

    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """Cone model opening-angle range."""
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)

    def segment(self):
        """Returns ``(inlier_indices, model_coefficients)``."""
        cdef PointIndices inliers
        cdef ModelCoefficients coefficients
        with nogil:
            self.me.segment(inliers, coefficients)
        return [inliers.indices[i] for i in range(inliers.indices.size())], \
               [coefficients.values[i]
                for i in range(coefficients.values.size())]


cdef class ProgressiveMorphologicalFilter:
    """Separate ground from objects in a terrain scan
    (pcl::ProgressiveMorphologicalFilter).

    `extract()` returns the ground point indices; feed them to
    `ExtractIndices` to get either half as a cloud.
    """

    cdef cProgressiveMorphologicalFilter[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cProgressiveMorphologicalFilter[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_MaxWindowSize(self, int size):
        self.me.setMaxWindowSize(size)

    def get_MaxWindowSize(self):
        return self.me.getMaxWindowSize()

    def set_Slope(self, float slope):
        self.me.setSlope(slope)

    def get_Slope(self):
        return self.me.getSlope()

    def set_InitialDistance(self, float distance):
        self.me.setInitialDistance(distance)

    def get_InitialDistance(self):
        return self.me.getInitialDistance()

    def set_MaxDistance(self, float distance):
        self.me.setMaxDistance(distance)

    def get_MaxDistance(self):
        return self.me.getMaxDistance()

    def set_CellSize(self, float size):
        self.me.setCellSize(size)

    def get_CellSize(self):
        return self.me.getCellSize()

    def set_Base(self, float base):
        self.me.setBase(base)

    def set_Exponential(self, bint exponential):
        self.me.setExponential(exponential)

    def extract(self):
        """Return the indices of the points classified as ground."""
        cdef vector[int] ground
        with nogil:
            self.me.extract(ground)
        return [ground[i] for i in range(ground.size())]
