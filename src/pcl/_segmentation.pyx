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

import traceback

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
from pcl.pxd.segmentation.min_cut_segmentation cimport (
    MinCutSegmentation as cMinCutSegmentation)
from pcl.pxd.segmentation.progressive_morphological_filter cimport (
    ProgressiveMorphologicalFilter as cProgressiveMorphologicalFilter)
from pcl.pxd.point_types cimport Normal
from pcl.pxd.compat.eigen_args cimport setSegmentationAxis
from pcl.pxd.segmentation.conditional_euclidean_clustering cimport (
    ConditionalEuclideanClustering as cConditionalEuclideanClustering)
from pcl.pxd.compat.organized_args cimport (
    ClusterConditionFn, setClusterCondition)
from libcpp.memory cimport shared_ptr
from pcl.pxd.point_types cimport PointXYZRGB
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.segmentation.region_growing cimport (
    RegionGrowing as cRegionGrowing)
from pcl.pxd.segmentation.region_growing_rgb cimport (
    RegionGrowingRGB as cRegionGrowingRGB)
from pcl.pxd.point_types cimport Label
from pcl.pxd.segmentation.extract_polygonal_prism_data cimport (
    ExtractPolygonalPrismData as cExtractPolygonalPrismData)
from pcl.pxd.segmentation.segment_differences cimport (
    SegmentDifferences as cSegmentDifferences)
from pcl.pxd.segmentation.approximate_progressive_morphological_filter cimport (
    ApproximateProgressiveMorphologicalFilter as cApproximateProgressiveMorphologicalFilter)
from pcl.pxd.segmentation.seeded_hue_segmentation cimport (
    SeededHueSegmentation as cSeededHueSegmentation)
from pcl.pxd.segmentation.organized_multi_plane_segmentation cimport (
    OrganizedMultiPlaneSegmentation as cOrganizedMultiPlaneSegmentation)
from pcl.pxd.point_types cimport PointXYZRGBA
from pcl.pxd.segmentation.supervoxel_clustering cimport (
    SupervoxelClustering as cSupervoxelClustering)
from pcl.pxd.compat.supervoxel_args cimport (
    SupervoxelSet, supervoxelExtract, supervoxelSummary,
    supervoxelPointLabels, lccpSegment)
from pcl._pointtypes cimport PointCloud_PointXYZRGBA
cimport pcl.pxd.sample_consensus.method_types as method_types
cimport pcl.pxd.sample_consensus.model_types as model_types

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointXYZRGB, wrap_xyzrgb_cloud)


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
        # SAC_RRANSAC never returns on data with outliers in PCL 1.14 --
        # see SampleConsensus in _sac.pyx for the mechanism.
        if m == method_types.SAC_RRANSAC:
            raise ValueError(
                "SAC_RRANSAC is refused: PCL's RandomizedRandomSampleConsensus "
                "never returns on data with outliers; use SAC_RANSAC")
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
        # SAC_RRANSAC never returns on data with outliers in PCL 1.14 --
        # see SampleConsensus in _sac.pyx for the mechanism.
        if m == method_types.SAC_RRANSAC:
            raise ValueError(
                "SAC_RRANSAC is refused: PCL's RandomizedRandomSampleConsensus "
                "never returns on data with outliers; use SAC_RANSAC")
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


cdef class MinCutSegmentation:
    """Foreground/background split by graph min-cut
    (pcl::MinCutSegmentation).

    Mark a few points as foreground and it partitions the rest around
    them. `extract()` returns two clusters: background first, then
    foreground.
    """

    cdef cMinCutSegmentation[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cMinCutSegmentation[PointXYZ]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_ForegroundPoints(self, PointCloud pc not None):
        """Seed points known to be foreground. Required."""
        self.me.setForegroundPoints(pc.thisptr_shared)

    def set_Sigma(self, double sigma):
        self.me.setSigma(sigma)

    def get_Sigma(self):
        return self.me.getSigma()

    def set_Radius(self, double radius):
        self.me.setRadius(radius)

    def get_Radius(self):
        return self.me.getRadius()

    def set_SourceWeight(self, double weight):
        self.me.setSourceWeight(weight)

    def get_SourceWeight(self):
        return self.me.getSourceWeight()

    def set_NumberOfNeighbours(self, unsigned int count):
        self.me.setNumberOfNeighbours(count)

    def get_NumberOfNeighbours(self):
        return self.me.getNumberOfNeighbours()

    def get_MaxFlow(self):
        """Cost of the cut; only meaningful after `extract()`."""
        return self.me.getMaxFlow()

    def extract(self):
        """Return the clusters as a list of index lists."""
        cdef vector[PointIndices] clusters
        with nogil:
            self.me.extract(clusters)
        return [
            [clusters[i].indices[j] for j in range(clusters[i].indices.size())]
            for i in range(clusters.size())
        ]


cdef list _clusters_to_lists(vector[PointIndices]& clusters):
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


cdef class RegionGrowing:
    """Grow smooth regions outwards from low-curvature seeds
    (pcl::RegionGrowing).

    Needs normals: pass what `NormalEstimation.compute_cloud()` returns
    to `set_InputNormals()`. A neighbour joins a region when its normal
    deviates from the seed's by less than `set_SmoothnessThreshold`
    (radians); growth stops at points whose curvature exceeds
    `set_CurvatureThreshold`. `extract()` returns one index list per
    region, `get_ColoredCloud()` the same points painted per region.
    """

    cdef cRegionGrowing[PointXYZ, Normal]* me
    cdef PointCloud _input

    def __cinit__(self, PointCloud pc=None):
        self.me = new cRegionGrowing[PointXYZ, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc

    def set_InputNormals(self, PointCloud_Normal normals not None):
        """Normals for the input cloud, one per point. Required."""
        self.me.setInputNormals(normals.thisptr_shared)

    def set_MinClusterSize(self, int size):
        self.me.setMinClusterSize(size)

    def get_MinClusterSize(self):
        return self.me.getMinClusterSize()

    def set_MaxClusterSize(self, int size):
        self.me.setMaxClusterSize(size)

    def get_MaxClusterSize(self):
        return self.me.getMaxClusterSize()

    def set_SmoothModeFlag(self, bint value):
        """Compare each candidate's normal with its own neighbour's (True,
        the default) or with the region's seed (False)."""
        self.me.setSmoothModeFlag(value)

    def get_SmoothModeFlag(self):
        return self.me.getSmoothModeFlag()

    def set_CurvatureTestFlag(self, bint value):
        self.me.setCurvatureTestFlag(value)

    def get_CurvatureTestFlag(self):
        return self.me.getCurvatureTestFlag()

    def set_ResidualTestFlag(self, bint value):
        self.me.setResidualTestFlag(value)

    def get_ResidualTestFlag(self):
        return self.me.getResidualTestFlag()

    def set_SmoothnessThreshold(self, float theta):
        """Largest angle between normals, in radians, that still joins."""
        self.me.setSmoothnessThreshold(theta)

    def get_SmoothnessThreshold(self):
        return self.me.getSmoothnessThreshold()

    def set_ResidualThreshold(self, float residual):
        self.me.setResidualThreshold(residual)

    def get_ResidualThreshold(self):
        return self.me.getResidualThreshold()

    def set_CurvatureThreshold(self, float curvature):
        self.me.setCurvatureThreshold(curvature)

    def get_CurvatureThreshold(self):
        return self.me.getCurvatureThreshold()

    def set_NumberOfNeighbours(self, unsigned int count):
        self.me.setNumberOfNeighbours(count)

    def get_NumberOfNeighbours(self):
        return self.me.getNumberOfNeighbours()

    def extract(self):
        """Return the regions as a list of index lists."""
        # Without normals PCL prints an error and hands back no clusters
        # -- a silent no-op from Python's side, so say what is missing.
        if not self.me.getInputNormals():
            raise RuntimeError(
                "RegionGrowing needs normals: call set_InputNormals() with "
                "the cloud NormalEstimation.compute_cloud() returns")
        cdef vector[PointIndices] clusters
        with nogil:
            self.me.extract(clusters)
        return _clusters_to_lists(clusters)

    def get_SegmentFromPoint(self, int index):
        """Return the indices of the region containing point `index`
        (empty when that point belongs to no region)."""
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        if index < 0 or index >= self._input.size:
            raise IndexError(
                "point index %d out of range for a cloud of %d points"
                % (index, self._input.size))
        if not self.me.getInputNormals():
            raise RuntimeError(
                "RegionGrowing needs normals: call set_InputNormals() with "
                "the cloud NormalEstimation.compute_cloud() returns")
        cdef PointIndices cluster
        with nogil:
            self.me.getSegmentFromPoint(index, cluster)
        cdef list result = []
        cdef size_t j
        for j in range(cluster.indices.size()):
            result.append(cluster.indices[j])
        return result

    def get_ColoredCloud(self):
        """Return the input points as a :class:`PointCloud_PointXYZRGB`
        with every region painted a random colour (points in no region
        stay black). Only available after `extract()`."""
        cdef shared_ptr[cPointCloud[PointXYZRGB]] colored
        colored = self.me.getColoredCloud()
        # PCL returns a null pointer until extract() has produced regions.
        if not colored:
            raise RuntimeError(
                "no regions yet: call extract() before get_ColoredCloud()")
        return wrap_xyzrgb_cloud(colored)


cdef class RegionGrowingRGB:
    """Region growing driven by colour (pcl::RegionGrowingRGB) on a
    :class:`PointCloud_PointXYZRGB`.

    Neighbours within `set_DistanceThreshold` join when their colours
    differ by less than `set_PointColorThreshold`; afterwards regions
    whose mean colours differ by less than `set_RegionColorThreshold`
    are merged, and regions below `set_MinClusterSize` are absorbed by
    their nearest neighbour. Normals are optional and only consulted when
    `set_NormalTestFlag(True)`.
    """

    cdef cRegionGrowingRGB[PointXYZRGB, Normal]* me
    cdef PointCloud_PointXYZRGB _input

    def __cinit__(self, PointCloud_PointXYZRGB pc=None):
        self.me = new cRegionGrowingRGB[PointXYZRGB, Normal]()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc

    def set_InputNormals(self, PointCloud_Normal normals not None):
        """Normals for the input cloud; needed only with the normal test
        (or curvature / residual tests) switched on."""
        self.me.setInputNormals(normals.thisptr_shared)

    def set_MinClusterSize(self, int size):
        self.me.setMinClusterSize(size)

    def get_MinClusterSize(self):
        return self.me.getMinClusterSize()

    def set_MaxClusterSize(self, int size):
        self.me.setMaxClusterSize(size)

    def get_MaxClusterSize(self):
        return self.me.getMaxClusterSize()

    def set_SmoothModeFlag(self, bint value):
        self.me.setSmoothModeFlag(value)

    def get_SmoothModeFlag(self):
        return self.me.getSmoothModeFlag()

    def set_CurvatureTestFlag(self, bint value):
        self.me.setCurvatureTestFlag(value)

    def get_CurvatureTestFlag(self):
        return self.me.getCurvatureTestFlag()

    def set_ResidualTestFlag(self, bint value):
        self.me.setResidualTestFlag(value)

    def get_ResidualTestFlag(self):
        return self.me.getResidualTestFlag()

    def set_NormalTestFlag(self, bint value):
        """Also require normals to agree (needs `set_InputNormals`)."""
        self.me.setNormalTestFlag(value)

    def get_NormalTestFlag(self):
        return self.me.getNormalTestFlag()

    def set_SmoothnessThreshold(self, float theta):
        self.me.setSmoothnessThreshold(theta)

    def get_SmoothnessThreshold(self):
        return self.me.getSmoothnessThreshold()

    def set_ResidualThreshold(self, float residual):
        self.me.setResidualThreshold(residual)

    def get_ResidualThreshold(self):
        return self.me.getResidualThreshold()

    def set_CurvatureThreshold(self, float curvature):
        self.me.setCurvatureThreshold(curvature)

    def get_CurvatureThreshold(self):
        return self.me.getCurvatureThreshold()

    def set_NumberOfNeighbours(self, unsigned int count):
        self.me.setNumberOfNeighbours(count)

    def get_NumberOfNeighbours(self):
        return self.me.getNumberOfNeighbours()

    def set_PointColorThreshold(self, float thresh):
        """Largest colour difference between neighbours that still joins
        (Euclidean distance in RGB, 0..441)."""
        self.me.setPointColorThreshold(thresh)

    def get_PointColorThreshold(self):
        return self.me.getPointColorThreshold()

    def set_RegionColorThreshold(self, float thresh):
        """Regions whose mean colours are closer than this are merged."""
        self.me.setRegionColorThreshold(thresh)

    def get_RegionColorThreshold(self):
        return self.me.getRegionColorThreshold()

    def set_DistanceThreshold(self, float thresh):
        """How far apart two points may be and still count as neighbours
        (in the cloud's own units)."""
        self.me.setDistanceThreshold(thresh)

    def get_DistanceThreshold(self):
        return self.me.getDistanceThreshold()

    def set_NumberOfRegionNeighbours(self, unsigned int count):
        self.me.setNumberOfRegionNeighbours(count)

    def get_NumberOfRegionNeighbours(self):
        return self.me.getNumberOfRegionNeighbours()

    def extract(self):
        """Return the regions as a list of index lists."""
        if self.me.getNormalTestFlag() and not self.me.getInputNormals():
            raise RuntimeError(
                "the normal test is on but no normals were set: call "
                "set_InputNormals() or set_NormalTestFlag(False)")
        cdef vector[PointIndices] clusters
        with nogil:
            self.me.extract(clusters)
        return _clusters_to_lists(clusters)

    def get_SegmentFromPoint(self, int index):
        """Return the indices of the region containing point `index`
        (empty when that point belongs to no region)."""
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        if index < 0 or index >= self._input.size:
            raise IndexError(
                "point index %d out of range for a cloud of %d points"
                % (index, self._input.size))
        if self.me.getNormalTestFlag() and not self.me.getInputNormals():
            raise RuntimeError(
                "the normal test is on but no normals were set: call "
                "set_InputNormals() or set_NormalTestFlag(False)")
        cdef PointIndices cluster
        with nogil:
            self.me.getSegmentFromPoint(index, cluster)
        cdef list result = []
        cdef size_t j
        for j in range(cluster.indices.size()):
            result.append(cluster.indices[j])
        return result

    def get_ColoredCloud(self):
        """Return the input points as a new :class:`PointCloud_PointXYZRGB`
        with every region painted a random colour. Only available after
        `extract()`."""
        cdef shared_ptr[cPointCloud[PointXYZRGB]] colored
        colored = self.me.getColoredCloud()
        if not colored:
            raise RuntimeError(
                "no regions yet: call extract() before get_ColoredCloud()")
        return wrap_xyzrgb_cloud(colored)


cdef list _indices_to_list(PointIndices& indices):
    cdef list result = []
    cdef size_t j
    for j in range(indices.indices.size()):
        result.append(indices.indices[j])
    return result


cdef class ExtractPolygonalPrismData:
    """The points inside a prism over a planar polygon
    (pcl::ExtractPolygonalPrismData).

    The polygon is a hull cloud (what `ConvexHull.reconstruct()` returns
    for a planar cloud); points whose projection falls inside it and
    whose height above the plane is within `set_HeightLimits` are
    returned by `segment()` as indices into the input. The "objects on a
    table" step of PCL's tutorials.
    """

    cdef cExtractPolygonalPrismData[PointXYZ]* me
    cdef bint has_hull

    def __cinit__(self, PointCloud pc=None):
        self.me = new cExtractPolygonalPrismData[PointXYZ]()
        self.has_hull = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_InputPlanarHull(self, PointCloud hull not None):
        """The polygon, as a planar hull cloud of at least three points."""
        if hull.size < 3:
            raise ValueError("a planar hull needs at least 3 points")
        self.me.setInputPlanarHull(hull.thisptr_shared)
        self.has_hull = True

    def set_HeightLimits(self, double height_min, double height_max):
        """Signed distance range above the hull's plane that counts as
        inside (PCL's defaults are 0 and FLT_MAX)."""
        if height_max < height_min:
            raise ValueError("height_max must be at least height_min")
        self.me.setHeightLimits(height_min, height_max)

    def set_ViewPoint(self, float x, float y, float z):
        """Which side of the plane is "above": the plane normal is
        flipped to face this point."""
        self.me.setViewPoint(x, y, z)

    def segment(self):
        """Return the indices of the points inside the prism."""
        if not self.has_hull:
            raise RuntimeError(
                "set_InputPlanarHull(hull) is required before segment()")
        cdef PointIndices output
        with nogil:
            self.me.segment(output)
        return _indices_to_list(output)


cdef class SegmentDifferences:
    """The points of one cloud that are missing from another
    (pcl::SegmentDifferences).

    `segment()` returns the input points that have no target point
    within `set_DistanceThreshold` -- the threshold is the SQUARED
    distance, as in PCL.
    """

    cdef cSegmentDifferences[PointXYZ]* me
    cdef bint has_target

    def __cinit__(self, PointCloud pc=None):
        self.me = new cSegmentDifferences[PointXYZ]()
        self.has_target = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_TargetCloud(self, PointCloud pc not None):
        self.me.setTargetCloud(pc.thisptr_shared)
        self.has_target = True

    def set_DistanceThreshold(self, double sqr_threshold):
        """Squared distance below which an input point counts as present
        in the target."""
        if sqr_threshold < 0:
            raise ValueError("the squared threshold cannot be negative")
        self.me.setDistanceThreshold(sqr_threshold)

    def get_DistanceThreshold(self):
        return self.me.getDistanceThreshold()

    def segment(self):
        """Return the input points absent from the target as a
        :class:`PointCloud`."""
        if not self.has_target:
            raise RuntimeError(
                "set_TargetCloud(cloud) is required before segment()")
        cdef PointCloud pc = PointCloud()
        cdef cPointCloud[PointXYZ]* out = pc.ptr()
        with nogil:
            self.me.segment(deref(out))
        return pc


cdef class ApproximateProgressiveMorphologicalFilter:
    """Ground extraction on a height grid
    (pcl::ApproximateProgressiveMorphologicalFilter).

    `ProgressiveMorphologicalFilter` with the same settings, run on a
    grid of cell size `set_CellSize` instead of per point: much faster,
    slightly coarser. `extract()` returns the ground point indices.
    """

    cdef cApproximateProgressiveMorphologicalFilter[PointXYZ]* me

    def __cinit__(self, PointCloud pc=None):
        self.me = new cApproximateProgressiveMorphologicalFilter[PointXYZ]()
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

    def get_Base(self):
        return self.me.getBase()

    def set_Exponential(self, bint exponential):
        self.me.setExponential(exponential)

    def get_Exponential(self):
        return self.me.getExponential()

    def set_NumberOfThreads(self, unsigned int threads):
        self.me.setNumberOfThreads(threads)

    def extract(self):
        """Return the indices of the ground points."""
        cdef vector[int] ground
        with nogil:
            self.me.extract(ground)
        cdef list result = []
        cdef size_t i
        for i in range(ground.size()):
            result.append(ground[i])
        return result


cdef class SeededHueSegmentation:
    """Grow regions of one hue from seed points
    (pcl::SeededHueSegmentation) on a :class:`PointCloud_PointXYZRGB`.

    Starting from the seed indices, neighbours within
    `set_ClusterTolerance` whose hue differs by less than `set_DeltaHue`
    (degrees) join; `segment(seeds)` returns the grown index list.
    """

    cdef cSeededHueSegmentation* me
    cdef PointCloud_PointXYZRGB _input

    def __cinit__(self, PointCloud_PointXYZRGB pc=None):
        self.me = new cSeededHueSegmentation()
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGB pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc

    def set_ClusterTolerance(self, double tolerance):
        self.me.setClusterTolerance(tolerance)

    def get_ClusterTolerance(self):
        return self.me.getClusterTolerance()

    def set_DeltaHue(self, float delta_hue):
        """Largest hue difference (degrees, 0..360) that still joins."""
        self.me.setDeltaHue(delta_hue)

    def get_DeltaHue(self):
        return self.me.getDeltaHue()

    def segment(self, seeds):
        """Return the indices grown from the seed indices."""
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        cdef PointIndices seed_indices
        cdef Py_ssize_t n = self._input.size
        for i in seeds:
            if not 0 <= i < n:
                raise IndexError(
                    "seed index %d out of range for a cloud of %d points"
                    % (i, n))
            seed_indices.indices.push_back(<int> i)
        if seed_indices.indices.empty():
            raise ValueError("at least one seed index is required")
        cdef PointIndices output
        with nogil:
            self.me.segment(seed_indices, output)
        return _indices_to_list(output)


cdef class OrganizedMultiPlaneSegmentation:
    """Every plane of an organized cloud at once
    (pcl::OrganizedMultiPlaneSegmentation).

    Needs an ORGANIZED cloud (a sensor's width x height grid) and its
    normals -- `IntegralImageNormalEstimation.compute_cloud()` is the
    matching source. Connected pixels whose normals agree within
    `set_AngularThreshold` (radians) and whose plane distance agrees
    within `set_DistanceThreshold` form a plane; `segment()` returns
    ``(coefficients, inliers)``: one ``[a, b, c, d]`` and one index list
    per plane with at least `set_MinInliers` points.
    """

    cdef cOrganizedMultiPlaneSegmentation[PointXYZ, Normal, Label]* me
    cdef PointCloud _input
    cdef bint has_normals

    def __cinit__(self, PointCloud pc=None):
        self.me = new cOrganizedMultiPlaneSegmentation[PointXYZ, Normal, Label]()
        self.has_normals = False
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc

    def set_InputNormals(self, PointCloud_Normal normals not None):
        self.me.setInputNormals(normals.thisptr_shared)
        self.has_normals = True

    def set_MinInliers(self, unsigned int count):
        self.me.setMinInliers(count)

    def get_MinInliers(self):
        return self.me.getMinInliers()

    def set_AngularThreshold(self, double angle):
        self.me.setAngularThreshold(angle)

    def get_AngularThreshold(self):
        return self.me.getAngularThreshold()

    def set_DistanceThreshold(self, double distance):
        self.me.setDistanceThreshold(distance)

    def get_DistanceThreshold(self):
        return self.me.getDistanceThreshold()

    def set_MaximumCurvature(self, double curvature):
        self.me.setMaximumCurvature(curvature)

    def get_MaximumCurvature(self):
        return self.me.getMaximumCurvature()

    def set_ProjectPoints(self, bint project):
        self.me.setProjectPoints(project)

    def segment(self):
        """Return ``(coefficients, inliers)``, one entry each per plane."""
        if self._input is None:
            raise RuntimeError("no input cloud: call set_InputCloud() first")
        if self._input.height <= 1:
            raise ValueError(
                "OrganizedMultiPlaneSegmentation needs an organized cloud "
                "(height > 1); this one is %d x %d"
                % (self._input.width, self._input.height))
        if not self.has_normals:
            raise RuntimeError(
                "set_InputNormals() is required: use "
                "IntegralImageNormalEstimation.compute_cloud()")
        cdef vector[ModelCoefficients] coefficients
        cdef vector[PointIndices] inliers
        with nogil:
            self.me.segment(coefficients, inliers)
        cdef list planes = []
        cdef list one
        cdef size_t i, j
        for i in range(coefficients.size()):
            one = []
            for j in range(coefficients[i].values.size()):
                one.append(coefficients[i].values[j])
            planes.append(one)
        return planes, _clusters_to_lists(inliers)


cdef class SupervoxelClustering:
    """Supervoxels (pcl::SupervoxelClustering) on a :class:`PointCloud_PointXYZRGBA`.

    The cloud is split into compact, roughly equal "supervoxels" that
    respect colour, position and normal continuity (weighted by
    `set_ColorImportance` / `set_SpatialImportance` /
    `set_NormalImportance`): `voxel_resolution` is the voxel size the
    cloud is first binned at, `seed_resolution` the supervoxel size.
    `extract()` returns the supervoxels; `get_LabeledCloud()` the
    supervoxel label of every input point; `lccp()` merges adjacent
    supervoxels across convex boundaries into object segments
    (pcl::LCCPSegmentation).

    For data that is not a single depth frame, call
    `set_UseSingleCameraTransform(False)` first: PCL's default assumes a
    camera at the origin and scales the voxels with depth.
    """

    cdef cSupervoxelClustering[PointXYZRGBA]* me
    cdef SupervoxelSet[PointXYZRGBA]* result
    cdef bint extracted
    cdef float voxel_resolution
    cdef float seed_resolution
    cdef PointCloud_PointXYZRGBA _input

    def __cinit__(self, float voxel_resolution, float seed_resolution,
                  PointCloud_PointXYZRGBA pc=None):
        if voxel_resolution <= 0 or seed_resolution <= 0:
            raise ValueError("resolutions must be positive")
        self.me = new cSupervoxelClustering[PointXYZRGBA](
            voxel_resolution, seed_resolution)
        self.result = new SupervoxelSet[PointXYZRGBA]()
        self.extracted = False
        self.voxel_resolution = voxel_resolution
        self.seed_resolution = seed_resolution
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.result
        self.result = NULL
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud_PointXYZRGBA pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc
        self.extracted = False

    def set_NormalCloud(self, PointCloud_Normal normals not None):
        """Normals to use instead of estimating them."""
        self.me.setNormalCloud(normals.thisptr_shared)

    def set_VoxelResolution(self, float resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.me.setVoxelResolution(resolution)
        self.voxel_resolution = resolution

    def get_VoxelResolution(self):
        return self.me.getVoxelResolution()

    def set_SeedResolution(self, float resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.me.setSeedResolution(resolution)
        self.seed_resolution = resolution

    def get_SeedResolution(self):
        return self.me.getSeedResolution()

    def set_ColorImportance(self, float value):
        self.me.setColorImportance(value)

    def set_SpatialImportance(self, float value):
        self.me.setSpatialImportance(value)

    def set_NormalImportance(self, float value):
        self.me.setNormalImportance(value)

    def set_UseSingleCameraTransform(self, bint value):
        """Scale voxels with depth as for one depth-camera frame (PCL's
        default); switch off for any other data."""
        self.me.setUseSingleCameraTransform(value)

    def get_MaxLabel(self):
        return self.me.getMaxLabel()

    def extract(self):
        """Return ``(labels, centroids, normals, adjacency)``: the
        supervoxel labels (``(k,)`` uint32), their centroids ``(k, 3)``,
        their normals ``(k, 4)`` and the adjacent label pairs
        ``(m, 2)``."""
        import numpy as np
        if self._input is None or self._input.size == 0:
            raise RuntimeError("no input cloud, or an empty one")
        cdef vector[unsigned int] labels
        cdef vector[float] centroids
        cdef vector[float] normals
        cdef vector[unsigned int] adjacency
        with nogil:
            supervoxelExtract[PointXYZRGBA](deref(self.me), deref(self.result))
            supervoxelSummary[PointXYZRGBA](deref(self.result), labels, centroids,
                                       normals, adjacency)
        self.extracted = True
        cdef Py_ssize_t k = <Py_ssize_t> labels.size()
        label_arr = np.empty(k, dtype=np.uint32)
        centroid_arr = np.empty((k, 3), dtype=np.float32)
        normal_arr = np.empty((k, 4), dtype=np.float32)
        cdef unsigned int[::1] lv = label_arr
        cdef float[:, ::1] cv = centroid_arr
        cdef float[:, ::1] nv = normal_arr
        cdef Py_ssize_t i
        for i in range(k):
            lv[i] = labels[i]
            cv[i, 0] = centroids[3 * i]
            cv[i, 1] = centroids[3 * i + 1]
            cv[i, 2] = centroids[3 * i + 2]
            nv[i, 0] = normals[4 * i]
            nv[i, 1] = normals[4 * i + 1]
            nv[i, 2] = normals[4 * i + 2]
            nv[i, 3] = normals[4 * i + 3]
        cdef Py_ssize_t m = <Py_ssize_t> adjacency.size() // 2
        adjacency_arr = np.empty((m, 2), dtype=np.uint32)
        cdef unsigned int[:, ::1] av = adjacency_arr
        for i in range(m):
            av[i, 0] = adjacency[2 * i]
            av[i, 1] = adjacency[2 * i + 1]
        return label_arr, centroid_arr, normal_arr, adjacency_arr

    def get_LabeledCloud(self):
        """Return the supervoxel label of every input point as a
        ``(n,)`` uint32 array (0 for points in no supervoxel). Only
        available after `extract()`."""
        import numpy as np
        if not self.extracted:
            raise RuntimeError("call extract() before get_LabeledCloud()")
        cdef vector[unsigned int] labels
        with nogil:
            supervoxelPointLabels[PointXYZRGBA](deref(self.me), labels)
        cdef Py_ssize_t n = <Py_ssize_t> labels.size()
        result = np.empty(n, dtype=np.uint32)
        cdef unsigned int[::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = labels[i]
        return result

    def lccp(self, float concavity_tolerance=10.0,
             bint smoothness_check=True, float smoothness_threshold=0.1,
             bint sanity_check=False, unsigned int k_factor=0,
             unsigned int min_segment_size=0):
        """Merge the supervoxels into object segments along convex
        boundaries (pcl::LCCPSegmentation). Returns ``(point_labels,
        supervoxel_to_segment)``: the segment label of every input point
        as a ``(n,)`` uint32 array, and a dict from supervoxel label to
        segment label. `concavity_tolerance` is in degrees. Only
        available after `extract()`."""
        import numpy as np
        if not self.extracted:
            raise RuntimeError("call extract() before lccp()")
        cdef vector[unsigned int] point_labels
        cdef vector[unsigned int] mapping
        with nogil:
            lccpSegment[PointXYZRGBA](
                deref(self.result), deref(self.me), concavity_tolerance,
                smoothness_check, self.voxel_resolution,
                self.seed_resolution, smoothness_threshold, sanity_check,
                k_factor, min_segment_size, point_labels, mapping)
        cdef Py_ssize_t n = <Py_ssize_t> point_labels.size()
        result = np.empty(n, dtype=np.uint32)
        cdef unsigned int[::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = point_labels[i]
        cdef dict segments = {}
        for i in range(<Py_ssize_t> mapping.size() // 2):
            segments[mapping[2 * i]] = mapping[2 * i + 1]
        return result, segments


cdef class SupervoxelClusteringXYZ:
    """Supervoxels (pcl::SupervoxelClustering) on a plain :class:`PointCloud` (no colour term).

    The cloud is split into compact, roughly equal "supervoxels" that
    respect colour, position and normal continuity (weighted by
    `set_ColorImportance` / `set_SpatialImportance` /
    `set_NormalImportance`): `voxel_resolution` is the voxel size the
    cloud is first binned at, `seed_resolution` the supervoxel size.
    `extract()` returns the supervoxels; `get_LabeledCloud()` the
    supervoxel label of every input point; `lccp()` merges adjacent
    supervoxels across convex boundaries into object segments
    (pcl::LCCPSegmentation).

    For data that is not a single depth frame, call
    `set_UseSingleCameraTransform(False)` first: PCL's default assumes a
    camera at the origin and scales the voxels with depth.
    """

    cdef cSupervoxelClustering[PointXYZ]* me
    cdef SupervoxelSet[PointXYZ]* result
    cdef bint extracted
    cdef float voxel_resolution
    cdef float seed_resolution
    cdef PointCloud _input

    def __cinit__(self, float voxel_resolution, float seed_resolution,
                  PointCloud pc=None):
        if voxel_resolution <= 0 or seed_resolution <= 0:
            raise ValueError("resolutions must be positive")
        self.me = new cSupervoxelClustering[PointXYZ](
            voxel_resolution, seed_resolution)
        self.result = new SupervoxelSet[PointXYZ]()
        self.extracted = False
        self.voxel_resolution = voxel_resolution
        self.seed_resolution = seed_resolution
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.result
        self.result = NULL
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)
        self._input = pc
        self.extracted = False

    def set_NormalCloud(self, PointCloud_Normal normals not None):
        """Normals to use instead of estimating them."""
        self.me.setNormalCloud(normals.thisptr_shared)

    def set_VoxelResolution(self, float resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.me.setVoxelResolution(resolution)
        self.voxel_resolution = resolution

    def get_VoxelResolution(self):
        return self.me.getVoxelResolution()

    def set_SeedResolution(self, float resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.me.setSeedResolution(resolution)
        self.seed_resolution = resolution

    def get_SeedResolution(self):
        return self.me.getSeedResolution()

    def set_ColorImportance(self, float value):
        self.me.setColorImportance(value)

    def set_SpatialImportance(self, float value):
        self.me.setSpatialImportance(value)

    def set_NormalImportance(self, float value):
        self.me.setNormalImportance(value)

    def set_UseSingleCameraTransform(self, bint value):
        """Scale voxels with depth as for one depth-camera frame (PCL's
        default); switch off for any other data."""
        self.me.setUseSingleCameraTransform(value)

    def get_MaxLabel(self):
        return self.me.getMaxLabel()

    def extract(self):
        """Return ``(labels, centroids, normals, adjacency)``: the
        supervoxel labels (``(k,)`` uint32), their centroids ``(k, 3)``,
        their normals ``(k, 4)`` and the adjacent label pairs
        ``(m, 2)``."""
        import numpy as np
        if self._input is None or self._input.size == 0:
            raise RuntimeError("no input cloud, or an empty one")
        cdef vector[unsigned int] labels
        cdef vector[float] centroids
        cdef vector[float] normals
        cdef vector[unsigned int] adjacency
        with nogil:
            supervoxelExtract[PointXYZ](deref(self.me), deref(self.result))
            supervoxelSummary[PointXYZ](deref(self.result), labels, centroids,
                                       normals, adjacency)
        self.extracted = True
        cdef Py_ssize_t k = <Py_ssize_t> labels.size()
        label_arr = np.empty(k, dtype=np.uint32)
        centroid_arr = np.empty((k, 3), dtype=np.float32)
        normal_arr = np.empty((k, 4), dtype=np.float32)
        cdef unsigned int[::1] lv = label_arr
        cdef float[:, ::1] cv = centroid_arr
        cdef float[:, ::1] nv = normal_arr
        cdef Py_ssize_t i
        for i in range(k):
            lv[i] = labels[i]
            cv[i, 0] = centroids[3 * i]
            cv[i, 1] = centroids[3 * i + 1]
            cv[i, 2] = centroids[3 * i + 2]
            nv[i, 0] = normals[4 * i]
            nv[i, 1] = normals[4 * i + 1]
            nv[i, 2] = normals[4 * i + 2]
            nv[i, 3] = normals[4 * i + 3]
        cdef Py_ssize_t m = <Py_ssize_t> adjacency.size() // 2
        adjacency_arr = np.empty((m, 2), dtype=np.uint32)
        cdef unsigned int[:, ::1] av = adjacency_arr
        for i in range(m):
            av[i, 0] = adjacency[2 * i]
            av[i, 1] = adjacency[2 * i + 1]
        return label_arr, centroid_arr, normal_arr, adjacency_arr

    def get_LabeledCloud(self):
        """Return the supervoxel label of every input point as a
        ``(n,)`` uint32 array (0 for points in no supervoxel). Only
        available after `extract()`."""
        import numpy as np
        if not self.extracted:
            raise RuntimeError("call extract() before get_LabeledCloud()")
        cdef vector[unsigned int] labels
        with nogil:
            supervoxelPointLabels[PointXYZ](deref(self.me), labels)
        cdef Py_ssize_t n = <Py_ssize_t> labels.size()
        result = np.empty(n, dtype=np.uint32)
        cdef unsigned int[::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = labels[i]
        return result

    def lccp(self, float concavity_tolerance=10.0,
             bint smoothness_check=True, float smoothness_threshold=0.1,
             bint sanity_check=False, unsigned int k_factor=0,
             unsigned int min_segment_size=0):
        """Merge the supervoxels into object segments along convex
        boundaries (pcl::LCCPSegmentation). Returns ``(point_labels,
        supervoxel_to_segment)``: the segment label of every input point
        as a ``(n,)`` uint32 array, and a dict from supervoxel label to
        segment label. `concavity_tolerance` is in degrees. Only
        available after `extract()`."""
        import numpy as np
        if not self.extracted:
            raise RuntimeError("call extract() before lccp()")
        cdef vector[unsigned int] point_labels
        cdef vector[unsigned int] mapping
        with nogil:
            lccpSegment[PointXYZ](
                deref(self.result), deref(self.me), concavity_tolerance,
                smoothness_check, self.voxel_resolution,
                self.seed_resolution, smoothness_threshold, sanity_check,
                k_factor, min_segment_size, point_labels, mapping)
        cdef Py_ssize_t n = <Py_ssize_t> point_labels.size()
        result = np.empty(n, dtype=np.uint32)
        cdef unsigned int[::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = point_labels[i]
        cdef dict segments = {}
        for i in range(<Py_ssize_t> mapping.size() // 2):
            segments[mapping[2 * i]] = mapping[2 * i + 1]
        return result, segments


cdef bint _call_condition(object predicate, const PointXYZ& a,
                          const PointXYZ& b, float sqr_distance) noexcept:
    """Run the Python predicate, swallowing anything it raises.

    An exception escaping here would unwind out of a `noexcept` C
    function called from inside PCL's clustering loop: std::terminate,
    i.e. the interpreter dies with no traceback. A raising predicate is
    treated as "not the same cluster" and the failure is printed.
    """
    try:
        return bool(predicate((a.x, a.y, a.z), (b.x, b.y, b.z),
                              sqr_distance))
    except BaseException:
        traceback.print_exc()
        return False


cdef bint _condition_trampoline(const PointXYZ& a, const PointXYZ& b,
                                float sqr_distance,
                                void* user_data) noexcept nogil:
    """The C function pointer PCL ends up calling, once per neighbour."""
    cdef bint keep
    with gil:
        keep = _call_condition(<object> user_data, a, b, sqr_distance)
    return keep


cdef class ConditionalEuclideanClustering:
    """Euclidean clustering with a caller-supplied join test
    (pcl::ConditionalEuclideanClustering).

    The predicate decides whether two neighbouring points belong to the
    same cluster:

        def same_cluster(a, b, squared_distance):
            return abs(a[2] - b[2]) < 0.05

        cec = cloud.make_ConditionalEuclideanClustering()
        cec.set_ClusterTolerance(0.1)
        cec.set_ConditionFunction(same_cluster)
        clusters = cec.segment()

    PCL calls it once per candidate neighbour pair, so it is the slow
    path by construction — see bench/README.md. `EuclideanClusterExtraction`
    keeps the whole loop in C++ when a plain distance test will do.
    """

    cdef cConditionalEuclideanClustering[PointXYZ]* me
    # Holds the predicate alive for as long as PCL may call it; the shim
    # only has a borrowed void*.
    cdef object condition

    def __cinit__(self, PointCloud pc=None):
        self.me = new cConditionalEuclideanClustering[PointXYZ]()
        self.condition = None
        if pc is not None:
            self.set_InputCloud(pc)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputCloud(self, PointCloud pc not None):
        self.me.setInputCloud(pc.thisptr_shared)

    def set_ConditionFunction(self, predicate):
        """``predicate(a, b, squared_distance) -> bool``, where *a* and
        *b* are ``(x, y, z)`` tuples. Required."""
        if not callable(predicate):
            raise TypeError("condition must be callable")
        self.condition = predicate
        setClusterCondition(deref(self.me),
                            <ClusterConditionFn> _condition_trampoline,
                            <void*> self.condition)

    def set_ClusterTolerance(self, float tolerance):
        self.me.setClusterTolerance(tolerance)

    def get_ClusterTolerance(self):
        return self.me.getClusterTolerance()

    def set_MinClusterSize(self, int min_size):
        self.me.setMinClusterSize(min_size)

    def get_MinClusterSize(self):
        return self.me.getMinClusterSize()

    def set_MaxClusterSize(self, int max_size):
        self.me.setMaxClusterSize(max_size)

    def get_MaxClusterSize(self):
        return self.me.getMaxClusterSize()

    def segment(self):
        """Return the clusters as a list of index lists."""
        if self.condition is None:
            raise RuntimeError(
                "set_ConditionFunction() is required: PCL calls an unset "
                "condition and crashes")
        cdef vector[PointIndices] clusters
        with nogil:
            self.me.segment(clusters)
        return [
            [clusters[i].indices[j] for j in range(clusters[i].indices.size())]
            for i in range(clusters.size())
        ]
