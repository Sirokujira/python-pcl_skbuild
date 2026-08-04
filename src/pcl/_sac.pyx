# distutils: language = c++
# cython: language_level=3
"""RANSAC wrappers, named after sirokujira/python-pcl
(pcl/pxi/SampleConsensus/).

    model = pcl.SampleConsensusModelPlane(cloud)
    ransac = pcl.RandomSampleConsensus(model)
    ransac.set_DistanceThreshold(0.01)
    ransac.computeModel()
    inliers = ransac.get_Inliers()

This is the lower-level counterpart of `Segmentation`: same models and
same fitting, but you drive the loop and keep the model object. Reach for
`cloud.make_segmenter()` unless you need that.

Everything about PCL's RANSAC API is something Cython cannot express on
its own — the constructor takes a base-class shared_ptr the concrete
models must be converted to, and the coefficients come back in an
`Eigen::VectorXf` — so it all goes through pcl/compat/sac_models.h.
"""

from cython.operator cimport dereference as deref

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.sample_consensus.sac_model cimport SampleConsensusModel as cSacModel
from pcl.pxd.sample_consensus.ransac cimport (
    RandomSampleConsensus as cRansac)
from pcl.pxd.compat.sac_models cimport (
    makeRansac, makeSacModel, ransacCoefficients, ransacComputeModel,
    ransacInliers, sacModelIsNull)
cimport pcl.pxd.sample_consensus.model_types as model_types

from pcl._pointcloud cimport PointCloud


cdef class SampleConsensusModel:
    """A model RANSAC can fit to a cloud.

    Construct it with one of the ``pcl.SACMODEL_*`` constants. The
    ``SampleConsensusModel*`` helpers below name the model instead, which
    is how python-pcl spells it — they are functions rather than
    subclasses because the model type has to be known before PCL builds
    the object, and a subclass cannot supply it before its base
    ``__cinit__`` runs.
    """

    cdef shared_ptr[cSacModel[PointXYZ]] thisptr
    cdef readonly int model_type

    def __cinit__(self, PointCloud pc not None, int model_type):
        self.model_type = model_type
        self.thisptr = makeSacModel(model_type, pc.thisptr_shared)
        if sacModelIsNull(self.thisptr):
            raise ValueError(
                "model type %d is not wrapped; available: SACMODEL_PLANE, "
                "SACMODEL_LINE, SACMODEL_CIRCLE2D, SACMODEL_CIRCLE3D, "
                "SACMODEL_SPHERE, SACMODEL_STICK" % model_type)


def SampleConsensusModelPlane(PointCloud pc not None):
    """pcl::SampleConsensusModelPlane."""
    return SampleConsensusModel(pc, model_types.SACMODEL_PLANE)


def SampleConsensusModelLine(PointCloud pc not None):
    """pcl::SampleConsensusModelLine."""
    return SampleConsensusModel(pc, model_types.SACMODEL_LINE)


def SampleConsensusModelCircle2D(PointCloud pc not None):
    """pcl::SampleConsensusModelCircle2D."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CIRCLE2D)


def SampleConsensusModelCircle3D(PointCloud pc not None):
    """pcl::SampleConsensusModelCircle3D."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CIRCLE3D)


def SampleConsensusModelSphere(PointCloud pc not None):
    """pcl::SampleConsensusModelSphere."""
    return SampleConsensusModel(pc, model_types.SACMODEL_SPHERE)


def SampleConsensusModelStick(PointCloud pc not None):
    """pcl::SampleConsensusModelStick."""
    return SampleConsensusModel(pc, model_types.SACMODEL_STICK)


cdef class RandomSampleConsensus:
    """Fit a :class:`SampleConsensusModel` with RANSAC
    (pcl::RandomSampleConsensus)."""

    cdef shared_ptr[cRansac[PointXYZ]] thisptr
    # Keeps the model alive: PCL holds it by shared_ptr, but the Python
    # object owning that handle must outlive this one too.
    cdef object model

    def __cinit__(self, SampleConsensusModel model not None):
        self.model = model
        self.thisptr = makeRansac(model.thisptr)

    def set_DistanceThreshold(self, double threshold):
        """How far a point may sit from the model and still be an inlier."""
        deref(self.thisptr).setDistanceThreshold(threshold)

    def get_DistanceThreshold(self):
        return deref(self.thisptr).getDistanceThreshold()

    def set_MaxIterations(self, int max_iterations):
        deref(self.thisptr).setMaxIterations(max_iterations)

    def get_MaxIterations(self):
        return deref(self.thisptr).getMaxIterations()

    def set_Probability(self, double probability):
        deref(self.thisptr).setProbability(probability)

    def get_Probability(self):
        return deref(self.thisptr).getProbability()

    def computeModel(self):
        """Run RANSAC. Returns True when a model was found."""
        cdef cRansac[PointXYZ]* r = self.thisptr.get()
        cdef bint found
        with nogil:
            found = ransacComputeModel(deref(r))
        return found

    def get_Inliers(self):
        """Indices of the points the fitted model explains."""
        cdef vector[int] inliers
        ransacInliers(deref(self.thisptr), inliers)
        return [inliers[i] for i in range(inliers.size())]

    def get_ModelCoefficients(self):
        """The fitted model's coefficients, as a list of floats."""
        cdef vector[float] coefficients
        ransacCoefficients(deref(self.thisptr), coefficients)
        return [coefficients[i] for i in range(coefficients.size())]
