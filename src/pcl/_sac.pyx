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
from pcl.pxd.sample_consensus.sac cimport SampleConsensus as cSac
from pcl.pxd.compat.sac_models cimport (
    makeSac, makeSacModel, sacCoefficients, sacComputeModel, sacInliers,
    sacIsNull, sacModelIsNull, sacModelSetAxis, sacModelSetConeOpeningAngle,
    sacModelSetEpsAngle, sacModelSetNormals, sacModelSetRadiusLimits,
    sacModelSetTarget, SAC_MODEL_ELLIPSE3D)
cimport pcl.pxd.sample_consensus.model_types as model_types
cimport pcl.pxd.sample_consensus.method_types as method_types

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport PointCloud_Normal


cdef class SampleConsensusModel:
    """A model sample consensus can fit to a cloud.

    Construct it with one of the ``pcl.SACMODEL_*`` constants. The
    ``SampleConsensusModel*`` helpers below name the model instead, which
    is how python-pcl spells it — they are functions rather than
    subclasses because the model type has to be known before PCL builds
    the object, and a subclass cannot supply it before its base
    ``__cinit__`` runs.

    Model families and what they need:

    - plane, line, circle (2-D / 3-D), sphere, stick: nothing more
    - cylinder, cone, normal plane, normal sphere, normal parallel plane:
      `set_InputNormals`
    - parallel / perpendicular plane, parallel line, normal parallel
      plane, cylinder, cone: `set_Axis` and `set_EpsAngle`
    - registration: `set_InputTarget`
    - anything with a radius: `set_RadiusLimits` narrows the search
    """

    cdef shared_ptr[cSacModel[PointXYZ]] thisptr
    cdef readonly int model_type
    # The cloud, normals and target are held by PCL as shared_ptr; the
    # Python objects owning those handles must outlive this one too.
    cdef object _cloud
    cdef object _normals
    cdef object _target

    def __cinit__(self, PointCloud pc not None, int model_type):
        self.model_type = model_type
        self._cloud = pc
        if model_type in (model_types.SACMODEL_TORUS,
                          model_types.SACMODEL_PARALLEL_LINES):
            raise ValueError(
                "model type %d is declared in PCL's SacModel enum but has no "
                "implementation in PCL" % model_type)
        if model_type == model_types.SACMODEL_REGISTRATION_2D:
            raise ValueError(
                "SACMODEL_REGISTRATION_2D reads .u and .v from points that "
                "also need x, y, z -- a point type PCL does not define")
        self.thisptr = makeSacModel(model_type, pc.thisptr_shared)
        if sacModelIsNull(self.thisptr):
            raise ValueError(
                "model type %d is not wrapped; available: SACMODEL_PLANE, "
                "SACMODEL_LINE, SACMODEL_CIRCLE2D, SACMODEL_CIRCLE3D, "
                "SACMODEL_SPHERE, SACMODEL_STICK, SACMODEL_CYLINDER, "
                "SACMODEL_CONE, SACMODEL_NORMAL_PLANE, SACMODEL_NORMAL_SPHERE, "
                "SACMODEL_NORMAL_PARALLEL_PLANE, SACMODEL_PARALLEL_PLANE, "
                "SACMODEL_PERPENDICULAR_PLANE, SACMODEL_PARALLEL_LINE, "
                "SACMODEL_REGISTRATION, SACMODEL_ELLIPSE3D" % model_type)

    def set_InputNormals(self, PointCloud_Normal normals not None,
                         double distance_weight=0.1):
        """Normals for the normal-based models, and how much they weigh
        against distance (0..1)."""
        if not 0.0 <= distance_weight <= 1.0:
            raise ValueError("distance_weight must be in [0, 1]")
        if not sacModelSetNormals(self.thisptr, normals.thisptr_shared,
                                  distance_weight):
            raise TypeError(
                "model type %d does not take normals" % self.model_type)
        self._normals = normals

    def set_Axis(self, float x, float y, float z):
        """The axis the model is constrained to (parallel to it for the
        parallel models and the cylinder / cone, perpendicular for
        SACMODEL_PERPENDICULAR_PLANE)."""
        if not sacModelSetAxis(self.thisptr, self.model_type, x, y, z):
            raise TypeError(
                "model type %d has no axis constraint" % self.model_type)

    def set_EpsAngle(self, double angle):
        """How far (radians) the fit may deviate from `set_Axis`."""
        if not sacModelSetEpsAngle(self.thisptr, self.model_type, angle):
            raise TypeError(
                "model type %d has no axis constraint" % self.model_type)

    def set_RadiusLimits(self, double min_radius, double max_radius):
        """Reject fits whose radius falls outside the range."""
        if min_radius < 0 or max_radius < min_radius:
            raise ValueError("need 0 <= min_radius <= max_radius")
        sacModelSetRadiusLimits(self.thisptr, min_radius, max_radius)

    def set_MinMaxOpeningAngle(self, double min_angle, double max_angle):
        """Allowed cone opening angles, in radians (cone only)."""
        if not sacModelSetConeOpeningAngle(self.thisptr, min_angle,
                                           max_angle):
            raise TypeError(
                "model type %d is not a cone" % self.model_type)

    def set_InputTarget(self, PointCloud target not None):
        """The cloud the registration model aligns the input to, point
        for point (registration only)."""
        if not sacModelSetTarget(self.thisptr, target.thisptr_shared):
            raise TypeError(
                "model type %d is not the registration model"
                % self.model_type)
        self._target = target


def SampleConsensusModelPlane(PointCloud pc not None):
    """A plane: coefficients ``a, b, c, d``."""
    return SampleConsensusModel(pc, model_types.SACMODEL_PLANE)


def SampleConsensusModelLine(PointCloud pc not None):
    """A line: a point on it and its direction."""
    return SampleConsensusModel(pc, model_types.SACMODEL_LINE)


def SampleConsensusModelCircle2D(PointCloud pc not None):
    """A circle in the x/y plane: centre and radius."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CIRCLE2D)


def SampleConsensusModelCircle3D(PointCloud pc not None):
    """A circle in space: centre, radius and normal."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CIRCLE3D)


def SampleConsensusModelSphere(PointCloud pc not None):
    """A sphere: centre and radius."""
    return SampleConsensusModel(pc, model_types.SACMODEL_SPHERE)


def SampleConsensusModelStick(PointCloud pc not None):
    """A stick (a line with a width)."""
    return SampleConsensusModel(pc, model_types.SACMODEL_STICK)


def SampleConsensusModelCylinder(PointCloud pc not None):
    """A cylinder: a point on the axis, the axis direction and the
    radius. Needs `set_InputNormals`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CYLINDER)


def SampleConsensusModelCone(PointCloud pc not None):
    """A cone: apex, axis direction and opening angle. Needs
    `set_InputNormals`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_CONE)


def SampleConsensusModelNormalPlane(PointCloud pc not None):
    """A plane whose inliers must also have a matching normal. Needs
    `set_InputNormals`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_NORMAL_PLANE)


def SampleConsensusModelNormalSphere(PointCloud pc not None):
    """A sphere whose inliers must also have a matching normal. Needs
    `set_InputNormals`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_NORMAL_SPHERE)


def SampleConsensusModelParallelPlane(PointCloud pc not None):
    """A plane parallel to `set_Axis` within `set_EpsAngle`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_PARALLEL_PLANE)


def SampleConsensusModelPerpendicularPlane(PointCloud pc not None):
    """A plane perpendicular to `set_Axis` within `set_EpsAngle`."""
    return SampleConsensusModel(
        pc, model_types.SACMODEL_PERPENDICULAR_PLANE)


def SampleConsensusModelNormalParallelPlane(PointCloud pc not None):
    """A normal plane that is also parallel to `set_Axis`. Needs
    `set_InputNormals`."""
    return SampleConsensusModel(
        pc, model_types.SACMODEL_NORMAL_PARALLEL_PLANE)


def SampleConsensusModelParallelLine(PointCloud pc not None):
    """A line parallel to `set_Axis` within `set_EpsAngle`."""
    return SampleConsensusModel(pc, model_types.SACMODEL_PARALLEL_LINE)


def SampleConsensusModelEllipse3D(PointCloud pc not None):
    """A 3-D ellipse (pcl::SampleConsensusModelEllipse3D): 11 coefficients
    -- centre (3), semi-axes a and b (2), plane normal (3) and the
    direction of the a axis (3). Needs PCL 1.13 or newer."""
    if SAC_MODEL_ELLIPSE3D < 0:
        raise ValueError("SampleConsensusModelEllipse3D needs PCL 1.13+")
    return SampleConsensusModel(pc, SAC_MODEL_ELLIPSE3D)


def SampleConsensusModelRegistration(PointCloud pc not None):
    """A rigid transform taking the input onto `set_InputTarget`, point
    i to point i; the coefficients are the 4x4 matrix, row-major."""
    return SampleConsensusModel(pc, model_types.SACMODEL_REGISTRATION)


cdef class SampleConsensus:
    """Fit a :class:`SampleConsensusModel` with one of PCL's sample
    consensus methods (pcl::SampleConsensus and its subclasses).

    `method` is one of the ``pcl.SAC_*`` constants: RANSAC (the default),
    LMEDS, MSAC, RMSAC, MLESAC or PROSAC. All share the distance
    threshold, iteration cap and probability.

    SAC_RRANSAC is refused: PCL's RandomizedRandomSampleConsensus never
    returns on data with outliers (its pre-test rejects a candidate as
    soon as one point of a random 10% subset is an outlier, and the loop
    bound is only lowered once a candidate passes, so `max_iterations`
    is never consulted). SAC_RMSAC shares the pre-test but does stop;
    expect a poor model unless the data is nearly outlier-free.
    """

    cdef shared_ptr[cSac[PointXYZ]] thisptr
    # Keeps the model alive: PCL holds it by shared_ptr, but the Python
    # object owning that handle must outlive this one too.
    cdef object model
    cdef readonly int method

    def __cinit__(self, SampleConsensusModel model not None,
                  int method=method_types.SAC_RANSAC):
        self.model = model
        self.method = method
        if method == method_types.SAC_RRANSAC:
            raise ValueError(
                "SAC_RRANSAC is refused: PCL's RandomizedRandomSampleConsensus "
                "never returns on data with outliers (see the class docstring); "
                "use SAC_RANSAC, or SAC_RMSAC on nearly clean data")
        self.thisptr = makeSac(method, model.thisptr)
        if sacIsNull(self.thisptr):
            raise ValueError(
                "method %d is not a SAC_* constant (SAC_RANSAC, SAC_LMEDS, "
                "SAC_MSAC, SAC_RRANSAC, SAC_RMSAC, SAC_MLESAC, SAC_PROSAC)"
                % method)

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

    def set_NumberOfThreads(self, int threads):
        """OpenMP threads for the inlier count (-1: automatic)."""
        deref(self.thisptr).setNumberOfThreads(threads)

    def get_NumberOfThreads(self):
        return deref(self.thisptr).getNumberOfThreads()

    def computeModel(self):
        """Run the method. Returns True when a model was found."""
        cdef cSac[PointXYZ]* sac = self.thisptr.get()
        cdef bint found
        with nogil:
            found = sacComputeModel(deref(sac))
        return found

    def get_Inliers(self):
        """Indices of the points the fitted model explains."""
        cdef vector[int] inliers
        sacInliers(deref(self.thisptr), inliers)
        return [inliers[i] for i in range(inliers.size())]

    def get_ModelCoefficients(self):
        """The fitted model's coefficients, as a list of floats."""
        cdef vector[float] coefficients
        sacCoefficients(deref(self.thisptr), coefficients)
        return [coefficients[i] for i in range(coefficients.size())]


class RandomSampleConsensus(SampleConsensus):
    """RANSAC (pcl::RandomSampleConsensus): `SampleConsensus` with
    `pcl.SAC_RANSAC`, under python-pcl's name."""
