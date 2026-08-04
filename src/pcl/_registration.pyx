# distutils: language = c++
# cython: language_level=3
"""Registration wrappers, named after sirokujira/python-pcl
(pcl/pxi/registration/).

Every class exposes the same shape as python-pcl: one call that takes
source and target and returns

    (converged, transform, estimate, fitness)

where *transform* is a 4x4 float32 array in Fortran order (Eigen is
column-major, so that is the layout PCL already has), *estimate* is the
transformed source cloud, and *fitness* is the mean squared distance to
the target.

`NormalDistributionsTransform` is here too, which python-pcl never had —
strawlab/python-pcl#265 asked for it and the repository is archived.

`getFinalTransformation` returns an `Eigen::Matrix4f`, which neither a
mirror header nor Cython can name; pcl/compat/eigen_results.h copies the
16 floats out. Same rung-4 shim pattern as the grabber callback.
"""

from cython.operator cimport dereference as deref

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.registration.icp cimport IterativeClosestPoint as cICP
from pcl.pxd.registration.icp_nl cimport (
    IterativeClosestPointNonLinear as cICPNL)
from pcl.pxd.registration.gicp cimport (
    GeneralizedIterativeClosestPoint as cGICP)
from pcl.pxd.registration.ndt cimport NormalDistributionsTransform as cNDT
from pcl.pxd.compat.eigen_results cimport finalTransformation

from pcl._pointcloud cimport PointCloud


cdef object _as_matrix(float* buf):
    """The 16 floats the shim wrote, as a 4x4 Fortran-ordered array."""
    import numpy as np
    transf = np.empty((4, 4), dtype=np.float32, order="F")
    cdef float[::1] flat = transf.reshape(-1, order="A")
    cdef int i
    for i in range(16):
        flat[i] = buf[i]
    return transf


cdef class IterativeClosestPoint:
    """Point-to-point ICP (pcl::IterativeClosestPoint)."""

    cdef cICP[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cICP[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_EuclideanFitnessEpsilon(self, double epsilon):
        self.me.setEuclideanFitnessEpsilon(epsilon)

    def set_UseReciprocalCorrespondences(self, bint value):
        self.me.setUseReciprocalCorrespondences(value)

    def icp(self, PointCloud source not None, PointCloud target not None,
            max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class IterativeClosestPointNonLinear:
    """ICP with a Levenberg-Marquardt transformation estimator
    (pcl::IterativeClosestPointNonLinear)."""

    cdef cICPNL[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cICPNL[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def icp_nl(self, PointCloud source not None, PointCloud target not None,
               max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class GeneralizedIterativeClosestPoint:
    """Plane-to-plane ICP (pcl::GeneralizedIterativeClosestPoint)."""

    cdef cGICP[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cGICP[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_MaxCorrespondenceDistance(self, double distance):
        self.me.setMaxCorrespondenceDistance(distance)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_RotationEpsilon(self, double epsilon):
        self.me.setRotationEpsilon(epsilon)

    def set_CorrespondenceRandomness(self, int k):
        self.me.setCorrespondenceRandomness(k)

    def set_MaximumOptimizerIterations(self, int count):
        self.me.setMaximumOptimizerIterations(count)

    def gicp(self, PointCloud source not None, PointCloud target not None,
             max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())


cdef class NormalDistributionsTransform:
    """NDT registration (pcl::NormalDistributionsTransform).

    Matches voxelised normal distributions rather than point pairs, which
    makes it the usual choice for large scans where ICP is too slow.
    Unlike the others it needs a voxel `resolution` to be useful; the PCL
    tutorial default is 1.0 with a step size of 0.1.
    """

    cdef cNDT[PointXYZ, PointXYZ]* me

    def __cinit__(self):
        self.me = new cNDT[PointXYZ, PointXYZ]()

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_InputTarget(self, PointCloud cloud not None):
        self.me.setInputTarget(cloud.thisptr_shared)

    def set_MaximumIterations(self, int count):
        self.me.setMaximumIterations(count)

    def set_TransformationEpsilon(self, double epsilon):
        self.me.setTransformationEpsilon(epsilon)

    def set_Resolution(self, float resolution):
        """Voxel size of the target's normal-distribution grid."""
        self.me.setResolution(resolution)

    def get_Resolution(self):
        return self.me.getResolution()

    def set_StepSize(self, double step_size):
        """More-Thuente line-search maximum step length."""
        self.me.setStepSize(step_size)

    def get_StepSize(self):
        return self.me.getStepSize()

    def set_OulierRatio(self, double outlier_ratio):
        """Expected fraction of outliers (PCL's spelling of `outlier`)."""
        self.me.setOulierRatio(outlier_ratio)

    def get_OulierRatio(self):
        return self.me.getOulierRatio()

    def get_TransformationProbability(self):
        return self.me.getTransformationProbability()

    def get_FinalNumIteration(self):
        return self.me.getFinalNumIteration()

    def ndt(self, PointCloud source not None, PointCloud target not None,
            max_iter=None):
        """Align *source* onto *target*.

        Returns ``(converged, transform, estimate, fitness)``.
        """
        cdef float buf[16]
        cdef PointCloud estimate = PointCloud()
        cdef cPointCloud[PointXYZ]* out = estimate.ptr()

        self.me.setInputSource(source.thisptr_shared)
        self.me.setInputTarget(target.thisptr_shared)
        if max_iter is not None:
            self.me.setMaximumIterations(max_iter)
        with nogil:
            self.me.align(deref(out))
            finalTransformation(deref(self.me), buf)
        return (self.me.hasConverged(), _as_matrix(buf), estimate,
                self.me.getFitnessScore())
