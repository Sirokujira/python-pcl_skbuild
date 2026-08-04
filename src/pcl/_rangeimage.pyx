# distutils: language = c++
# cython: language_level=3
"""RangeImage and NARF keypoints, named after sirokujira/python-pcl
(pcl/pxi/Common/RangeImage/, pcl/pxi/KeyPoint/NarfKeypoint.pxi).

A range image is a cloud seen as a depth image from one viewpoint: the
form the NARF detector and descriptor work in.

    image = pcl.RangeImage(cloud, angular_resolution=0.5)
    keypoints = image.narf_keypoints(support_size=0.2)

Everything here goes through pcl/compat/range_image_args.h, which owns
the whole pipeline. Building a RangeImage needs an `Eigen::Affine3f`
pose and an enum nested in the class; reading one means indexing a pixel
grid of `PointWithRange`; and NARF needs a `RangeImageBorderExtractor`
wired to the same image, its settings in a nested `Parameters` struct,
and returns a `PointCloud<int>`. Mirroring four types to run one
algorithm would buy nothing — the shim hands back a float buffer and a
list of indices instead.
"""

from cython.operator cimport dereference as deref

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.range_image.range_image cimport RangeImage as cRangeImage
from pcl.pxd.compat.range_image_args cimport (
    RANGE_IMAGE_CAMERA_FRAME, RANGE_IMAGE_LASER_FRAME, makeRangeImage,
    narfKeypoints, rangeImageHeight, rangeImagePoints, rangeImageRanges,
    rangeImageSetUnseenToMaxRange, rangeImageWidth)

from pcl._pointcloud cimport PointCloud


# RangeImage::CoordinateFrame, re-exported from the header through the
# shim so the values cannot drift.
CAMERA_FRAME = RANGE_IMAGE_CAMERA_FRAME
LASER_FRAME = RANGE_IMAGE_LASER_FRAME

DEG2RAD = 0.017453292519943295


cdef class RangeImage:
    """A cloud rendered as a depth image (pcl::RangeImage).

    Angles are in radians; `angular_resolution` is the angle one pixel
    spans. The sensor sits at the origin looking along the coordinate
    frame's axis — the pose every PCL example uses, and the only one an
    Eigen-free signature can offer.
    """

    cdef shared_ptr[cRangeImage] thisptr

    def __cinit__(self, PointCloud pc not None,
                  float angular_resolution=0.5 * DEG2RAD,
                  float max_angle_width=360.0 * DEG2RAD,
                  float max_angle_height=180.0 * DEG2RAD,
                  int coordinate_frame=RANGE_IMAGE_CAMERA_FRAME,
                  float noise_level=0.0,
                  float min_range=0.0,
                  int border_size=1):
        if angular_resolution <= 0:
            raise ValueError(
                "angular_resolution must be > 0, got %r" % angular_resolution)
        self.thisptr = makeRangeImage(
            deref(pc.ptr()), angular_resolution, max_angle_width,
            max_angle_height, coordinate_frame, noise_level, min_range,
            border_size)

    @property
    def width(self):
        return rangeImageWidth(deref(self.thisptr))

    @property
    def height(self):
        return rangeImageHeight(deref(self.thisptr))

    @property
    def size(self):
        return self.width * self.height

    def __len__(self):
        return self.size

    def set_unseen_to_max_range(self):
        """Give unobserved pixels the maximum range instead of infinity,
        which is what NARF wants when the scan has holes."""
        rangeImageSetUnseenToMaxRange(deref(self.thisptr))

    def to_array(self):
        """Return the ranges as a ``(height, width)`` float32 array.

        Unobserved pixels are infinity — that is how PCL marks them.
        """
        import numpy as np
        cdef vector[float] ranges
        rangeImageRanges(deref(self.thisptr), ranges)

        cdef Py_ssize_t n = <Py_ssize_t> ranges.size()
        result = np.empty(n, dtype=np.float32)
        cdef float[::1] view = result
        cdef Py_ssize_t i
        for i in range(n):
            view[i] = ranges[i]
        return result.reshape(self.height, self.width)

    def to_cloud(self):
        """Return the 3-D points behind the pixels as a PointCloud."""
        cdef vector[PointXYZ] points
        rangeImagePoints(deref(self.thisptr), points)

        cdef PointCloud result = PointCloud()
        cdef Py_ssize_t n = <Py_ssize_t> points.size()
        result.resize(n)
        cdef Py_ssize_t i
        for i in range(n):
            deref(result.ptr())[<size_t> i] = points[i]
        return result

    def narf_keypoints(self, float support_size=0.2,
                       bint add_points_on_straight_edges=False):
        """Indices of the NARF interest points, into this image's pixels.

        *support_size* is the diameter, in world units, of the surface a
        keypoint describes — the one setting that matters.
        """
        if support_size <= 0:
            raise ValueError(
                "support_size must be > 0, got %r" % support_size)
        cdef vector[int] indices
        cdef cRangeImage* image = self.thisptr.get()
        with nogil:
            narfKeypoints(deref(image), support_size,
                          add_points_on_straight_edges, indices)
        return [indices[i] for i in range(indices.size())]
