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

`RangeImagePlanar` (a pinhole depth image) and `RangeImageSpherical` (a
spinning LIDAR's scan) are RangeImage subclasses that only change how a
pixel maps to a direction, so every method below works on them.
`RangeImageBorderExtractor` and `NarfDescriptor` take any of the three.
"""

from cython.operator cimport dereference as deref

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport PointXYZ
from pcl.pxd.range_image.range_image cimport RangeImage as cRangeImage
from pcl.pxd.compat.range_image_args cimport (
    RANGE_IMAGE_CAMERA_FRAME, RANGE_IMAGE_LASER_FRAME, makeRangeImage,
    makeRangeImagePlanar, makeRangeImagePlanarFromDepth,
    bearingAngle, bearingAngleImage,
    makeRangeImageSpherical, narfDescriptors, narfKeypoints,
    rangeImageBorderTraits, rangeImageHeight, rangeImagePlanarIntrinsics,
    rangeImagePoints, rangeImageRanges, rangeImageSetUnseenToMaxRange,
    rangeImageWidth)
cimport pcl.pxd.compat.range_image_args as ria

from pcl._pointcloud cimport PointCloud


# RangeImage::CoordinateFrame, re-exported from the header through the
# shim so the values cannot drift.
CAMERA_FRAME = RANGE_IMAGE_CAMERA_FRAME
LASER_FRAME = RANGE_IMAGE_LASER_FRAME

DEG2RAD = 0.017453292519943295

# pcl::BorderTrait: bit positions in the per-pixel values
# RangeImageBorderExtractor.compute() returns. Test one with
# ``traits & (1 << pcl.BORDER_TRAIT__OBSTACLE_BORDER)``.
BORDER_TRAIT__OBSTACLE_BORDER = ria.BORDER_TRAIT_OBSTACLE_BORDER
BORDER_TRAIT__SHADOW_BORDER = ria.BORDER_TRAIT_SHADOW_BORDER
BORDER_TRAIT__VEIL_POINT = ria.BORDER_TRAIT_VEIL_POINT
BORDER_TRAIT__SHADOW_BORDER_TOP = ria.BORDER_TRAIT_SHADOW_BORDER_TOP
BORDER_TRAIT__SHADOW_BORDER_RIGHT = ria.BORDER_TRAIT_SHADOW_BORDER_RIGHT
BORDER_TRAIT__SHADOW_BORDER_BOTTOM = ria.BORDER_TRAIT_SHADOW_BORDER_BOTTOM
BORDER_TRAIT__SHADOW_BORDER_LEFT = ria.BORDER_TRAIT_SHADOW_BORDER_LEFT
BORDER_TRAIT__OBSTACLE_BORDER_TOP = ria.BORDER_TRAIT_OBSTACLE_BORDER_TOP
BORDER_TRAIT__OBSTACLE_BORDER_RIGHT = ria.BORDER_TRAIT_OBSTACLE_BORDER_RIGHT
BORDER_TRAIT__OBSTACLE_BORDER_BOTTOM = ria.BORDER_TRAIT_OBSTACLE_BORDER_BOTTOM
BORDER_TRAIT__OBSTACLE_BORDER_LEFT = ria.BORDER_TRAIT_OBSTACLE_BORDER_LEFT
BORDER_TRAIT__VEIL_POINT_TOP = ria.BORDER_TRAIT_VEIL_POINT_TOP
BORDER_TRAIT__VEIL_POINT_RIGHT = ria.BORDER_TRAIT_VEIL_POINT_RIGHT
BORDER_TRAIT__VEIL_POINT_BOTTOM = ria.BORDER_TRAIT_VEIL_POINT_BOTTOM
BORDER_TRAIT__VEIL_POINT_LEFT = ria.BORDER_TRAIT_VEIL_POINT_LEFT


cdef class RangeImage:
    """A cloud rendered as a depth image (pcl::RangeImage).

    Angles are in radians; `angular_resolution` is the angle one pixel
    spans. The sensor sits at the origin looking along the coordinate
    frame's axis — the pose every PCL example uses, and the only one an
    Eigen-free signature can offer.
    """

    cdef shared_ptr[cRangeImage] thisptr

    def __cinit__(self, *args, **kwargs):
        # Built in __init__, so RangeImagePlanar / RangeImageSpherical can
        # take their own constructor arguments.
        pass

    cdef cRangeImage* _image(self) except NULL:
        if not self.thisptr:
            raise RuntimeError("range image was never initialised")
        return self.thisptr.get()

    def __init__(self, PointCloud pc not None,
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
        return rangeImageWidth(deref(self._image()))

    @property
    def height(self):
        return rangeImageHeight(deref(self._image()))

    @property
    def size(self):
        return self.width * self.height

    def __len__(self):
        return self.size

    def set_unseen_to_max_range(self):
        """Give unobserved pixels the maximum range instead of infinity,
        which is what NARF wants when the scan has holes."""
        rangeImageSetUnseenToMaxRange(deref(self._image()))

    def to_array(self):
        """Return the ranges as a ``(height, width)`` float32 array.

        Unobserved pixels are infinity — that is how PCL marks them.
        """
        import numpy as np
        cdef vector[float] ranges
        rangeImageRanges(deref(self._image()), ranges)

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
        rangeImagePoints(deref(self._image()), points)

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
        cdef cRangeImage* image = self._image()
        with nogil:
            narfKeypoints(deref(image), support_size,
                          add_points_on_straight_edges, indices)
        return [indices[i] for i in range(indices.size())]


cdef class RangeImagePlanar(RangeImage):
    """A range image from a pinhole camera (pcl::RangeImagePlanar): the
    pixel grid IS the camera's, so a depth frame maps onto it one to one.

        image = pcl.RangeImagePlanar(cloud, 640, 480, 320, 240, 525, 525)
        image = pcl.RangeImagePlanar.from_depth_image(depth, 320, 240,
                                                       525, 525)

    The camera sits at the origin looking along +z (camera frame).
    """

    def __init__(self, PointCloud pc not None, int width, int height,
                 float center_x, float center_y, float focal_length_x,
                 float focal_length_y,
                 int coordinate_frame=RANGE_IMAGE_CAMERA_FRAME,
                 float noise_level=0.0, float min_range=0.0):
        _check_intrinsics(width, height, focal_length_x, focal_length_y)
        self.thisptr = makeRangeImagePlanar(
            deref(pc.ptr()), width, height, center_x, center_y,
            focal_length_x, focal_length_y, coordinate_frame, noise_level,
            min_range)

    @staticmethod
    def from_depth_image(depth, float center_x, float center_y,
                         float focal_length_x, float focal_length_y):
        """Build from a ``(height, width)`` depth array in the cloud's own
        units (non-finite or <= 0 pixels count as unobserved)."""
        import numpy as np
        arr = np.ascontiguousarray(depth, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("depth must be a (height, width) array")
        cdef int height = arr.shape[0]
        cdef int width = arr.shape[1]
        _check_intrinsics(width, height, focal_length_x, focal_length_y)
        cdef float[:, ::1] view = arr
        cdef RangeImagePlanar image = RangeImagePlanar.__new__(RangeImagePlanar)
        image.thisptr = makeRangeImagePlanarFromDepth(
            &view[0, 0], width, height, center_x, center_y, focal_length_x,
            focal_length_y)
        return image

    def _intrinsics(self):
        cdef float buf[4]
        if not rangeImagePlanarIntrinsics(deref(self._image()), buf):
            raise RuntimeError("not a planar range image")
        return buf[0], buf[1], buf[2], buf[3]

    def get_FocalLength(self):
        """``(fx, fy)`` in pixels."""
        fx, fy, _, _ = self._intrinsics()
        return fx, fy

    def get_Center(self):
        """``(cx, cy)``, the principal point in pixels."""
        _, _, cx, cy = self._intrinsics()
        return cx, cy


cdef _check_intrinsics(int width, int height, float fx, float fy):
    if width < 1 or height < 1:
        raise ValueError("image must be at least 1x1, got %dx%d"
                         % (width, height))
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")


cdef class RangeImageSpherical(RangeImage):
    """A range image whose pixels are equal steps of azimuth and
    elevation (pcl::RangeImageSpherical) — the native layout of a
    spinning LIDAR. Same constructor as :class:`RangeImage`.

    Not available on Windows: the class's inline constructor puts its
    vtable in the caller's module, and that vtable needs base-class
    virtuals and static lookup tables PCL does not export, so no module
    outside PCL can link it there; construction raises RuntimeError."""

    def __init__(self, PointCloud pc not None,
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
        self.thisptr = makeRangeImageSpherical(
            deref(pc.ptr()), angular_resolution, max_angle_width,
            max_angle_height, coordinate_frame, noise_level, min_range,
            border_size)


cdef class RangeImageBorderExtractor:
    """Classify every pixel of a range image as obstacle border, shadow
    border, veil point or none (pcl::RangeImageBorderExtractor) — the
    first stage of NARF, useful on its own to find object outlines.

        traits = pcl.RangeImageBorderExtractor(image).compute()
        outline = traits & (1 << pcl.BORDER_TRAIT__OBSTACLE_BORDER) != 0
    """

    cdef RangeImage image
    cdef int pixel_radius_borders
    cdef float minimum_border_probability

    def __cinit__(self, RangeImage range_image not None):
        range_image._image()
        self.image = range_image
        self.pixel_radius_borders = 3
        self.minimum_border_probability = 0.8

    def set_PixelRadiusBorders(self, int radius):
        """Pixels searched on each side for the border (PCL default 3)."""
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.pixel_radius_borders = radius

    def get_PixelRadiusBorders(self):
        return self.pixel_radius_borders

    def set_MinimumBorderProbability(self, float probability):
        """Score a pixel needs to count as a border (0..1, default 0.8)."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.minimum_border_probability = probability

    def get_MinimumBorderProbability(self):
        return self.minimum_border_probability

    def compute(self):
        """Return a ``(height, width)`` uint32 array of BorderTraits
        bitsets, one per pixel; test bits with the
        ``pcl.BORDER_TRAIT__*`` positions."""
        import numpy as np
        cdef vector[unsigned int] traits
        cdef cRangeImage* image = self.image._image()
        with nogil:
            rangeImageBorderTraits(deref(image), self.pixel_radius_borders,
                                   self.minimum_border_probability, traits)
        result = np.empty(traits.size(), dtype=np.uint32)
        cdef unsigned int[::1] view = result
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t> traits.size()):
            view[i] = traits[i]
        return result.reshape(self.image.height, self.image.width)


cdef class NarfDescriptor:
    """NARF descriptors at chosen pixels of a range image
    (pcl::NarfDescriptor) — the descriptor half of NARF, paired with
    `RangeImage.narf_keypoints()`:

        keypoints = image.narf_keypoints(support_size=0.2)
        features = pcl.NarfDescriptor(image, support_size=0.2).compute(
            keypoints)

    `support_size` is the diameter of the described surface in world
    units and must be set (PCL's default, -1, describes nothing).
    """

    cdef RangeImage image
    cdef float support_size
    cdef bint rotation_invariant

    def __cinit__(self, RangeImage range_image not None,
                  float support_size=0.2, bint rotation_invariant=True):
        range_image._image()
        self.image = range_image
        self.set_SupportSize(support_size)
        self.rotation_invariant = rotation_invariant

    def set_SupportSize(self, float size):
        if size <= 0:
            raise ValueError("support_size must be > 0, got %r" % size)
        self.support_size = size

    def get_SupportSize(self):
        return self.support_size

    def set_RotationInvariant(self, bint value):
        """Rotate each descriptor to its dominant orientation (default).
        A pixel with several dominant orientations then yields several
        descriptors."""
        self.rotation_invariant = value

    def get_RotationInvariant(self):
        return self.rotation_invariant

    def compute(self, indices):
        """Describe the pixels *indices* (as `narf_keypoints` returns
        them). Returns an ``(n, 42)`` float32 array: x, y, z, roll, pitch,
        yaw of the descriptor's frame, then the 36 descriptor values. With
        rotation invariance `n` can exceed ``len(indices)``; the x, y, z
        columns say which keypoint a row belongs to."""
        import numpy as np
        cdef cRangeImage* image = self.image._image()
        cdef Py_ssize_t n_pixels = self.image.size
        cdef vector[int] pixels
        cdef int index
        for value in indices:
            index = <int> value
            if index < 0 or index >= n_pixels:
                raise IndexError("pixel index %d is outside the image (%d "
                                 "pixels)" % (index, n_pixels))
            pixels.push_back(index)
        cdef vector[float] flat
        cdef int count
        with nogil:
            count = narfDescriptors(deref(image), pixels, self.support_size,
                                    self.rotation_invariant, flat)
        result = np.empty((count, 42), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef Py_ssize_t i, j
        for i in range(count):
            for j in range(42):
                view[i, j] = flat[i * 42 + j]
        return result


def bearing_angle_image(PointCloud pc not None):
    """The bearing angle image of an ORGANIZED cloud
    (pcl::BearingAngleImage): per pixel, the angle between the laser beam
    and the step to its neighbour, as a ``(height, width)`` uint8 grey
    level (0..255 for 0..180 degrees) -- an edge-friendly image of a
    range scan for 2-D feature detectors."""
    import numpy as np
    if pc.height <= 1:
        raise ValueError("bearing_angle_image needs an organized cloud")
    cdef vector[unsigned char] grey
    cdef unsigned int w = 0, h = 0
    bearingAngleImage(deref(pc.ptr()), grey, w, h)
    out = np.empty(grey.size(), dtype=np.uint8)
    cdef unsigned char[::1] view = out
    cdef size_t i
    for i in range(grey.size()):
        view[i] = grey[i]
    return out.reshape(h, w)


def bearing_angle(point1, point2):
    """BearingAngleImage::getAngle: the angle in degrees, at `point1`,
    between the ray from the origin and the step to `point2`."""
    cdef PointXYZ a, b
    a.x, a.y, a.z = point1
    b.x, b.y, b.z = point2
    return bearingAngle(a, b)

