# distutils: language = c++
# cython: language_level=3
"""File formats and images beyond PCD / PLY / OBJ: IFS and ASCII clouds
(also reachable through `pcl.load` / `pcl.save`), PNG writing, images
extracted from organized clouds, PCL's LZF codec and its .pclzf image
files, debayering, and `ImageGrabber` -- a directory of depth (+ colour)
images read frame by frame.

    depth = pcl.extract_image(cloud, "z")        # (h, w) uint16
    pcl.save_png("depth.png", depth)
    frames = pcl.ImageGrabber("recording/", pclzf_mode=True)
    first = frames[0]                            # PointCloud_PointXYZRGBA
"""

import os

from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector

from pcl.pxd.point_types cimport (
    Normal, PointXYZ, PointXYZI, PointXYZRGB, PointXYZRGBA)
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.io.ifs_io cimport loadIFSFile
from pcl.pxd.io.lzf cimport lzfCompress, lzfDecompress
from pcl.pxd.io.debayer cimport DeBayer
from pcl.pxd.compat.io_args cimport (
    ImageSequence, extractImage, havePNG, labelImage, loadASCIICloud,
    saveIFSCloud,
    readPCLZFDepth, readPCLZFImage, readPCLZFRGBD, savePNG16, savePNG8,
    writePCLZFBayer8, writePCLZFDepth, writePCLZFParameters, writePCLZFRGB,
    writePCLZFYUV422)
cimport pcl.pxd.compat.io_args as ia

from pcl._pointcloud cimport PointCloud
from pcl._pointtypes cimport (
    PointCloud_Normal, PointCloud_PointXYZI, PointCloud_PointXYZRGB,
    PointCloud_PointXYZRGBA, wrap_xyzrgba_cloud)
from libcpp.memory cimport shared_ptr


cdef string _path(value) except *:
    return os.fsencode(os.fspath(value))


# --- IFS and ASCII clouds ---------------------------------------------------------

def _read_ifs(PointCloud cloud not None, path):
    """`pcl.load` back-end: fill *cloud* from an IFS file; PCL's code."""
    cdef string p = _path(path)
    cdef cPointCloud[PointXYZ]* out = cloud.ptr()
    cdef int error
    with nogil:
        error = loadIFSFile(p, deref(out))
    return error


def _write_ifs(PointCloud cloud not None, path, binary=False):
    """`pcl.save` back-end: IFS is binary only; *binary* is ignored. PCL's
    own writer stores PointXYZ's padding and corrupts the file, so the
    vertices are packed here first; IFS cannot hold NaN points."""
    import numpy as np
    if not np.all(np.isfinite(cloud.to_array())):
        raise ValueError("IFS cannot store NaN / inf points; remove them first")
    cdef string p = _path(path)
    cdef cPointCloud[PointXYZ]* src = cloud.ptr()
    cdef int error
    with nogil:
        error = saveIFSCloud(p, deref(src))
    return error


def load_ascii(path, separators=" ,\t", extension=None):
    """A :class:`PointCloud` from a text file of ``x y z`` lines
    (pcl::ASCIIReader), split on any of `separators`. PCL insists the
    file end in a fixed extension; `extension` defaults to the file's
    own. ``pcl.load("cloud.txt")`` uses this with the defaults."""
    cdef string p = _path(path)
    if extension is None:
        extension = os.path.splitext(os.fspath(path))[1]
    cdef string seps = separators.encode()
    cdef string ext = extension.encode()
    cdef PointCloud cloud = PointCloud()
    cdef cPointCloud[PointXYZ]* out = cloud.ptr()
    cdef int error
    with nogil:
        error = loadASCIICloud(p, seps, ext, deref(out))
    if error:
        raise IOError("error while loading %s (code %d)" % (path, error))
    return cloud


def _read_ascii(PointCloud cloud not None, path):
    """`pcl.load` back-end for .txt."""
    cdef string p = _path(path)
    cdef string seps = b" ,\t"
    cdef string ext = os.path.splitext(os.fspath(path))[1].encode()
    cdef cPointCloud[PointXYZ]* out = cloud.ptr()
    cdef int error
    with nogil:
        error = loadASCIICloud(p, seps, ext, deref(out))
    return error


# --- PNG --------------------------------------------------------------------------

def save_png(path, image):
    """Write a ``(h, w)`` / ``(h, w, c)`` uint8 or uint16 array as PNG
    (pcl::io::saveCharPNGFile / saveShortPNGFile; c is 1, 3 or 4).
    uint16 suits depth in millimetres, what `ImageGrabber` reads back.
    PCL writes PNG through VTK: without it this raises RuntimeError."""
    import numpy as np
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError("expected (h, w) or (h, w, 1|3|4)")
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError("expected uint8 or uint16 pixels")
    if not havePNG():
        raise RuntimeError("this PCL was built without VTK: no PNG writer")
    arr = np.ascontiguousarray(arr)
    cdef string p = _path(path)
    cdef int h = arr.shape[0], w = arr.shape[1], c = arr.shape[2]
    cdef const uint8_t[:, :, ::1] v8
    cdef const uint16_t[:, :, ::1] v16
    if arr.dtype == np.uint8:
        v8 = arr
        with nogil:
            savePNG8(p, &v8[0, 0, 0], w, h, c)
    else:
        v16 = arr
        with nogil:
            savePNG16(p, &v16[0, 0, 0], w, h, c)


def have_png():
    """Whether this PCL can write PNG (it needs VTK)."""
    return havePNG()


# --- images from organized clouds --------------------------------------------------

IMAGE_SCALING_NO = ia.IMAGE_SCALING_NO
IMAGE_SCALING_FULL_RANGE = ia.IMAGE_SCALING_FULL_RANGE
IMAGE_SCALING_FIXED_FACTOR = ia.IMAGE_SCALING_FIXED_FACTOR
IMAGE_COLORS_MONO = ia.IMAGE_COLORS_MONO
IMAGE_COLORS_RGB_RANDOM = ia.IMAGE_COLORS_RGB_RANDOM
IMAGE_COLORS_RGB_GLASBEY = ia.IMAGE_COLORS_RGB_GLASBEY

_FIELDS = {"z": ia.IMAGE_FIELD_Z, "normal": ia.IMAGE_FIELD_NORMAL,
           "rgb": ia.IMAGE_FIELD_RGB, "curvature": ia.IMAGE_FIELD_CURVATURE,
           "intensity": ia.IMAGE_FIELD_INTENSITY}
# PCL's defaults per extractor: (scaling, factor)
_DEFAULT_SCALING = {"z": (ia.IMAGE_SCALING_FIXED_FACTOR, 10000.0),
                    "curvature": (ia.IMAGE_SCALING_FULL_RANGE, 1.0),
                    "intensity": (ia.IMAGE_SCALING_NO, 1.0)}


cdef object _image_out(vector[uint8_t]& data, int channels, unsigned int w,
                       unsigned int h):
    import numpy as np
    if channels == 3:
        out = np.empty((h, w, 3), np.uint8)
    else:
        out = np.empty((h, w), np.uint16)
    cdef uint8_t[::1] flat = out.reshape(-1).view(np.uint8)
    if data.size() != <size_t> flat.shape[0]:
        raise RuntimeError("PCL returned an image of unexpected size")
    if data.size():
        memcpy(&flat[0], data.data(), data.size())
    return out


def extract_image(cloud, field, scaling=None, scaling_factor=None,
                  bint paint_nans_with_black=False):
    """An image from one field of an ORGANIZED cloud
    (pcl::io::PointCloudImageExtractor*): ``"z"`` (uint16, scaled by
    10000 unless told otherwise), ``"curvature"`` (uint16, stretched to
    the full range), ``"intensity"`` (uint16, unscaled) from a cloud
    holding that field; ``"rgb"`` (``(h, w, 3)`` uint8) from a colour
    cloud; ``"normal"`` (uint8 RGB of the mapped normal) from a
    :class:`PointCloud_Normal`. `scaling` is one of
    ``pcl.IMAGE_SCALING_*``. NaN pixels become 0, or black with
    `paint_nans_with_black` (colour fields)."""
    if field not in _FIELDS:
        raise ValueError("field must be one of %s" % ", ".join(sorted(_FIELDS)))
    if cloud.height <= 1:
        raise ValueError("extract_image needs an organized cloud (height > 1)")
    cdef int f = _FIELDS[field]
    default = _DEFAULT_SCALING.get(field, (0, 1.0))
    cdef int method = default[0] if scaling is None else scaling
    cdef float factor = default[1] if scaling_factor is None else scaling_factor
    if not ia.IMAGE_SCALING_NO <= method <= ia.IMAGE_SCALING_FIXED_FACTOR:
        raise ValueError("scaling must be one of pcl.IMAGE_SCALING_*")
    cdef vector[uint8_t] data
    cdef int channels = 0
    cdef unsigned int w = 0, h = 0
    cdef bint ok
    cdef PointCloud xyz
    cdef PointCloud_PointXYZRGB rgb
    cdef PointCloud_PointXYZRGBA rgba
    cdef PointCloud_PointXYZI xyzi
    cdef PointCloud_Normal normals
    if isinstance(cloud, PointCloud):
        xyz = cloud
        ok = extractImage(deref(xyz.ptr()), f, method, factor,
                          paint_nans_with_black, 0, data, channels, w, h)
    elif isinstance(cloud, PointCloud_PointXYZRGB):
        rgb = cloud
        ok = extractImage(deref(rgb.ptr()), f, method, factor,
                          paint_nans_with_black, 0, data, channels, w, h)
    elif isinstance(cloud, PointCloud_PointXYZRGBA):
        rgba = cloud
        ok = extractImage(deref(rgba.ptr()), f, method, factor,
                          paint_nans_with_black, 0, data, channels, w, h)
    elif isinstance(cloud, PointCloud_PointXYZI):
        xyzi = cloud
        ok = extractImage(deref(xyzi.ptr()), f, method, factor,
                          paint_nans_with_black, 0, data, channels, w, h)
    elif isinstance(cloud, PointCloud_Normal):
        normals = cloud
        ok = extractImage(deref(normals.ptr()), f, method, factor,
                          paint_nans_with_black, 0, data, channels, w, h)
    else:
        raise TypeError("unsupported cloud type %s" % type(cloud).__name__)
    if not ok:
        raise ValueError("%s has no %r field" % (type(cloud).__name__, field))
    return _image_out(data, channels, w, h)


def label_image(labels, int color_mode=0):
    """Colour a ``(h, w)`` label image (pcl::io::PointCloudImageExtractorFromLabelField):
    ``pcl.IMAGE_COLORS_MONO`` returns the labels as uint16;
    ``IMAGE_COLORS_RGB_RANDOM`` / ``IMAGE_COLORS_RGB_GLASBEY`` an
    ``(h, w, 3)`` uint8 image, one colour per label (Glasbey's table is
    built for telling neighbours apart)."""
    import numpy as np
    arr = np.asarray(labels)
    if arr.ndim != 2 or arr.dtype.kind not in "iu" or (arr.size and arr.min() < 0):
        raise ValueError("labels must be a (h, w) array of non-negative ints")
    if not ia.IMAGE_COLORS_MONO <= color_mode <= ia.IMAGE_COLORS_RGB_GLASBEY:
        raise ValueError("color_mode must be one of pcl.IMAGE_COLORS_*")
    cdef vector[uint32_t] flat = arr.astype(np.uint32).ravel().tolist()
    cdef vector[uint8_t] data
    cdef int channels = 0
    cdef unsigned int w = 0, h = 0
    if not labelImage(flat, arr.shape[1], arr.shape[0], color_mode, data,
                      channels, w, h):
        raise RuntimeError("PCL could not colour the labels")
    return _image_out(data, channels, w, h)


# --- LZF ------------------------------------------------------------------------

def lzf_compress(data):
    """Compress bytes with PCL's LZF codec (pcl::lzfCompress) -- what
    binary_compressed PCD and .pclzf use. The result carries no length:
    keep ``len(data)`` for `lzf_decompress`."""
    cdef const uint8_t[::1] src = memoryview(bytes(data)).cast("B")
    cdef unsigned int n = src.shape[0]
    if n == 0:
        return b""
    cdef unsigned int cap = n + n // 16 + 64
    out = bytearray(cap)
    cdef uint8_t[::1] dst = out
    cdef unsigned int written
    with nogil:
        written = lzfCompress(&src[0], n, &dst[0], cap)
    if written == 0:
        raise RuntimeError("LZF compression failed")
    return bytes(out[:written])


def lzf_decompress(data, unsigned int size):
    """Inverse of `lzf_compress`; `size` is the uncompressed length."""
    cdef const uint8_t[::1] src = memoryview(bytes(data)).cast("B")
    if size == 0:
        return b""
    if src.shape[0] == 0:
        raise ValueError("no compressed data")
    out = bytearray(size)
    cdef uint8_t[::1] dst = out
    cdef unsigned int written
    with nogil:
        written = lzfDecompress(&src[0], src.shape[0], &dst[0], size)
    if written != size:
        raise ValueError("corrupt LZF data, or size is not the original "
                         "length (got %d bytes)" % written)
    return bytes(out)


# --- PCL-LZF image files -------------------------------------------------------------

def write_pclzf_depth(path, depth):
    """Write a ``(h, w)`` uint16 depth image (millimetres) as .pclzf
    (pcl::io::LZFDepth16ImageWriter)."""
    import numpy as np
    arr = np.ascontiguousarray(depth)
    if arr.ndim != 2 or arr.dtype != np.uint16:
        raise ValueError("depth must be a (h, w) uint16 array")
    cdef const uint16_t[:, ::1] v = arr
    cdef string p = _path(path)
    cdef bint ok
    with nogil:
        ok = writePCLZFDepth(p, &v[0, 0], v.shape[1], v.shape[0])
    if not ok:
        raise IOError("could not write %s" % path)


def write_pclzf_rgb(path, rgb):
    """Write a ``(h, w, 3)`` uint8 image as .pclzf
    (pcl::io::LZFRGB24ImageWriter)."""
    import numpy as np
    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError("rgb must be a (h, w, 3) uint8 array")
    cdef const uint8_t[:, :, ::1] v = arr
    cdef string p = _path(path)
    cdef bint ok
    with nogil:
        ok = writePCLZFRGB(p, &v[0, 0, 0], v.shape[1], v.shape[0])
    if not ok:
        raise IOError("could not write %s" % path)


def write_pclzf_yuv422(path, uyvy):
    """Write a YUV 4:2:2 image as .pclzf (pcl::io::LZFYUV422ImageWriter):
    ``(h, w, 2)`` uint8 with pixel pairs packed U, Y0 | V, Y1 (UYVY), so
    ``uyvy[..., 1]`` is the luma and ``uyvy[:, 0::2, 0]`` /
    ``uyvy[:, 1::2, 0]`` the chroma; w must be even."""
    import numpy as np
    arr = np.ascontiguousarray(uyvy)
    if arr.ndim != 3 or arr.shape[2] != 2 or arr.dtype != np.uint8 or \
            arr.shape[1] % 2:
        raise ValueError("uyvy must be an (h, w, 2) uint8 array, w even")
    cdef const uint8_t[:, :, ::1] v = arr
    cdef string p = _path(path)
    cdef bint ok
    with nogil:
        ok = writePCLZFYUV422(p, &v[0, 0, 0], v.shape[1], v.shape[0])
    if not ok:
        raise IOError("could not write %s" % path)


def write_pclzf_bayer8(path, bayer):
    """Write an ``(h, w)`` uint8 Bayer mosaic as .pclzf
    (pcl::io::LZFBayer8ImageWriter); read back, it is demosaiced as
    GRBG."""
    import numpy as np
    arr = np.ascontiguousarray(bayer)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError("bayer must be an (h, w) uint8 array")
    cdef const uint8_t[:, ::1] v = arr
    cdef string p = _path(path)
    cdef bint ok
    with nogil:
        ok = writePCLZFBayer8(p, &v[0, 0], v.shape[1], v.shape[0])
    if not ok:
        raise IOError("could not write %s" % path)


def read_pclzf_image(path):
    """The colour image in an rgb24, yuv422 or bayer8 .pclzf file (its
    header says which) as ``(h, w, 3)`` uint8 RGB: YUV converted, Bayer
    demosaiced, by PCL's readers. Depth files go through
    `read_pclzf_depth`."""
    import numpy as np
    cdef vector[uint8_t] rgb
    cdef unsigned int w = 0, h = 0
    cdef string p = _path(path)
    cdef int status
    with nogil:
        status = readPCLZFImage(p, rgb, w, h)
    if status == 1:
        raise IOError("could not read a .pclzf image from %s" % path)
    if status == 2:
        raise ValueError("%s holds no colour image; use read_pclzf_depth"
                         % path)
    return np.asarray(rgb, dtype=np.uint8).reshape(h, w, 3)


def write_pclzf_parameters(path, double fx, double fy, double cx, double cy):
    """Write the camera parameters XML the .pclzf readers (and
    `ImageGrabber` in pclzf mode) need, for depth and colour alike."""
    cdef string p = _path(path)
    if not writePCLZFParameters(p, fx, fy, cx, cy):
        raise IOError("could not write %s" % path)


def read_pclzf_depth(path, parameters_path):
    """An organized :class:`PointCloud` from a depth .pclzf and its XML
    (pcl::io::LZFDepth16ImageReader); zero depth gives NaN points."""
    cdef string p = _path(path)
    cdef string xml = _path(parameters_path)
    cdef PointCloud cloud = PointCloud()
    cdef cPointCloud[PointXYZ]* out = cloud.ptr()
    cdef bint ok
    with nogil:
        ok = readPCLZFDepth(p, xml, deref(out))
    if not ok:
        raise IOError("could not read %s with %s" % (path, parameters_path))
    return cloud


def read_pclzf(depth_path, rgb_path, parameters_path):
    """An organized :class:`PointCloud_PointXYZRGBA` from a depth and an
    RGB .pclzf and their XML."""
    cdef string d = _path(depth_path)
    cdef string c = _path(rgb_path)
    cdef string xml = _path(parameters_path)
    cdef shared_ptr[cPointCloud[PointXYZRGBA]] holder
    holder.reset(new cPointCloud[PointXYZRGBA]())
    cdef cPointCloud[PointXYZRGBA]* out = holder.get()
    cdef bint ok
    with nogil:
        ok = readPCLZFRGBD(d, c, xml, deref(out))
    if not ok:
        raise IOError("could not read %s / %s with %s"
                      % (depth_path, rgb_path, parameters_path))
    return wrap_xyzrgba_cloud(holder)


# --- debayering ----------------------------------------------------------------------

_DEBAYER = ("bilinear", "edge_aware", "edge_aware_weighted")


def debayer(bayer, method="bilinear"):
    """A ``(h, w, 3)`` uint8 RGB image from a ``(h, w)`` uint8 Bayer
    mosaic in the GRBG layout the Kinect delivers (pcl::io::DeBayer):
    ``"bilinear"``, ``"edge_aware"`` or ``"edge_aware_weighted"``."""
    import numpy as np
    if method not in _DEBAYER:
        raise ValueError("method must be one of %s" % ", ".join(_DEBAYER))
    arr = np.ascontiguousarray(bayer)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError("bayer must be a (h, w) uint8 array")
    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[0] % 2 or arr.shape[1] % 2:
        raise ValueError("the mosaic needs an even width and height >= 2")
    out = np.empty(arr.shape + (3,), np.uint8)
    cdef const uint8_t[:, ::1] src = arr
    cdef uint8_t[:, :, ::1] dst = out
    cdef unsigned int w = arr.shape[1], h = arr.shape[0]
    cdef DeBayer d
    cdef int which = _DEBAYER.index(method)
    with nogil:
        if which == 0:
            d.debayerBilinear(&src[0, 0], &dst[0, 0, 0], w, h)
        elif which == 1:
            d.debayerEdgeAware(&src[0, 0], &dst[0, 0, 0], w, h)
        else:
            d.debayerEdgeAwareWeighted(&src[0, 0], &dst[0, 0, 0], w, h)
    return out


# --- ImageGrabber --------------------------------------------------------------------

cdef class ImageGrabber:
    """A directory of depth (+ colour) images as organized clouds
    (pcl::ImageGrabber), read by index: ``len(grabber)`` frames,
    ``grabber[i]`` a :class:`PointCloud_PointXYZRGBA` (black when there
    are no colour images).

    Default mode reads images PCL can decode through VTK (PNG, TIFF,
    ...), pairing files whose names contain ``depth`` and ``rgb``; give
    the intrinsics with `set_CameraIntrinsics` (PCL otherwise assumes a
    Kinect's). `pclzf_mode` reads ``*depth*.pclzf`` / ``*rgb*.pclzf``
    plus one camera-parameters XML per frame, as PCL's recorders write
    them (see `write_pclzf_depth` and friends).
    """

    cdef ImageSequence* me

    def __cinit__(self, path, bint pclzf_mode=False):
        if not os.path.isdir(os.fspath(path)):
            raise ValueError("%s is not a directory" % (path,))
        cdef string p = _path(path)
        self.me = new ImageSequence(p, pclzf_mode)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def __len__(self):
        return self.me.size()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = self.me.size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("frame index out of range")
        cdef shared_ptr[cPointCloud[PointXYZRGBA]] holder
        holder.reset(new cPointCloud[PointXYZRGBA]())
        cdef cPointCloud[PointXYZRGBA]* out = holder.get()
        with nogil:
            self.me.frame(index, deref(out))
        return wrap_xyzrgba_cloud(holder)

    def set_CameraIntrinsics(self, double fx, double fy, double cx, double cy):
        self.me.setCameraIntrinsics(fx, fy, cx, cy)

    def get_CameraIntrinsics(self):
        """``(fx, fy, cx, cy)`` as given to `set_CameraIntrinsics`, else
        PCL's Kinect defaults (525, 525, 319.5, 239.5). In pclzf mode
        each frame is projected with its own XML regardless; reading by
        index does not update these values (PCL only does so while
        replaying)."""
        cdef double out[4]
        self.me.cameraIntrinsics(out)
        return (out[0], out[1], out[2], out[3])

    def set_DepthImageUnits(self, float units):
        """Metres per depth unit for image files (default 0.001: mm)."""
        if units <= 0:
            raise ValueError("units must be positive")
        self.me.setDepthImageUnits(units)
