# distutils: language = c++
# cython: language_level=3
"""Clouds of the coloured and intensity point types.

Named after sirokujira/python-pcl (pcl/pxi/PointCloud_PointXYZI.pxi and
friends): `PointCloud_PointXYZI`, `PointCloud_PointXYZRGB`,
`PointCloud_PointXYZRGBA`, reached through `pcl.load_XYZI`,
`pcl.load_XYZRGB` and `pcl.load_XYZRGBA`.

`to_array()` / `from_array()` keep python-pcl's ``(n, 4)`` float32 layout,
where the fourth column is intensity, or -- for the coloured types -- the
packed RGB value reinterpreted as a float. That last part is a real trap:
the number in that column is a bit pattern, not a colour, so arithmetic
on it is meaningless and any processing step that interpolates points
produces nonsense colours (strawlab/python-pcl#242 is someone hitting
exactly this). `to_rgb_array()` / `from_rgb_array()` expose the same
field as plain ``(n, 3)`` uint8, which is what colour handling actually
wants; the packed form stays for compatibility.

PCL stores those bytes in a union, so neither view costs a conversion --
they are two ways of reading the same four bytes.
"""

import numbers

from cython.operator cimport dereference as deref

from libc.stdint cimport uint8_t
from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from pcl.pxd.point_types cimport (
    Normal, PointNormal, PointWithViewpoint, PointXYZI, PointXYZRGB,
    PointXYZRGBA)
from pcl.pxd.point_types cimport PointXYZ
from pcl._pointcloud cimport PointCloud
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.io.pcd_io cimport loadPCDFile, savePCDFile
from pcl.pxd.io.ply_io cimport loadPLYFile, savePLYFile
from pcl.pxd.io.obj_io cimport loadOBJFile
from pcl.pxd.compat.transform_args cimport (
    isRigidTransform, transformCloud)


cdef string _topath(path):
    if isinstance(path, str):
        return <string> path.encode()
    return <string> path


cdef class PointCloud_PointXYZI:
    """A cloud of pcl::PointXYZI (x, y, z, intensity)."""

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[PointXYZI]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_PointXYZI):
            self.ptr()[0] = (<PointCloud_PointXYZI> init).ptr()[0]
        else:
            try:
                self.from_array(init)
            except (TypeError, ValueError):
                self.from_list(init)

    cdef cPointCloud[PointXYZI]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    @property
    def is_dense(self):
        return self.ptr().is_dense

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def make_BilateralFilter(self):
        """Return a BilateralFilter with this cloud as input."""
        from pcl._filters import BilateralFilter
        return BilateralFilter(self)

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 4)`` float32 array: x, y, z, intensity."""
        if arr.shape[1] != 4:
            raise ValueError(
                "array must have shape (n, 4), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef PointXYZI* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = arr[i, 0]
            p.y = arr[i, 1]
            p.z = arr[i, 2]
            p.intensity = arr[i, 3]

    def to_array(self):
        """Return an ``(n, 4)`` float32 array: x, y, z, intensity."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef PointXYZI* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.intensity
        return result

    def from_list(self, _list):
        pts = list(_list)
        self.resize(len(pts))
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef PointXYZI* p
        cdef Py_ssize_t i = 0
        for x, y, z, intensity in pts:
            p = &(deref(c)[<size_t> i])
            p.x = x
            p.y = y
            p.z = z
            p.intensity = intensity
            i += 1

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef PointXYZI* p = &(deref(self.ptr())[<size_t> index])
        return p.x, p.y, p.z, p.intensity

    def __reduce__(self):
        return type(self), (self.to_list(),)

    def _from_pcd_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPCDFile[PointXYZI](s, deref(c))
        return error

    def _to_pcd_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePCDFile[PointXYZI](s, deref(c), binary)
        return error

    def _from_ply_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPLYFile[PointXYZI](s, deref(c))
        return error

    def _from_obj_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadOBJFile[PointXYZI](s, deref(c))
        return error

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZI](s, deref(c), binary)
        return error


    def transform(self, matrix):
        """Return this cloud moved by a 4x4 transform.

        Takes what registration and recognition hand back, so a result
        can be applied without a numpy round trip:

            converged, matrix, _, _ = icp.icp(source, target)
            aligned = source.transform(matrix)

        Raises when the matrix is not a rigid motion — PCL would apply a
        scaling or reflecting one silently.
        """
        cdef float buf[16]
        _fill_transform(matrix, buf)
        cdef PointCloud_PointXYZI result = PointCloud_PointXYZI()
        transformCloud(deref(self.ptr()), deref(result.ptr()), buf)
        return result


cdef class PointCloud_PointXYZRGB:
    """A cloud of pcl::PointXYZRGB (x, y, z + 8-bit colour)."""

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[PointXYZRGB]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_PointXYZRGB):
            self.ptr()[0] = (<PointCloud_PointXYZRGB> init).ptr()[0]
        else:
            try:
                self.from_array(init)
            except (TypeError, ValueError):
                self.from_list(init)

    cdef cPointCloud[PointXYZRGB]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    @property
    def is_dense(self):
        return self.ptr().is_dense

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def make_RegionGrowingRGB(self):
        """Return a RegionGrowingRGB with this cloud as input."""
        from pcl._segmentation import RegionGrowingRGB
        return RegionGrowingRGB(self)

    def make_GASDColorEstimation(self):
        """Return a GASDColorEstimation with this cloud as input."""
        from pcl._features import GASDColorEstimation
        return GASDColorEstimation(self)

    def make_SIFTKeypoint(self):
        """Return a SIFTKeypoint detector with this cloud as input."""
        from pcl._keypoints import SIFTKeypoint
        return SIFTKeypoint(self)

    def make_SUSANKeypoint(self):
        """Return a SUSANKeypoint detector with this cloud as input."""
        from pcl._keypoints import SUSANKeypoint
        return SUSANKeypoint(self)

    def make_SeededHueSegmentation(self):
        """Return a SeededHueSegmentation with this cloud as input."""
        from pcl._segmentation import SeededHueSegmentation
        return SeededHueSegmentation(self)

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 4)`` float32 array: x, y, z, packed rgb."""
        if arr.shape[1] != 4:
            raise ValueError(
                "array must have shape (n, 4), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef PointXYZRGB* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = arr[i, 0]
            p.y = arr[i, 1]
            p.z = arr[i, 2]
            p.rgb = arr[i, 3]

    def to_array(self):
        """Return an ``(n, 4)`` float32 array: x, y, z, packed rgb.

        Column 3 is a bit pattern, not a number — see `to_rgb_array`.
        """
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef PointXYZRGB* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.rgb
        return result

    def to_xyz_array(self):
        """Return just the geometry as an ``(n, 3)`` float32 array."""
        return self.to_array()[:, :3].copy()

    def to_rgb_array(self):
        """Return the colours as an ``(n, 3)`` uint8 array: r, g, b."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 3), dtype=np.uint8)
        cdef uint8_t[:, ::1] view = result
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef PointXYZRGB* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.r
            view[i, 1] = p.g
            view[i, 2] = p.b
        return result

    def from_rgb_array(self, float[:, :] xyz not None,
                       uint8_t[:, :] rgb not None):
        """Fill from geometry ``(n, 3)`` float32 and colour ``(n, 3)``
        uint8 — the pair `to_xyz_array` / `to_rgb_array` hand back."""
        if xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (n, 3)")
        if rgb.shape[1] != 3:
            raise ValueError("rgb must have shape (n, 3)")
        if xyz.shape[0] != rgb.shape[0]:
            raise ValueError(
                "xyz and rgb must have the same length, got %d and %d"
                % (xyz.shape[0], rgb.shape[0]))

        cdef Py_ssize_t npts = xyz.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef PointXYZRGB* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = xyz[i, 0]
            p.y = xyz[i, 1]
            p.z = xyz[i, 2]
            p.r = rgb[i, 0]
            p.g = rgb[i, 1]
            p.b = rgb[i, 2]
            p.a = 255

    def from_list(self, _list):
        pts = list(_list)
        self.resize(len(pts))
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef PointXYZRGB* p
        cdef Py_ssize_t i = 0
        for x, y, z, rgb in pts:
            p = &(deref(c)[<size_t> i])
            p.x = x
            p.y = y
            p.z = z
            p.rgb = rgb
            i += 1

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef PointXYZRGB* p = &(deref(self.ptr())[<size_t> index])
        return p.x, p.y, p.z, p.rgb

    def __reduce__(self):
        return type(self), (self.to_list(),)

    def _from_pcd_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPCDFile[PointXYZRGB](s, deref(c))
        return error

    def _to_pcd_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePCDFile[PointXYZRGB](s, deref(c), binary)
        return error

    def _from_ply_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPLYFile[PointXYZRGB](s, deref(c))
        return error

    def _from_obj_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadOBJFile[PointXYZRGB](s, deref(c))
        return error

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZRGB](s, deref(c), binary)
        return error


    def transform(self, matrix):
        """Return this cloud moved by a 4x4 transform.

        Takes what registration and recognition hand back, so a result
        can be applied without a numpy round trip:

            converged, matrix, _, _ = icp.icp(source, target)
            aligned = source.transform(matrix)

        Raises when the matrix is not a rigid motion — PCL would apply a
        scaling or reflecting one silently.
        """
        cdef float buf[16]
        _fill_transform(matrix, buf)
        cdef PointCloud_PointXYZRGB result = PointCloud_PointXYZRGB()
        transformCloud(deref(self.ptr()), deref(result.ptr()), buf)
        return result


cdef class PointCloud_PointXYZRGBA:
    """A cloud of pcl::PointXYZRGBA (x, y, z + 8-bit colour with alpha)."""

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[PointXYZRGBA]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_PointXYZRGBA):
            self.ptr()[0] = (<PointCloud_PointXYZRGBA> init).ptr()[0]
        else:
            try:
                self.from_array(init)
            except (TypeError, ValueError):
                self.from_list(init)

    cdef cPointCloud[PointXYZRGBA]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    @property
    def is_dense(self):
        return self.ptr().is_dense

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def make_SupervoxelClustering(self, float voxel_resolution,
                                  float seed_resolution):
        """Return a SupervoxelClustering with this cloud as input."""
        from pcl._segmentation import SupervoxelClustering
        return SupervoxelClustering(voxel_resolution, seed_resolution, self)

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 4)`` float32 array: x, y, z, packed rgb."""
        if arr.shape[1] != 4:
            raise ValueError(
                "array must have shape (n, 4), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef PointXYZRGBA* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = arr[i, 0]
            p.y = arr[i, 1]
            p.z = arr[i, 2]
            p.rgb = arr[i, 3]

    def to_array(self):
        """Return an ``(n, 4)`` float32 array: x, y, z, packed rgb."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef PointXYZRGBA* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.rgb
        return result

    def to_xyz_array(self):
        """Return just the geometry as an ``(n, 3)`` float32 array."""
        return self.to_array()[:, :3].copy()

    def to_rgba_array(self):
        """Return the colours as an ``(n, 4)`` uint8 array: r, g, b, a."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 4), dtype=np.uint8)
        cdef uint8_t[:, ::1] view = result
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef PointXYZRGBA* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.r
            view[i, 1] = p.g
            view[i, 2] = p.b
            view[i, 3] = p.a
        return result

    def from_rgba_array(self, float[:, :] xyz not None,
                        uint8_t[:, :] rgba not None):
        """Fill from geometry ``(n, 3)`` float32 and colour ``(n, 4)``
        uint8."""
        if xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (n, 3)")
        if rgba.shape[1] != 4:
            raise ValueError("rgba must have shape (n, 4)")
        if xyz.shape[0] != rgba.shape[0]:
            raise ValueError(
                "xyz and rgba must have the same length, got %d and %d"
                % (xyz.shape[0], rgba.shape[0]))

        cdef Py_ssize_t npts = xyz.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef PointXYZRGBA* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = xyz[i, 0]
            p.y = xyz[i, 1]
            p.z = xyz[i, 2]
            p.r = rgba[i, 0]
            p.g = rgba[i, 1]
            p.b = rgba[i, 2]
            p.a = rgba[i, 3]

    @property
    def is_organized(self):
        """True for a cloud laid out as an image (height > 1)."""
        return self.ptr().height > 1

    def from_organized_rgba_array(self, float[:, :, :] xyz not None,
                                  uint8_t[:, :, :] rgba not None):
        """Fill from geometry ``(height, width, 3)`` float32 and colour
        ``(height, width, 4)`` uint8, keeping the pixel grid — what an
        RGB-D frame looks like, and what `BilateralUpsampling` needs."""
        if xyz.shape[2] != 3:
            raise ValueError("xyz must have shape (height, width, 3)")
        if rgba.shape[2] != 4:
            raise ValueError("rgba must have shape (height, width, 4)")
        if xyz.shape[0] != rgba.shape[0] or xyz.shape[1] != rgba.shape[1]:
            raise ValueError(
                "xyz and rgba must share (height, width), got (%d, %d) and "
                "(%d, %d)" % (xyz.shape[0], xyz.shape[1], rgba.shape[0],
                              rgba.shape[1]))
        cdef Py_ssize_t height = xyz.shape[0]
        cdef Py_ssize_t width = xyz.shape[1]
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        c.resize(<size_t> (width * height))
        c.width = <unsigned int> width
        c.height = <unsigned int> height
        cdef PointXYZRGBA* p
        cdef Py_ssize_t row, col
        for row in range(height):
            for col in range(width):
                p = &(deref(c)[<size_t> (row * width + col)])
                p.x = xyz[row, col, 0]
                p.y = xyz[row, col, 1]
                p.z = xyz[row, col, 2]
                p.r = rgba[row, col, 0]
                p.g = rgba[row, col, 1]
                p.b = rgba[row, col, 2]
                p.a = rgba[row, col, 3]

    def make_BilateralUpsampling(self):
        """Return a BilateralUpsampling filter with this organized cloud
        as input."""
        from pcl._surface import BilateralUpsampling
        return BilateralUpsampling(self)

    def from_list(self, _list):
        pts = list(_list)
        self.resize(len(pts))
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef PointXYZRGBA* p
        cdef Py_ssize_t i = 0
        for x, y, z, rgb in pts:
            p = &(deref(c)[<size_t> i])
            p.x = x
            p.y = y
            p.z = z
            p.rgb = rgb
            i += 1

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef PointXYZRGBA* p = &(deref(self.ptr())[<size_t> index])
        return p.x, p.y, p.z, p.rgb

    def __reduce__(self):
        return type(self), (self.to_list(),)

    def _from_pcd_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPCDFile[PointXYZRGBA](s, deref(c))
        return error

    def _to_pcd_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePCDFile[PointXYZRGBA](s, deref(c), binary)
        return error

    def _from_ply_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPLYFile[PointXYZRGBA](s, deref(c))
        return error

    def _from_obj_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadOBJFile[PointXYZRGBA](s, deref(c))
        return error

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZRGBA](s, deref(c), binary)
        return error


    def transform(self, matrix):
        """Return this cloud moved by a 4x4 transform.

        Takes what registration and recognition hand back, so a result
        can be applied without a numpy round trip:

            converged, matrix, _, _ = icp.icp(source, target)
            aligned = source.transform(matrix)

        Raises when the matrix is not a rigid motion — PCL would apply a
        scaling or reflecting one silently.
        """
        cdef float buf[16]
        _fill_transform(matrix, buf)
        cdef PointCloud_PointXYZRGBA result = PointCloud_PointXYZRGBA()
        transformCloud(deref(self.ptr()), deref(result.ptr()), buf)
        return result


cdef class PointCloud_Normal:
    """A cloud of pcl::Normal (normal_x, normal_y, normal_z, curvature).

    What `NormalEstimation.compute_cloud()` returns, and what
    `SegmentationNormal.set_InputNormals()` takes — the two ends of the
    normal-based segmentation workflow. `to_array()` gives the same
    ``(n, 4)`` array `NormalEstimation.compute()` returns directly.
    """

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[Normal]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_Normal):
            self.ptr()[0] = (<PointCloud_Normal> init).ptr()[0]
        else:
            self.from_array(init)

    cdef cPointCloud[Normal]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[Normal]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 4)`` float32 array:
        normal_x, normal_y, normal_z, curvature."""
        if arr.shape[1] != 4:
            raise ValueError(
                "array must have shape (n, 4), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[Normal]* c = self.ptr()
        cdef Normal* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.normal_x = arr[i, 0]
            p.normal_y = arr[i, 1]
            p.normal_z = arr[i, 2]
            p.curvature = arr[i, 3]

    def to_array(self):
        """Return an ``(n, 4)`` float32 array:
        normal_x, normal_y, normal_z, curvature."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 4), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[Normal]* c = self.ptr()
        cdef Normal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.normal_x
            view[i, 1] = p.normal_y
            view[i, 2] = p.normal_z
            view[i, 3] = p.curvature
        return result

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef Normal* p = &(deref(self.ptr())[<size_t> index])
        return p.normal_x, p.normal_y, p.normal_z, p.curvature

    def __reduce__(self):
        return type(self), (self.to_array(),)


cdef class PointCloud_PointNormal:
    """A cloud of pcl::PointNormal (x, y, z, normal_x, normal_y, normal_z, curvature).

    Position and normal in one point, the type MovingLeastSquares and
    GreedyProjectionTriangulation work in. Build one from an existing
    cloud and its normals with `from_xyz_and_normals()`; split it back
    with `to_xyz_cloud()` / `to_normal_cloud()`.
    """

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[PointNormal]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_PointNormal):
            self.ptr()[0] = (<PointCloud_PointNormal> init).ptr()[0]
        else:
            try:
                self.from_array(init)
            except (TypeError, ValueError):
                self.from_list(init)

    cdef cPointCloud[PointNormal]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    @property
    def is_dense(self):
        return self.ptr().is_dense

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 7)`` float32 array: x, y, z, normal_x, normal_y, normal_z, curvature."""
        if arr.shape[1] != 7:
            raise ValueError(
                "array must have shape (n, 7), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef PointNormal* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = arr[i, 0]
            p.y = arr[i, 1]
            p.z = arr[i, 2]
            p.normal_x = arr[i, 3]
            p.normal_y = arr[i, 4]
            p.normal_z = arr[i, 5]
            p.curvature = arr[i, 6]

    def to_array(self):
        """Return an ``(n, 7)`` float32 array: x, y, z, normal_x, normal_y, normal_z, curvature."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 7), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef PointNormal* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.normal_x
            view[i, 4] = p.normal_y
            view[i, 5] = p.normal_z
            view[i, 6] = p.curvature
        return result

    def from_list(self, _list):
        pts = list(_list)
        self.resize(len(pts))
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef PointNormal* p
        cdef Py_ssize_t i = 0
        for x, y, z, normal_x, normal_y, normal_z, curvature in pts:
            p = &(deref(c)[<size_t> i])
            p.x = x
            p.y = y
            p.z = z
            p.normal_x = normal_x
            p.normal_y = normal_y
            p.normal_z = normal_z
            p.curvature = curvature
            i += 1

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef PointNormal* p = &(deref(self.ptr())[<size_t> index])
        return p.x, p.y, p.z, p.normal_x, p.normal_y, p.normal_z, p.curvature

    def __reduce__(self):
        return type(self), (self.to_list(),)

    def _from_pcd_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPCDFile[PointNormal](s, deref(c))
        return error

    def _to_pcd_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePCDFile[PointNormal](s, deref(c), binary)
        return error

    def _from_ply_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPLYFile[PointNormal](s, deref(c))
        return error

    def _from_obj_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadOBJFile[PointNormal](s, deref(c))
        return error

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointNormal](s, deref(c), binary)
        return error

    def transform(self, matrix):
        """Return this cloud moved by a 4x4 transform, normals included.

        Goes through pcl::transformPointCloudWithNormals, so the normals
        turn with the points (plain transformPointCloud would leave them
        pointing the old way). Raises when the matrix is not a rigid
        motion.
        """
        cdef float buf[16]
        _fill_transform(matrix, buf)
        cdef PointCloud_PointNormal result = PointCloud_PointNormal()
        transformCloud(deref(self.ptr()), deref(result.ptr()), buf)
        return result

    def from_xyz_and_normals(self, PointCloud cloud not None,
                             PointCloud_Normal normals not None):
        """Fill from a geometry cloud and the normals computed for it —
        the pair `NormalEstimation.compute_cloud()` leaves you with."""
        cdef Py_ssize_t n = <Py_ssize_t> cloud.ptr().size()
        if <Py_ssize_t> normals.ptr().size() != n:
            raise ValueError(
                "cloud has %d points but normals has %d"
                % (n, normals.ptr().size()))
        self.resize(n)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef cPointCloud[PointXYZ]* xyz = cloud.ptr()
        cdef cPointCloud[Normal]* nrm = normals.ptr()
        cdef PointNormal* p
        cdef PointXYZ* q
        cdef Normal* m
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            q = &(deref(xyz)[<size_t> i])
            m = &(deref(nrm)[<size_t> i])
            p.x = q.x
            p.y = q.y
            p.z = q.z
            p.normal_x = m.normal_x
            p.normal_y = m.normal_y
            p.normal_z = m.normal_z
            p.curvature = m.curvature
        return self

    def to_xyz_cloud(self):
        """Return the positions alone as a :class:`PointCloud`."""
        cdef PointCloud result = PointCloud()
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result.resize(n)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef cPointCloud[PointXYZ]* xyz = result.ptr()
        cdef PointNormal* p
        cdef PointXYZ* q
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            q = &(deref(xyz)[<size_t> i])
            q.x = p.x
            q.y = p.y
            q.z = p.z
        return result

    def make_Poisson(self):
        """Return a Poisson reconstructor with this cloud as input."""
        from pcl._surface import Poisson
        return Poisson(self)

    def make_MarchingCubesHoppe(self):
        """Return a MarchingCubesHoppe reconstructor with this cloud as
        input."""
        from pcl._surface import MarchingCubesHoppe
        return MarchingCubesHoppe(self)

    def make_MarchingCubesRBF(self):
        """Return a MarchingCubesRBF reconstructor with this cloud as
        input."""
        from pcl._surface import MarchingCubesRBF
        return MarchingCubesRBF(self)

    def make_SamplingSurfaceNormal(self):
        """Return a SamplingSurfaceNormal filter with this cloud as input."""
        from pcl._filters import SamplingSurfaceNormal
        return SamplingSurfaceNormal(self)

    def make_IterativeClosestPointWithNormals(self):
        """Return an IterativeClosestPointWithNormals."""
        from pcl._registration import IterativeClosestPointWithNormals
        return IterativeClosestPointWithNormals()

    def to_normal_cloud(self):
        """Return the normals alone as a :class:`PointCloud_Normal`."""
        cdef PointCloud_Normal result = PointCloud_Normal()
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result.resize(n)
        cdef cPointCloud[PointNormal]* c = self.ptr()
        cdef cPointCloud[Normal]* nrm = result.ptr()
        cdef PointNormal* p
        cdef Normal* m
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            m = &(deref(nrm)[<size_t> i])
            m.normal_x = p.normal_x
            m.normal_y = p.normal_y
            m.normal_z = p.normal_z
            m.curvature = p.curvature
        return result


cdef class PointCloud_PointWithViewpoint:
    """A cloud of pcl::PointWithViewpoint (x, y, z, vp_x, vp_y, vp_z).

    Each point remembers the sensor position it was seen from. There is
    no `transform()`: PCL's transformPointCloud moves the points and
    leaves the viewpoints where they were, which would be a silently
    half-applied motion.
    """

    def __cinit__(self, init=None):
        self.thisptr_shared.reset(new cPointCloud[PointWithViewpoint]())
        if init is None:
            return
        elif isinstance(init, numbers.Integral):
            self.resize(init)
        elif isinstance(init, PointCloud_PointWithViewpoint):
            self.ptr()[0] = (<PointCloud_PointWithViewpoint> init).ptr()[0]
        else:
            try:
                self.from_array(init)
            except (TypeError, ValueError):
                self.from_list(init)

    cdef cPointCloud[PointWithViewpoint]* ptr(self) except NULL:
        if not self.thisptr_shared:
            raise MemoryError("point cloud not allocated")
        return self.thisptr_shared.get()

    @property
    def width(self):
        return self.ptr().width

    @property
    def height(self):
        return self.ptr().height

    @property
    def size(self):
        return self.ptr().size()

    @property
    def is_dense(self):
        return self.ptr().is_dense

    def __len__(self):
        return self.ptr().size()

    def empty(self):
        return self.ptr().empty()

    def resize(self, Py_ssize_t count):
        if count < 0:
            raise ValueError("negative size %d" % count)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        c.resize(<size_t> count)
        c.width = count
        c.height = 1

    def from_array(self, float[:, :] arr not None):
        """Fill from an ``(n, 6)`` float32 array: x, y, z, vp_x, vp_y, vp_z."""
        if arr.shape[1] != 6:
            raise ValueError(
                "array must have shape (n, 6), got (%d, %d)"
                % (arr.shape[0], arr.shape[1]))
        cdef Py_ssize_t npts = arr.shape[0]
        self.resize(npts)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef PointWithViewpoint* p
        cdef Py_ssize_t i
        for i in range(npts):
            p = &(deref(c)[<size_t> i])
            p.x = arr[i, 0]
            p.y = arr[i, 1]
            p.z = arr[i, 2]
            p.vp_x = arr[i, 3]
            p.vp_y = arr[i, 4]
            p.vp_z = arr[i, 5]

    def to_array(self):
        """Return an ``(n, 6)`` float32 array: x, y, z, vp_x, vp_y, vp_z."""
        import numpy as np
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result = np.empty((n, 6), dtype=np.float32)
        cdef float[:, ::1] view = result
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef PointWithViewpoint* p
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            view[i, 0] = p.x
            view[i, 1] = p.y
            view[i, 2] = p.z
            view[i, 3] = p.vp_x
            view[i, 4] = p.vp_y
            view[i, 5] = p.vp_z
        return result

    def from_list(self, _list):
        pts = list(_list)
        self.resize(len(pts))
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef PointWithViewpoint* p
        cdef Py_ssize_t i = 0
        for x, y, z, vp_x, vp_y, vp_z in pts:
            p = &(deref(c)[<size_t> i])
            p.x = x
            p.y = y
            p.z = z
            p.vp_x = vp_x
            p.vp_y = vp_y
            p.vp_z = vp_z
            i += 1

    def to_list(self):
        return self.to_array().tolist()

    def __getitem__(self, Py_ssize_t index):
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("point index out of range")
        cdef PointWithViewpoint* p = &(deref(self.ptr())[<size_t> index])
        return p.x, p.y, p.z, p.vp_x, p.vp_y, p.vp_z

    def __reduce__(self):
        return type(self), (self.to_list(),)

    def _from_pcd_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPCDFile[PointWithViewpoint](s, deref(c))
        return error

    def _to_pcd_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePCDFile[PointWithViewpoint](s, deref(c), binary)
        return error

    def _from_ply_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadPLYFile[PointWithViewpoint](s, deref(c))
        return error

    def _from_obj_file(self, path):
        cdef string s = _topath(path)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef int error
        with nogil:
            error = loadOBJFile[PointWithViewpoint](s, deref(c))
        return error

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointWithViewpoint](s, deref(c), binary)
        return error

    def to_xyz_cloud(self):
        """Return the positions alone as a :class:`PointCloud`."""
        cdef PointCloud result = PointCloud()
        cdef Py_ssize_t n = <Py_ssize_t> self.ptr().size()
        result.resize(n)
        cdef cPointCloud[PointWithViewpoint]* c = self.ptr()
        cdef cPointCloud[PointXYZ]* xyz = result.ptr()
        cdef PointWithViewpoint* p
        cdef PointXYZ* q
        cdef Py_ssize_t i
        for i in range(n):
            p = &(deref(c)[<size_t> i])
            q = &(deref(xyz)[<size_t> i])
            q.x = p.x
            q.y = p.y
            q.z = p.z
        return result


cdef PointCloud_Normal wrap_normal_cloud(shared_ptr[cPointCloud[Normal]] ptr):
    """Adopt an existing pcl::PointCloud<Normal> without copying."""
    cdef PointCloud_Normal pc = PointCloud_Normal()
    pc.thisptr_shared = ptr
    return pc


cdef PointCloud_PointXYZRGBA wrap_xyzrgba_cloud(
        shared_ptr[cPointCloud[PointXYZRGBA]] ptr):
    """Adopt an existing pcl::PointCloud<PointXYZRGBA> without copying."""
    cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
    pc.thisptr_shared = ptr
    return pc


cdef PointCloud_PointXYZRGB wrap_xyzrgb_cloud(
        shared_ptr[cPointCloud[PointXYZRGB]] ptr):
    """Adopt an existing pcl::PointCloud<PointXYZRGB> without copying —
    how RegionGrowing's coloured cloud comes back."""
    cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
    pc.thisptr_shared = ptr
    return pc


cdef PointCloud_PointNormal wrap_pointnormal_cloud(
        shared_ptr[cPointCloud[PointNormal]] ptr):
    """Adopt an existing pcl::PointCloud<PointNormal> without copying."""
    cdef PointCloud_PointNormal pc = PointCloud_PointNormal()
    pc.thisptr_shared = ptr
    return pc


cdef _fill_transform(matrix, float* buf):
    """Validate a 4x4 transform and flatten it column-major.

    Column-major is Eigen's own storage, and what the shim expects — so
    a matrix that came out of `finalTransformation` or `recognize()`
    goes straight back in without a reordering step.
    """
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    if values.shape != (4, 4):
        raise ValueError(
            "transform must be a (4, 4) matrix, got shape %r"
            % (values.shape,))
    if not np.isfinite(values).all():
        raise ValueError("transform contains non-finite values")

    flat = np.asfortranarray(values).reshape(-1, order="F")
    cdef Py_ssize_t i
    for i in range(16):
        buf[i] = <float> flat[i]

    if not isRigidTransform(buf, 1e-3):
        raise ValueError(
            "transform is not a rigid motion (its 3x3 block must be a "
            "rotation and its last row (0, 0, 0, 1)); PCL would apply a "
            "scaling or reflecting matrix silently")
