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

from pcl.pxd.point_types cimport Normal, PointXYZI, PointXYZRGB, PointXYZRGBA
from pcl.pxd.point_cloud cimport PointCloud as cPointCloud
from pcl.pxd.io.pcd_io cimport loadPCDFile, savePCDFile
from pcl.pxd.io.ply_io cimport loadPLYFile, savePLYFile


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

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZI]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZI](s, deref(c), binary)
        return error


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

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGB]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZRGB](s, deref(c), binary)
        return error


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

    def _to_ply_file(self, path, cbool binary=False):
        cdef string s = _topath(path)
        cdef cPointCloud[PointXYZRGBA]* c = self.ptr()
        cdef int error
        with nogil:
            error = savePLYFile[PointXYZRGBA](s, deref(c), binary)
        return error


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


cdef PointCloud_Normal wrap_normal_cloud(shared_ptr[cPointCloud[Normal]] ptr):
    """Adopt an existing pcl::PointCloud<Normal> without copying."""
    cdef PointCloud_Normal pc = PointCloud_Normal()
    pc.thisptr_shared = ptr
    return pc
