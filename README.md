# python-pcl_skbuild

PCL ([Point Cloud Library](https://pointclouds.org/)) Python bindings built
with **scikit-build + CMake + Cython**, with the Cython declaration files
(`.pxd`) **auto-generated** by
[cppast2autopxd](https://github.com/sirokujira/cppast2autopxd).

Project layout follows
[cython-scikit-build-template](https://github.com/sirokujira/cython-scikit-build-template).

## How it fits together

```
pxdgen/headers/pcl/*.h        self-contained "mirror" headers describing the
        |                     PCL API surface being wrapped
        v  cppast2autopxd (pxdgen/pcl_headers.toml)
src/pcl/pxd/*.pxd             auto-generated `cdef extern from "pcl/..."`
        |                     declarations (committed, regenerable)
        v  cimport
src/pcl/_pointcloud.pyx       hand-written Cython wrapper classes
        |  cython --cplus  (scikit-build / CMake: add_cython_target)
        v
_pointcloud.cpp  --(C++ compiler + real PCL headers)-->  _pointcloud.so / .pyd
```

Key idea: pxd generation runs against small, dependency-free *mirror headers*
(`pxdgen/headers/`), but the generated declarations reference the **real**
PCL header paths (`cdef extern from "pcl/point_types.h"`). So:

- regenerating pxd files needs no PCL install (works in CI and on any dev box)
- compiling the extension uses the actual PCL headers/libraries, which
  type-check every generated declaration

## Build

Requires PCL development files (e.g. `apt install libpcl-dev`, or set
`PCL_ROOT` for a custom install), CMake >= 3.12, and a C++14 compiler.

```sh
pip install .
```

```python
import pcl

cloud = pcl.PointCloudXYZ.from_list([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
print(cloud.size)      # 2
print(cloud[1])        # (1.0, 2.0, 3.0)
```

The built artifact is a native extension module (`_pointcloud.so` on
Linux/macOS, `_pointcloud.pyd` on Windows) installed into the `pcl` package.

## Regenerating the pxd files

```sh
pip install "git+https://github.com/sirokujira/cppast2autopxd.git"
python pxdgen/generate.py           # rewrites src/pcl/pxd/*.pxd
python pxdgen/generate.py --check   # CI mode: fail if committed pxd is stale
```

To wrap more of PCL:

1. Extend a mirror header in `pxdgen/headers/pcl/` (or add a new one that
   mirrors the corresponding real PCL header path).
2. Add/adjust the `[[headers]]` entry in `pxdgen/pcl_headers.toml`.
3. `python pxdgen/generate.py`
4. Use the new declarations from a `.pyx` module, add it to
   `src/pcl/CMakeLists.txt`, and add tests.

## Tests

```sh
pytest tests/test_pxd_pipeline.py   # no PCL needed: generation + transpile
pip install . && pytest tests/      # full runtime tests (needs PCL)
```

## License

MIT License (see `LICENSE`).
