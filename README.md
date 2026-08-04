# python-pcl_skbuild

PCL ([Point Cloud Library](https://pointclouds.org/)) Python bindings built
with **scikit-build + CMake + Cython**, with the Cython declaration files
(`.pxd`) **auto-generated** by
[cppast2autopxd](https://github.com/sirokujira/cppast2autopxd).

Project layout follows
[cython-scikit-build-template](https://github.com/sirokujira/cython-scikit-build-template).

## How it fits together

```
pxdgen/headers/pcl/**/*.h     self-contained "mirror" headers describing the
        |                     PCL API surface being wrapped
        v  cppast2autopxd (pxdgen/pcl_headers.toml + pxdgen/modules/*.toml)
src/pcl/pxd/**/*.pxd          auto-generated `cdef extern from "pcl/..."`
        |                     declarations (committed, regenerable)
        v  cimport
src/pcl/*.pyx                 hand-written Cython wrapper classes
        |  cython --cplus  (scikit-build / CMake: add_cython_target)
        v
*.cpp  --(C++ compiler + real PCL headers)-->  _pointcloud.so, _filters.so, ...
```

Key idea: pxd generation runs against small, dependency-free *mirror headers*
(`pxdgen/headers/`), but the generated declarations reference the **real**
PCL header paths (`cdef extern from "pcl/point_types.h"`). So:

- regenerating pxd files needs no PCL install (works in CI and on any dev box)
- compiling the extension uses the actual PCL headers/libraries, which
  type-check every generated declaration

## Build

Requires **PCL 1.11 or newer** development files (e.g.
`apt install libpcl-dev`, or set `PCL_ROOT` for a custom install),
CMake >= 3.12, and a C++14 compiler. Older PCL is rejected at configure
time: 1.11 is where `boost::shared_ptr` became `std::shared_ptr` across
the API, and supporting both would mean per-version declarations — the
duplication this pipeline exists to avoid. For PCL 1.7/1.8 use
[python-pcl](https://github.com/sirokujira/python-pcl).

```sh
pip install .
```

## Usage

The API follows [python-pcl](https://github.com/sirokujira/python-pcl),
including the `cloud.make_*()` factory methods.

```python
import numpy as np
import pcl

cloud = pcl.PointCloud(np.random.rand(1000, 3).astype(np.float32))

vg = cloud.make_voxel_grid_filter()
vg.set_leaf_size(0.1, 0.1, 0.1)
downsampled = vg.filter()

kd = cloud.make_kdtree_flann()
indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, k=5)

seg = cloud.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.01)
inliers, coefficients = seg.segment()

pcl.save(downsampled, "out.pcd", binary=True)
```

### Segmentation, end to end

PCL hands back indices and model coefficients; `ExtractIndices` and
`ProjectInliers` turn them into clouds:

```python
seg = cloud.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.01)
indices, coefficients = seg.segment()

extract = cloud.make_ExtractIndices()
extract.set_indices(indices)
plane = extract.filter()            # the fitted plane
extract.set_negative(True)
rest = extract.filter()             # everything else
```

### Colour

`to_array()` keeps python-pcl's `(n, 4)` layout, whose fourth column is
the packed RGB value reinterpreted as a float — a bit pattern, not a
number, so arithmetic on it is meaningless. The uint8 views are what
colour handling actually wants:

```python
cloud = pcl.load_XYZRGB("scene.pcd")
xyz = cloud.to_xyz_array()      # (n, 3) float32
rgb = cloud.to_rgb_array()      # (n, 3) uint8

cloud.from_rgb_array(xyz, rgb)  # and back
```

Both views read the same union, so neither costs a conversion.

### Sensors

A grabber is PCL's streaming-sensor interface. Register a callback and
PCL delivers a cloud per frame, on its own thread:

```python
import pcl

grabber = pcl.PCDGrabber("captures/", frames_per_second=30, repeat=True)
grabber.register_callback(lambda cloud: print(cloud.size))

with grabber:          # start() / stop()
    time.sleep(5)
```

`pcl.HDLGrabber(corrections_file, pcap_file)` replays a Velodyne
HDL/VLP capture through the same interface. Exceptions raised inside a
handler are printed and the stream continues — letting one escape into
PCL's callback would terminate the interpreter.

### Registration

```python
icp = source.make_IterativeClosestPoint()
converged, transform, estimate, fitness = icp.icp(source, target, max_iter=100)

ndt = source.make_NormalDistributionsTransform()   # not in python-pcl
ndt.set_Resolution(1.0)
ndt.set_StepSize(0.1)
converged, transform, estimate, fitness = ndt.ndt(source, target)
```

`transform` is a 4x4 float32 array in Fortran order (Eigen is
column-major, so that is the layout PCL already has).

### Wrapped so far

| area | classes |
|---|---|
| core | `PointCloud`, `pcl.load` / `pcl.save` (PCD, PLY, `.gz`) |
| point types | `PointCloud_PointXYZI`, `PointCloud_PointXYZRGB`, `PointCloud_PointXYZRGBA`, `PointCloud_Normal` (+ `pcl.load_XYZI` / `load_XYZRGB` / `load_XYZRGBA`) |
| filters | `VoxelGridFilter`, `ApproximateVoxelGrid`, `PassThroughFilter`, `StatisticalOutlierRemovalFilter`, `RadiusOutlierRemoval`, `ExtractIndices`, `CropBox`, `ProjectInliers`, `RandomSample`, `UniformSampling`, `FastBilateralFilter` |
| keypoints | `HarrisKeypoint3D` |
| conditions | `ConditionAnd`, `ConditionalRemoval` (+ `pcl.CompareOp_*`) |
| sample consensus | `RandomSampleConsensus` + `SampleConsensusModel{Plane,Line,Circle2D,Circle3D,Sphere,Stick}` |
| search | `KdTreeFLANN`, `OctreePointCloudSearch`, `OctreePointCloudChangeDetector` |
| features | `NormalEstimation`, `IntegralImageNormalEstimation`, `MomentOfInertiaEstimation`, `VFHEstimation` |
| surface | `MovingLeastSquares`, `ConcaveHull`, `ConvexHull` |
| registration | `IterativeClosestPoint`, `IterativeClosestPointNonLinear`, `GeneralizedIterativeClosestPoint`, `NormalDistributionsTransform` |
| segmentation | `Segmentation` (SAC), `SegmentationNormal`, `EuclideanClusterExtraction`, `ProgressiveMorphologicalFilter`, `MinCutSegmentation`, `ConditionalEuclideanClustering` |
| sensors | `PCDGrabber`, `HDLGrabber` |

The built artifacts are native extension modules (`_pointcloud`,
`_filters`, `_kdtree`, `_segmentation`; `.so` on Linux/macOS, `.pyd` on
Windows) installed into the `pcl` package.

## Performance

Measured against an identical native C++ program (PCL 1.14, 1M points —
run it yourself with `sh bench/run.sh`):

| operation | ratio vs C++ |
|---|---|
| `from_array` / `to_array` | ~1.0x |
| PCD save / load | ~1.0x |
| per-point `cloud[i]` in a Python loop | **80x** |

Bulk transfers and PCL calls run at native speed; only per-point Python
iteration is expensive (~85 ns/point). Use the array and whole-cloud APIs
and let PCL keep the loop — see [`bench/README.md`](bench/README.md).

## Regenerating the pxd files

```sh
pip install "git+https://github.com/sirokujira/cppast2autopxd.git"
python pxdgen/generate.py           # rewrites src/pcl/pxd/**/*.pxd
python pxdgen/generate.py --check   # CI mode: fail if committed pxd is stale
```

`src/pcl/pxd/` mirrors PCL's own directory tree — a flat layout would
collide, since PCL ships both `kdtree/kdtree.h` and `Search/kdtree.h`,
both `common/common.h` and `visualization/common/common.h`, and more.

To wrap more of PCL:

1. Extend a mirror header in `pxdgen/headers/pcl/` (or add a new one that
   mirrors the corresponding real PCL header path).
2. Add the `[[headers]]` entry to `pxdgen/modules/<group>.toml` (these
   files hold `[[headers]]` entries only; generator settings live once in
   `pxdgen/pcl_headers.toml` so they cannot drift).
3. `python pxdgen/generate.py`
4. Use the new declarations from a `.pyx` module, add its name to the list
   in `src/pcl/CMakeLists.txt`, and add tests.

`cppast2autopxd <header> --scaffold` generates a starting-point `.pyx`
(owned pointer, constructor/destructor, primitive-typed methods forwarded,
everything else as TODO comments) so the wrapper layer starts from
working code rather than a blank file.

## Tests

```sh
pytest tests/test_pxd_pipeline.py   # no PCL needed: generation + transpile
pip install . && pytest tests/      # full runtime tests (needs PCL)
```

## License

MIT License (see `LICENSE`).
