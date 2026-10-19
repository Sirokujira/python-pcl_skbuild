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

### Windows

CI builds Windows against conda-forge's prebuilt PCL — the same recipe
works locally and avoids the hours-long vcpkg compile:

```powershell
micromamba create -n pcl -c conda-forge pcl eigen flann libboost-devel qhull
$env:PCL_ROOT = "$env:MAMBA_ROOT_PREFIX\envs\pcl\Library"
pip install .
```

Keep `PCL_ROOT` set when *running* too: Python 3.8+ ignores PATH when
resolving an extension module's dependent DLLs, so `pcl/__init__.py`
calls `os.add_dll_directory` on `%PCL_ROOT%\bin` (or each entry of
`PCL_DLL_DIRS`) before the first extension import. A "DLL load failed"
on `import pcl` means neither variable was set.

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

### Meshes

Triangulation and the hulls produce index tuples into a cloud;
`save_mesh` writes that pair out, and `load_mesh` reads it back:

```python
mesher = cloud.make_GreedyProjectionTriangulation()
mesher.set_InputCloud(cloud, normals)
mesher.set_SearchRadius(0.3)
pcl.save_mesh(cloud, mesher.reconstruct(), "surface.ply")

points, triangles = pcl.load_mesh("surface.ply")
```

PLY, OBJ and VTK can be written; PLY and OBJ read back. VTK is
write-only here because PCL's VTK-format *reader* lives in
`pcl/io/vtk_lib_io.h`, which needs the VTK libraries.

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
column-major, so that is the layout PCL already has). Apply it without a
numpy round trip — the same call takes what `recognize()` returns:

```python
aligned = source.transform(transform)      # or pcl.transform_cloud(source, m)
```

A matrix that is not a rigid motion is rejected: PCL would apply a
scaling or reflecting one silently and hand back a distorted cloud.

All four of those *refine* a pose that is already roughly right. When
there is no initial guess at all, align from FPFH descriptors first and
hand the result to ICP:

```python
def describe(cloud):
    normals = cloud.make_NormalEstimation(); normals.set_KSearch(15)
    fpfh = cloud.make_FPFHEstimation()
    fpfh.set_InputNormals(normals.compute_cloud())
    fpfh.set_RadiusSearch(0.12)
    return fpfh.compute()

align = source.make_SampleConsensusPrerejective()
align.set_MaxCorrespondenceDistance(0.05)      # cloud units; no default fits
converged, transform, estimate, fitness = align.align(
    source, describe(source), target, describe(target))

converged, refined, _, _ = icp.icp(source.transform(transform), target)
```

### Wrapped so far

| area | classes |
|---|---|
| core | `PointCloud`, `pcl.load` / `pcl.save` (PCD, PLY, `.gz`; OBJ vertices read-only), `cloud.transform` / `pcl.transform_cloud` |
| point types | `PointCloud_PointXYZI`, `PointCloud_PointXYZRGB`, `PointCloud_PointXYZRGBA`, `PointCloud_Normal`, `PointCloud_PointNormal`, `PointCloud_PointWithViewpoint` (+ `pcl.load_XYZI` / `load_XYZRGB` / `load_XYZRGBA` / `load_PointNormal` / `load_PointWithViewpoint`) |
| filters | `VoxelGridFilter`, `ApproximateVoxelGrid`, `PassThroughFilter`, `StatisticalOutlierRemovalFilter`, `RadiusOutlierRemoval`, `ExtractIndices`, `CropBox`, `ProjectInliers`, `RandomSample`, `UniformSampling`, `FastBilateralFilter`, `CropHull`, `ShadowPoints`, `NormalSpaceSampling`, `SamplingSurfaceNormal`, `GridMinimum`, `LocalMaximum`, `FrustumCulling`, `MedianFilter`, `BilateralFilter`, `ModelOutlierRemoval`, `cloud.remove_nan()`, `FarthestPointSampling`, `CovarianceSampling`, `FastBilateralFilterOMP`, `NormalRefinement`, `VoxelGridLabel`, `VoxelGridCovariance`, `VoxelGridOcclusionEstimation`, `PlaneClipper3D`, `BoxClipper3D`, `Convolution3D` (Gaussian, + RGB), `Convolution` (2-D, + `pcl.BORDERS_POLICY_*`), `pcl.apply_morphological_operator` (+ `pcl.MORPH_*`) |
| keypoints | `HarrisKeypoint3D`, `HarrisKeypoint6D` (RGB, serial), `ISSKeypoint3D`, `SIFTKeypoint` (RGB), `SUSANKeypoint` (RGB), `TrajkovicKeypoint3D`, `SmoothedSurfacesKeypoint`; on organized RGBA images `HarrisKeypoint2D`, `TrajkovicKeypoint2D` (+ `pcl.FOUR_CORNERS` / `EIGHT_CORNERS`), `AgastKeypoint2D`, `BriskKeypoint2D` + `BRISK2DEstimation`; NARF (via `RangeImage.narf_keypoints`) |
| range image | `RangeImage`, `RangeImagePlanar` (from a cloud or a depth image), `RangeImageSpherical` (not on Windows, see below), `RangeImageBorderExtractor` (+ `pcl.BORDER_TRAIT__*`), `NarfDescriptor` (+ `pcl.CAMERA_FRAME` / `LASER_FRAME`), `pcl.bearing_angle_image` / `bearing_angle` |
| conditions | `ConditionAnd`, `ConditionOr` (nestable; field, `TfQuadraticXYZ`, packed RGB / HSI comparisons), `ConditionalRemoval` on XYZ, XYZRGB and XYZRGBA clouds (+ `pcl.CompareOp_*`) |
| console | `pcl.set_verbosity_level` / `get_verbosity_level` (+ `pcl.L_ALWAYS` .. `L_VERBOSE`) |
| sample consensus | `SampleConsensus` (RANSAC, LMedS, MSAC, RMSAC, MLESAC, PROSAC via `pcl.SAC_*`; RRANSAC refused, see below) / `RandomSampleConsensus` + `SampleConsensusModel{Plane,Line,Circle2D,Circle3D,Sphere,Stick,Cylinder,Cone,NormalPlane,NormalSphere,ParallelPlane,PerpendicularPlane,NormalParallelPlane,ParallelLine,Registration,Ellipse3D}` (TORUS / PARALLEL_LINES / REGISTRATION_2D refused, see below) |
| search | `KdTreeFLANN`, `OrganizedNeighbor`, `BruteForceSearch`, `FlannSearch` (single / multi kd-tree, k-means index), `OctreePointCloudSearch` (+ ray casting: `get_intersected_voxel_centers` / `_indices`), `OctreePointCloudChangeDetector`, `OctreePointCloudVoxelCentroid`, `OctreePointCloudDensity`, `OctreePointCloudOccupancy`, `OctreePointCloudPointVector`, `OctreePointCloudSinglePoint`, `OctreePointCloudAdjacency` (voxel graph, occlusion test) |
| compression | `OctreePointCloudCompression` / `OctreePointCloudCompressionRGB` (bytes in, bytes out; the `pcl.*_COMPRESSION_*` profiles), `OrganizedPointCloudCompression` (depth + colour images, organized clouds; only where PCL was built with libpng) |
| features | `NormalEstimation`, `IntegralImageNormalEstimation`, `DifferenceOfNormalsEstimation`, `MomentOfInertiaEstimation`, `VFHEstimation`, `CVFHEstimation`, `ESFEstimation`, `FPFHEstimation`, `PFHEstimation`, `SHOTEstimation`, `BoundaryEstimation`, `PrincipalCurvaturesEstimation`, `RSDEstimation`, `GRSDEstimation`, `SpinImageEstimation`, `GASDEstimation` / `GASDColorEstimation`, `OURCVFHEstimation`, `CRHEstimation`, `MomentInvariantsEstimation`, `NormalEstimationOMP`, `FPFHEstimationOMP`, `SHOTEstimationOMP`, `SHOTColorEstimation` (+ OMP), `PFHRGBEstimation`, `ShapeContext3DEstimation`, `UniqueShapeContext`, `PPFEstimation`, `PPFRGBEstimation`, `CPPFEstimation`, `SHOTLocalReferenceFrameEstimation` (+ OMP), `BOARDLocalReferenceFrameEstimation`, `FLARELocalReferenceFrameEstimation`, `IntensityGradientEstimation`, `IntensitySpinEstimation`, `RIFTEstimation`, `LinearLeastSquaresNormalEstimation`, `OrganizedEdgeBase` / `FromNormals` / `FromRGB` / `FromRGBNormals` (+ `pcl.EDGELABEL_*`), `GFPFHEstimation`, `ROPSEstimation`, `StatisticalMultiscaleInterestRegionExtraction`, `MultiscaleFeaturePersistence` (FPFH, + `pcl.NORM_*`), `pcl.compute_approximate_normals`, `pcl.compute_pair_features` |
| surface | `MovingLeastSquares`, `ConcaveHull`, `ConvexHull`, `GreedyProjectionTriangulation`, `Poisson`, `MarchingCubesHoppe`, `MarchingCubesRBF`, `GridProjection`, `OrganizedFastMesh` (+ `pcl.TRIANGLE_*` / `QUAD_MESH`), `BilateralUpsampling`, `SurfelSmoothing`, `EarClipping`, `SimplificationRemoveUnusedVertices`, `pcl.save_mesh` / `pcl.load_mesh` (PLY, OBJ, VTK) |
| registration | `IterativeClosestPoint`, `IterativeClosestPointNonLinear`, `GeneralizedIterativeClosestPoint`, `NormalDistributionsTransform`, `IterativeClosestPointWithNormals`, `SampleConsensusPrerejective` / `SampleConsensusInitialAlignment` / `FPCSInitialAlignment` / `KFPCSInitialAlignment` (global), and the correspondence pipeline as functions: `pcl.estimate_correspondences`, `reject_one_to_one` / `reject_by_distance` / `reject_median_distance` / `reject_trimmed` / `reject_sample_consensus` / `reject_surface_normal`, `estimate_transformation` (SVD, LM, dual quaternion, SVD with scale, 2-D, three-point), `estimate_transformation_point_to_plane` (LM, linear least squares, weighted, symmetric), `estimate_correspondences_normal_shooting` / `_back_projection` / `_organized_projection`, `reject_var_trimmed` / `reject_poly` / `reject_organized_boundary`, `validate_transformation`; `GeneralizedIterativeClosestPoint6D`, `NormalDistributionsTransform2D`, `JointIterativeClosestPoint`, `PPFRegistration`, `IncrementalRegistration`, `MetaRegistration`, `LUM`, `ELCH`, `pcl.compare_pyramid_feature_histograms` |
| segmentation | `Segmentation` (SAC), `SegmentationNormal`, `EuclideanClusterExtraction`, `ProgressiveMorphologicalFilter`, `MinCutSegmentation`, `ConditionalEuclideanClustering`, `RegionGrowing`, `RegionGrowingRGB`, `ExtractPolygonalPrismData`, `SegmentDifferences`, `ApproximateProgressiveMorphologicalFilter`, `SeededHueSegmentation`, `OrganizedMultiPlaneSegmentation`, `SupervoxelClustering` / `SupervoxelClusteringXYZ` (+ `.lccp()` / `.cpc()` for LCCP / CPC segmentation), `OrganizedMultiPlaneSegmentation.segment_and_refine()`, `OrganizedConnectedComponentSegmentation` with `EuclideanClusterComparator`, `PlaneCoefficientComparator`, `EuclideanPlaneCoefficientComparator`, `EdgeAwarePlaneComparator`, `GroundPlaneComparator`, `RGBPlaneCoefficientComparator`, `LabeledEuclideanClusterExtraction`, `GrabCut`, `UnaryClassifier`, `pcl.random_walker` |
| recognition | `GeometricConsistencyGrouping`, `Hough3DGrouping`, `GreedyVerification`, `PapazovHV`, `pcl.match_descriptors` (FLANN) |
| tracking | `ParticleFilterTracker`, `KLDAdaptiveParticleFilterTracker` (on `PointCloud`, on `PointCloud_PointXYZRGBA` with `add_HSVColorCoherence`, on `PointCloud_PointNormal` with `add_NormalCoherence`; `exact_coherence` for kd-tree instead of voxel pairing), `PyramidalKLTTracker` (pixels across organized colour frames + the rigid motion they imply; PCL's own barely moves a point, so it runs with its gradients and window placement corrected) |
| people | `pcl.hog` / `pcl.hog_descriptor_size` |
| sensors | `PCDGrabber`, `HDLGrabber`, `VLPGrabber` (both replaying a .pcap), `ImageGrabber` (a directory of depth + colour images, PNG / TIFF or .pclzf, read by index) |
| io | `pcl.load` / `pcl.save` for PCD, PLY, OBJ (vertices), IFS and x-y-z text (`pcl.load_ascii` for other separators); `pcl.load_mesh` for PLY, OBJ, IFS; `pcl.save_png`, `pcl.extract_image` (z / rgb / normal / curvature / intensity of an organized cloud), `pcl.label_image`, `pcl.lzf_compress` / `lzf_decompress`, `.pclzf` files (`write_pclzf_depth` / `_rgb` / `_parameters`, `read_pclzf_depth`, `read_pclzf`), `pcl.debayer` |

The built artifacts are native extension modules (`_pointcloud`,
`_filters`, `_kdtree`, `_segmentation`; `.so` on Linux/macOS, `.pyd` on
Windows) installed into the `pcl` package.

### Not wrapped, and why

- **`pcl/visualization`** — needs VTK development headers, which turns a
  `pip install` into a VTK build. Everything here works headless; render
  with whatever the caller already has (matplotlib, Open3D, `pcl_viewer`).
- **The rest of `pcl/people`** — `person_cluster.h` does an unconditional
  `#include <pcl/visualization/pcl_visualizer.h>`, so `PersonCluster`,
  `HeightMap2D`, `HeadBasedSubclustering` and
  `GroundBasedPeopleDetectionApp` all inherit the VTK dependency above.
  `GroundBasedPeopleDetectionApp` additionally needs a trained SVM file
  that this package cannot ship. `HOG` includes only `point_types.h`, so
  it is wrapped.
- **`SAC_RRANSAC`** — PCL 1.14's `RandomizedRandomSampleConsensus` never
  returns on data with outliers: its pre-test rejects a candidate as soon
  as one point of a random 10% subset is an outlier, and the loop bound
  is only lowered once a candidate passes, so `max_iterations` is never
  consulted (reproduced on a plane with 20% outliers; the process had to
  be killed). `SampleConsensus` and `Segmentation.set_method_type` refuse
  it with that reason. `SAC_RMSAC` shares the pre-test but does stop.
- **`GlobalHypothesesVerification`** — keeps every hypothesis it is
  given. Its `addModel` filters the model's points with the NaN check
  inverted (`if (isNormalFinite(...)) continue;`, still so on PCL master),
  keeping only the points whose normal could NOT be estimated, so a
  well-sampled hypothesis is reduced to nothing and every one costs the
  same. Measured in plain C++ against 1.14: a model floating in empty
  space is kept; with only that line fixed, it is rejected.
  `GreedyVerification` and `PapazovHV` are wrapped and get it right.
- **`NormalBasedSignatureEstimation`** — returns only NaN. Its DFT step
  evaluates `cos(2π / (N·k·n))`, a division by zero for the `n = 0`
  term of every coefficient (the arguments are inverted; the DCT step
  has the same slip, and `(k + 1) / N_` is integer division). Measured
  in plain C++ against 1.14: 0 of 18,000 values finite on a sphere with
  exact normals. Still so on PCL master.
- **`CorrespondenceRejectorFeatures`** — does not compile in PCL 1.14
  for any feature type: its nested `FeatureContainer` calls a
  `getClassName()` it does not have (fixed on master).
  `reject_sample_consensus` and `pcl.match_descriptors` cover the
  feature-gated rejection it would add.
- **`CorrespondenceRejectorSampleConsensus2D`, `SACMODEL_REGISTRATION_2D`**
  — they read `.u` and `.v` from points that also need `x, y, z`, and
  PCL defines no point type with both, so there is nothing to
  instantiate them on. `SampleConsensusModel` refuses the constant.
- **`SACMODEL_TORUS`, `SACMODEL_PARALLEL_LINES`** — values of PCL's
  `SacModel` enum with no model class behind them anywhere in PCL.
  `SampleConsensusModel` refuses both with that reason.
- **Live sensors** — `OpenNIGrabber`, `OpenNI2Grabber`, `ONIGrabber`,
  `RealSenseGrabber`, `DinastGrabber`, `EnsensoGrabber`,
  `DavidSDKGrabber`, `FotonicGrabber`, `RobotEyeGrabber`, `TimGrabber`
  and the live-network `HDLGrabber` / `VLPGrabber` constructors each
  need a device (or its vendor SDK at build time), so nothing here could
  test them; `ONIGrabber` replays `.oni` files but only through the
  OpenNI 1 runtime. Recorded data is covered: `.pcap` through
  `HDLGrabber` / `VLPGrabber` (tested on a synthetic VLP-16 capture),
  depth + colour images and `.pclzf` through `ImageGrabber`, PCD
  sequences through `PCDGrabber`.
- **`pcl/io/vtk_lib_io.h`** (`loadPolygonFile`, STL, VTK XML meshes) —
  needs the VTK headers, like `pcl/visualization`; PLY, OBJ, IFS and
  the legacy VTK writer cover meshes without them.
- **`RangeImageSpherical` on Windows** — its constructor is inline, so
  its vtable is built in the caller's module, and that vtable needs
  `RangeImage::getSubImage` / `getHalfImage` (not exported from
  `pcl_common`) plus inline overrides reading `RangeImage`'s static
  lookup tables, which no module outside PCL can link there
  (`PCL_EXPORTS` is empty for consumers, so data never gets `dllimport`).
  Measured: the MSVC link fails on exactly those five symbols. The class
  raises `RuntimeError` on Windows; Linux and macOS have it.
- **`CrfSegmentation`** — returns wrong labels. Its unary potentials
  put each annotated point's energy in slot `labels.size()` (the number
  of distinct labels seen so far) instead of that label's own slot, and
  the output reads `labels[r[i]]` past the end of the label list. Measured
  in plain C++ against 1.14 on two coloured blobs annotated 2 and 5: the
  first came back all 0, the second mostly 0 and partly 2. Unset kernel
  parameters are uninitialised (an Eigen assertion aborts the process in
  `Permutohedral::init`), and it writes progress lines to stdout.
- **`CrfNormalSegmentation`** — `segmentPoints()` has an empty body.
- **`PlanarPolygonFusion`** — `addInputPolygons` only appends to a
  protected vector and nothing reads it back; there is no fusion to call.
  `segment_and_refine()` returns the planar regions it would be fed.
- **`OctreePointCloud2Buf`** (python-pcl's double-buffered octree) — its
  one use, finding the voxels that are new since the previous frame, is
  what `OctreePointCloudChangeDetector` wraps; the second buffer is an
  implementation detail of that. The base `OctreePointCloud` methods
  python-pcl exposes (`is_voxel_occupied_at_point`,
  `delete_voxel_at_point`, `get_occupied_voxel_centers`, ...) live on
  `OctreePointCloudSearch` here.

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
