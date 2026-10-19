# Pipeline rules (always loaded)

## The one pipeline (memorize)

```
pxdgen/headers/pcl/**/*.h   mirror headers (self-contained PCL API stand-ins)
   -> cppast2autopxd        config: pxdgen/pcl_headers.toml + pxdgen/modules/*.toml
src/pcl/pxd/**/*.pxd        GENERATED, committed
   -> cimport
src/pcl/*.pyx               hand-written wrappers
   -> cython --cplus -> C++ compile with the REAL PCL headers -> .so/.pyd
```

## Two backends, one committed output

- cppast2autopxd has two implementations. The committed pxd are the
  **libclang** backend's output; `generate.py --backend cppast` writes the
  C++ tool's output for the same config instead, so the two can be diffed.
  It does not combine with `--check` — the backends differ cosmetically
  (block-level vs per-function `nogil`, cimport order), so a byte
  comparison is meaningless there.
- Parity is gated by `tests/test_pxd_pipeline.py::test_cppast_backend_parity`:
  every job through the cppast backend must declare the same API surface
  per extern block, skip nothing, and compile under cython with its
  siblings. It auto-skips where the `cppast_autopxd` binary is absent and
  gates where it is present — the same shape as the real-PCL tests.

## Generated files are read-only

- `src/pcl/pxd/**/*.pxd` are generator output; the settings deny editing
  them. To change a declaration: edit the mirror header and/or the config,
  then run `python pxdgen/generate.py`.
- `python pxdgen/generate.py --check` must pass before any commit that
  touches the mirror headers or generator config
  (`pytest tests/test_pxd_pipeline.py` enforces it).

## Layout mirrors PCL's own tree

- `src/pcl/pxd/` has one subpackage per PCL module (`filters/`, `kdtree/`,
  `io/`, `segmentation/`, `sample_consensus/`, ...), matching
  `pxdgen/headers/pcl/`. A flat layout collides: PCL ships
  `kdtree/kdtree.h` **and** `Search/kdtree.h`, `common/common.h` **and**
  `visualization/common/common.h`, and four more such pairs.
- Every pxd directory needs `__init__.py` (so the wheel ships it) **and**
  `__init__.pxd` (so `cimport pcl.pxd.<group>.<module>` resolves). Tests
  check this.
- `generate.py` refuses two headers that write the same output path.

## Config split

- `pxdgen/pcl_headers.toml` — `[generator]` settings, typemap
  substitutions, and headers at PCL's include root.
- `pxdgen/modules/<group>.toml` — `[[headers]]` entries only. Anything
  else in these files is a hard error, so generator settings cannot drift
  between modules. Relative paths resolve against `pxdgen/`, exactly as in
  the base config.

## Mirror header discipline

- Mirror headers only need NAMES and TYPES to match the real PCL API; the
  real layout comes from the actual PCL headers at C++ compile time
  (`extern_from` points at the `pcl/...` real paths).
- Keep them self-contained (std headers only, no Eigen/Boost) so pxd
  regeneration works on machines without PCL.
- **Flatten inherited methods.** A concrete class re-declares the
  `setInputCloud` / `filter` entry points it needs instead of mirroring
  the `PCLBase` -> `Filter` -> `FilterIndices` chain. A pxd only claims a
  name exists on a type; C++ resolves it through the real bases.
- **Drop `const` inside template arguments.** Declare
  `shared_ptr<PointCloud<PointT>>` where PCL takes
  `shared_ptr<const PointCloud<PointT>>`: the implicit conversion makes
  the call compile and Cython handles the const-free form far better.

## PCL version policy: 1.11+

PCL 1.11 is the supported floor, enforced in `CMakeLists.txt` with a
`FATAL_ERROR` so an old PCL fails at configure time with a readable
message instead of inside template instantiation. 1.11 is where
`boost::shared_ptr` became `std::shared_ptr` across the API — the break
that forced python-pcl into per-version copies of every pxd and pxi.

Handle a version difference at the LOWEST rung that works:

1. **Declare the common subset.** Only put names that exist in every
   supported version into the mirror header. A declaration you leave out
   is an API nobody can call — which is the correct outcome for something
   that does not exist everywhere.
2. **Type renames -> typemap substitution.** One `[typemap.substitutions]`
   entry in `pcl_headers.toml`; no mirror header or pxd changes.
3. **Presence/signature differences -> config.** `include` / `exclude` in
   the `[[headers]]` entry, split across per-version module configs. The
   diff stays a few lines instead of a copied file.
4. **Genuinely incompatible -> C++ shim.** A header under `src/pcl/compat/`
   using `PCL_VERSION_COMPARE(>=, 1, 11, 0)` from `pcl/pcl_config.h`,
   exposing ONE version-independent name to the pxd.

Never use Cython `IF` / `DEF` for version branching (deprecated in Cython
3.0), and never duplicate a pxd or pyx per version — that is the
maintenance trap this pipeline exists to avoid.

CI only builds 1.14, so the floor is checked by audit: every PCL header
an `extern_from` or a shim `#include` names was looked up at the
`pcl-1.11.0` tag. Two are newer, both 1.13: `farthest_point_sampling.h`
and `sac_model_ellipse3d.h`. Both sit behind rung 4 —
`pclcompat::FarthestPointSampler` (`filter()` throws before 1.13) and
`SAC_MODEL_ELLIPSE3D` (-1 before 1.13) — and both shims were compiled
with `PCL_VERSION` forced to 1.12 to prove the fallback builds. Re-run
the audit when a batch adds headers; it checks headers, not every
symbol in them.

## C++ shims (src/pcl/compat/)

Shims are the rung-4 escape hatch, and not only for version differences —
also for anything Cython cannot state. They live in `src/pcl/compat/`,
are reached as `pcl/compat/<name>.h` (CMake puts `src/` on the include
path), and get a mirror header like any other.

The rule that makes them work: a shim exposes ONE name Cython can express
and keeps everything else on the C++ side. `grabber_callback.h` is the
worked example — it hides both a `std::function` (unbuildable from a
Python callable) and a `boost::signals2::connection` (a type nothing
above needs), exposing a C function pointer plus an opaque `void*`.

A shim lives in its own namespace (`pclcompat`), so its signatures name
PCL's types with a `pcl::` qualifier the extern block does not own. Those
resolve through `extra_cimports` — cppast2autopxd accepts a qualified
name whose tail is already known to the pxd.

## Callbacks and the GIL

PCL invokes grabber callbacks on its own thread with no GIL held. The
Cython trampoline must be `noexcept nogil` and acquire the GIL itself,
and it must swallow every Python exception: letting one unwind out of a
`noexcept` C function called from a signals2 slot is `std::terminate`,
i.e. the interpreter dies with no traceback. Print and continue instead.

Whatever the C++ side holds a `void*` to must be kept alive by the
wrapper for as long as a callback can fire, and disconnected in
`__dealloc__` BEFORE the grabber is freed.

## Upstream bugs belong in the wrapper, not in the caller's lap

`VFHEstimation::compute()` segfaults when no normals were set, and
`PCDGrabber` with `frames_per_second=0` calls `std::terminate`, both
inside PCL 1.14.0 — reproducible in plain C++, so nothing a binding can catch. The
wrapper raises a `RuntimeError` naming the cause instead of letting the
process abort. Same principle for a silent no-op: PCL treats a directory
path as a single file and returns a zero-frame grabber, so the wrapper
expands directories itself.

`DifferenceOfNormalsEstimation` is the same shape of problem one level
up: it makes `Feature::compute()` **private**, so `computeFeature()` is
the only way in — and `computeFeature()` writes `output[i]` for every
input point without ever resizing `output`, because resizing was
`compute()`'s job. Calling the only public entry point as documented
therefore writes past the end of an empty cloud. The wrapper sizes the
output itself before the call. Verified both ways in plain C++: empty
output segfaults, pre-sized output returns.

`ParticleFilterTracker` is a third: it declares `bool changed_{false}`
and never assigns it — only the OMP subclasses do — while
`computeTracking()` gates both `resample()` and `update()` on that flag.
The plain tracker therefore computes particle weights and throws them
away, and `getResult()` returns the initial pose on every frame. The
wrapper tracks `ParticleFilterOMPTracker` instead. Verified in plain C++
against 1.14: the base reports (0,0,0) for a moving object, the OMP one
follows it.

`RandomizedRandomSampleConsensus` (`SAC_RRANSAC`) is a fourth, and a
hang rather than a crash: its loop runs while `iterations_ < k` with `k`
initialised to the largest double and only lowered once a candidate
passes the pre-test, and the pre-test rejects a candidate as soon as ONE
point of a random 10% subset is an outlier — so with any real outlier
fraction no candidate ever passes and `max_iterations` is never
consulted (read in `impl/rransac.hpp`; a plane with 20% outliers ran
for ten minutes before being killed). `SampleConsensus` and
`Segmentation.set_method_type` refuse the constant with that reason.

Hypothesis verification has a quieter version of the same trap:
`PapazovHV` iterates `complete_models_`, which `addModels()` only fills
when occlusion reasoning is on, so called the obvious way it returns an
EMPTY mask (measured in plain C++ against 1.14; GreedyVerification reads
the visible models and is fine). `hypotheses_args.h` always passes the
models as both. `GlobalHypothesesVerification` is past absorbing: its
`addModel` keeps only the points whose normals are NOT finite (an
inverted check, still on master), so it keeps every hypothesis — proven
by patching that one line in a local copy, which makes it reject the
wrong one. It is listed under "Not wrapped" in the README.

`EarClipping` mis-triangulates most concave polygons (L, U and C shapes,
both windings, jittered or not) while getting every convex one right.
`surface_args.h` clips polygon by polygon and checks that the triangles
tile the polygon, and the wrapper raises on a failure instead of
returning triangles outside it. Same for the octree's ray casting:
`max_voxel_count` is compared against each subtree's own count, so the
result overshoots (3 voxels for a maximum of 2); the list is nearest
first, so `octree_args.h` truncates it to what the limit means.

Several more absorbed in the second surface/search batch, each a silent
wrong answer otherwise: `BilateralUpsampling` never initialises its
camera matrix (the wrapper always sets one, Kinect VGA by default);
`SurfelSmoothing::computeSmoothedCloud` returns nothing at all when the
normals are missing or mis-sized (the wrapper checks first);
`NarfDescriptor`'s `support_size` defaults to -1 (the wrapper requires a
positive one); `OrganizedNeighbor::setInputCloud` returns false for a
non-projective cloud and then answers queries anyway (the wrapper
raises).

The remaining features added their own: `IntensitySpinEstimation` and
`RIFTEstimation` have a fixed-size output type but settable bin counts,
and a product above 20 / 32 writes past every histogram, so the
wrappers take the bins only as a pair whose product fits; `ROPSEstimation`
fixes its rotations at 3 for the same reason. `SHOTColorEstimation`
packs whichever halves are enabled at the front of its 1344 values, so
the wrapper returns exactly that width. `NormalBasedSignatureEstimation`
is not wrapped at all: it divides by zero in its DFT and returns only
NaN (0 of 18,000 values finite in plain C++; same on master). And the
`PPFRGBEstimation` impl calls a function its header never includes,
which `compat/ppfrgb.h` supplies.

The second keypoint/filter batch: `HarrisKeypoint2D` runs the 3-D
`Keypoint::initCompute`, which logs "init failed" and returns nothing
unless a neighbour-search radius or k is set -- a parameter the 2-D
detector never uses -- so the wrapper sets k = 1. `BRISK2DEstimation`
erases border keypoints from the cloud it is given; the wrapper returns
the survivors with the descriptors so the rows still line up.
`HarrisKeypoint6D` is header-only and its OpenMP loops say
`default(none)` without listing what they read, which GCC 9+ rejects
(same on master); it lives in `_keypoints_serial`, built with the
`PCL_WITHOUT_OPENMP` target property (top-level CMakeLists.txt) so the
pragmas are ignored. `convolution_3d.h` uses `pcl::search::KdTree`
without including it; `filter_args.h` includes it first.

Compile-time, too: `OctreePointCloudVoxelCentroid::getVoxelCentroidAtPoint`
in 1.14 calls a dependent-base member without `this->`, which two-phase
lookup rejects, and PCL never instantiates the header-only class itself
so nothing upstream notices. `octree_args.h` subclasses it and does the
lookup correctly; the broken member is never instantiated.

The registration batch: `PyramidFeatureHistogram` is a DENSE array --
every level allocates the product of its per-dimension bin counts, so
FPFH's 33 dimensions need 2^33 counters at two bins an axis (a
`vector::_M_fill_insert` length error in practice) -- and `at()`
indexes it with no bounds check, so a value outside the input range
writes past the end. `compare_pyramid_feature_histograms` takes 1..16
dimensions, caps the level-0 total and refuses out-of-range values.
`CorrespondenceRejectorFeatures` does not compile in 1.14 (it calls a
`getClassName()` its container lacks), the 2-D SAC registration pair
needs a point type with x, y, z AND u, v that PCL does not define, and
`SACMODEL_TORUS` / `PARALLEL_LINES` are enum values with no class: all
four are listed in the README and refused where a constant reaches us.
Two merely surprising ones are documented in the docstrings instead:
`CorrespondenceEstimationOrganizedProjection` truncates the projected
pixel coordinate, and `NormalDistributionsTransform2D` stays at the
initial pose (reporting convergence) on zero-thickness structure,
whose near-singular cell Gaussians give no gradient.

The segmentation batch, each measured before it was absorbed:
`EuclideanClusterComparator::setExcludeLabels` keeps ONLY the labels in
its set (the 1.11 rewrite from `vector<bool>` inverted the test; still
so on master), so `segmentation_args.h` probes which way the installed
PCL reads it -- two coincident points, one label, that label in the
set -- and passes the set or its complement. `extractLabeledEuclideanClusters`
writes `labeled_clusters[label]` without sizing the vector, a segfault
in plain C++ (still on master); the shim sizes it. `PlaneCoefficientComparator`'s
constructor stores its 2-degree default as an angle where `compare()`
wants the cosine, so the shim always goes through the setter.
`GrabCut::setBackgroundPointsIndices` treats its indices as the possible
FOREGROUND, the opposite of its name; the wrapper calls it
`set_ForegroundRegion`. `segmentAndRefine` traces contours from the
last inlier, which `refine()` no longer keeps in raster order (a
refined wall ended on the image corner and got an empty contour); the
shim re-traces from the raster-last one. `pcl::Kmeans` divides by an
empty cluster's size, so `UnaryClassifier` training can return a NaN
centre that then poisons the chi-square search; the wrapper replaces
it with a duplicate centre of the same set. `CrfSegmentation` is not
wrapped: its unary potentials land in the wrong slots and its output
labels were wrong in every C++ run.

The io batch: `saveIFSFile<PointXYZ>` writes `PCLPointCloud2::data`
verbatim, PointXYZ's padding float included, while the reader takes 12
bytes per vertex -- every point after the first comes back shifted
(plain C++; same on master); `io_args.h` writes a packed blob through
`IFSWriter` instead. `ImageGrabberBase::getCloudPCLZF` bounds-checks
with `>` instead of `>=`; the shim checks first. And `ImageGrabber`
projects each indexed frame with that frame's XML but never updates the
intrinsics `getCameraIntrinsics` reports -- documented, not absorbed.
PNG writing goes through VTK inside PCL and `png_io.h` is only installed
when PCL had VTK, so the shim tests `__has_include` rather than the
private `PCL_BUILT_WITH_VTK`.

Not every quirk is worth absorbing, though. PCL's octree treats the FIRST
point of a cloud specially: `voxelSearch` can miss it, and
`getPointIndicesFromNewVoxels` reports index 0 as new even for an
unchanged cloud. Both reproduce in plain C++ against 1.14.0. Nothing here
papers over them — a wrapper that silently dropped index 0 would be lying
about what PCL returned — so the tests document the behaviour instead.
The root cause is where the grid goes: with no `defineBoundingBox`, PCL
aligns it on the first point added, which then sits exactly on a voxel
corner, so anything clustered around that point is split across up to
eight voxels. `OctreePointCloudVoxelCentroid` and
`OctreePointCloudDensity` expose `define_bounding_box` and say so in
their docstrings; the tests place the first point away from the
clusters they measure.

And sometimes the bug is in the *documentation*: `pcl::people::HOG`'s
`descriptor` out-parameter has no stated length, and a short buffer is a
silent heap overflow. The size was measured by sentinel-filling across
six parameter sets — `(h/bin - 2) * (w/bin - 2) * n_orients * 4` — and
`people_args.h` owns the allocation so a caller cannot get it wrong.

When PCL does something unusable, verify it in C++ first (that is what
tells you it is not your bug), then decide in the `.pyx` layer: absorb it
when it would otherwise crash or silently no-op, document it when it is
merely surprising.

## Windows linking (MSVC against conda-forge PCL)

`PCL_EXPORTS` is `__declspec(dllexport)` while PCL builds and EMPTY for
everyone else, so a consumer reaches PCL's exported FUNCTIONS through
the import library but can never link its static DATA (that needs
`dllimport`). Anything that makes our module instantiate PCL inline code
touching such data fails to link on Windows only. The live case:
`RangeImageSpherical` has an inline constructor, so its vtable lands in
our module, dragging in `RangeImage::getSubImage` / `getHalfImage` (not
exported) and inline overrides that read the static lookup tables --
Linux links it, MSVC cannot, and neither could any C++ program outside
PCL. `range_image_args.h` refuses it under `_WIN32`. Before wrapping a
PCL class with an inline constructor, check that every virtual it
inherits is exported. Separately, MSVC needs `/bigobj` for the biggest
generated modules (`src/pcl/CMakeLists.txt` sets it for all).

## Cython traps found the hard way

- **No comprehension or generator over a C++ reference.** Cython builds a
  closure that holds the reference, and the generated scope object
  segfaults on construction. `_polygon_list` in `_surface.pyx` uses plain
  loops for exactly this reason; the crash looked like a PCL bug until a
  backtrace put it in the closure's `tp_new`.
- Every `cdef` method needs a declaration in the type's `.pxd` once one
  exists, internal helpers included.

## Environment (relative/discovered only)

- PCL is found via CMake `find_package(PCL)`; a custom install is passed
  with the `PCL_ROOT` environment variable — never hard-code a path.
- Runtime tests auto-skip when the extension is not built; "all skipped"
  after a successful `pip install .` means the install went to the wrong
  interpreter.
- Stale `_skbuild/` cache after a failed rebuild: delete the `_skbuild`
  directory and rebuild.

## Performance shape (measured, see bench/README.md)

Bulk transfers and PCL calls run at native speed; per-point access from
Python costs ~85 ns/point (80x a C++ loop). So: every wrapped operation
needs a bulk form, and loops belong on the C++ side. Do not spend effort
optimizing the transfer layer — it is already at parity.
