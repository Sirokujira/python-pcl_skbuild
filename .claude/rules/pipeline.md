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

`PCDGrabber` with `frames_per_second=0` calls `std::terminate` inside PCL
1.14.0 — reproducible in plain C++, so nothing a binding can catch. The
wrapper raises a `RuntimeError` naming the cause instead of letting the
process abort. Same principle for a silent no-op: PCL treats a directory
path as a single file and returns a zero-frame grabber, so the wrapper
expands directories itself.

Not every quirk is worth absorbing, though. PCL's octree treats the FIRST
point of a cloud specially: `voxelSearch` can miss it, and
`getPointIndicesFromNewVoxels` reports index 0 as new even for an
unchanged cloud. Both reproduce in plain C++ against 1.14.0. Nothing here
papers over them — a wrapper that silently dropped index 0 would be lying
about what PCL returned — so the tests document the behaviour instead.

When PCL does something unusable, verify it in C++ first (that is what
tells you it is not your bug), then decide in the `.pyx` layer: absorb it
when it would otherwise crash or silently no-op, document it when it is
merely surprising.

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
