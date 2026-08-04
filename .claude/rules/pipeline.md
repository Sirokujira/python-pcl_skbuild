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
