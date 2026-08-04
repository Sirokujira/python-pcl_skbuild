# Pipeline rules (always loaded)

## The one pipeline (memorize)

```
pxdgen/headers/pcl/*.h   mirror headers (self-contained PCL API stand-ins)
   -> cppast2autopxd     config: pxdgen/pcl_headers.toml
src/pcl/pxd/*.pxd        GENERATED, committed
   -> cimport
src/pcl/*.pyx            hand-written wrappers
   -> cython --cplus -> C++ compile with the REAL PCL headers -> .so/.pyd
```

## Generated files are read-only

- `src/pcl/pxd/*.pxd` are generator output; the settings deny editing them.
  To change a declaration: edit the mirror header and/or
  `pxdgen/pcl_headers.toml`, then run `python pxdgen/generate.py`.
- `python pxdgen/generate.py --check` must pass before any commit that
  touches the mirror headers or generator config
  (`pytest tests/test_pxd_pipeline.py` enforces it).

## Mirror header discipline

- Mirror headers only need NAMES and TYPES to match the real PCL API; the
  real layout comes from the actual PCL headers at C++ compile time
  (`extern_from` points at the `pcl/...` real paths).
- Keep them self-contained (std headers only, no Eigen/Boost) so pxd
  regeneration works on machines without PCL.

## Environment (relative/discovered only)

- PCL is found via CMake `find_package(PCL)`; a custom install is passed
  with the `PCL_ROOT` environment variable — never hard-code a path.
- Runtime tests (`tests/test_pointcloud.py`) auto-skip when the extension
  is not built; "all skipped" after a successful `pip install .` means the
  install went to the wrong interpreter.
- Stale `_skbuild/` cache after a failed rebuild: delete the `_skbuild`
  directory and rebuild.
