---
description: Build the native extension (.so/.pyd) and run runtime tests
---

Build the PCL extension module and run the full test suite:

1. Check PCL availability: PCL dev files must be installed (`libpcl-dev` on
   Debian/Ubuntu, or `PCL_ROOT` set). If missing and apt is available,
   install `libpcl-dev`; otherwise report that only the pxd-pipeline tests
   can run here and run those instead.
2. `pip install -v .` — scikit-build + CMake compile
   `src/pcl/_pointcloud.pyx` into `_pointcloud.so` (`.pyd` on Windows).
   On CMake errors, check `find_package(PCL ...)` output first.
3. `pytest tests/ -v` — runtime tests must pass (they auto-skip when the
   extension isn't importable, so "all skipped" means the build didn't
   install correctly).
