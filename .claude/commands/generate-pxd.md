---
description: Regenerate the auto-generated Cython pxd files from the mirror headers
---

Regenerate the committed pxd files and verify the pipeline:

1. Ensure cppast2autopxd is installed
   (`pip install "git+https://github.com/sirokujira/cppast2autopxd.git"`).
2. Run `python pxdgen/generate.py` — report every warning it prints; warnings
   mean some declaration was skipped and the wrapper may lose API surface.
3. Run `pytest tests/test_pxd_pipeline.py -v` and make it pass.
4. Show a `git diff --stat src/pcl/pxd/` summary of what changed.

Arguments (optional): $ARGUMENTS — if a header/API name is given, first extend
the matching mirror header under `pxdgen/headers/pcl/` and
`pxdgen/pcl_headers.toml`, then do the steps above.
