"""Python binding side of the benchmark; mirrors bench/bench.cpp exactly.

Run both through bench/run.sh, which compiles the C++ baseline against the
same PCL install the extension was built against and diffs the results.

Usage: python bench/bench.py [n_points] [repeats] [pcd_path]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import pcl


def report(name: str, best: float) -> None:
    print(f"{name}\t{best:.9f}")


def bench(fn, repeats: int) -> float:
    """Minimum wall-clock time over `repeats` runs (matches bench.cpp)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 1000000
    repeats = int(argv[2]) if len(argv) > 2 else 5
    path = argv[3] if len(argv) > 3 else "/tmp/bench_py.pcd"

    # Deterministic source data, identical formula to bench.cpp.
    idx = np.arange(n, dtype=np.float32)
    src = np.empty((n, 3), dtype=np.float32)
    src[:, 0] = idx * 0.001
    src[:, 1] = idx * 0.002
    src[:, 2] = idx * 0.003

    cloud = pcl.PointCloud()

    def do_from_array():
        c = pcl.PointCloud()
        c.from_array(src)
        return c

    report("from_array", bench(do_from_array, repeats))
    cloud = do_from_array()

    report("to_array", bench(cloud.to_array, repeats))
    report("save_pcd_binary",
           bench(lambda: pcl.save(cloud, path, binary=True), repeats))
    report("load_pcd_binary", bench(lambda: pcl.load(path), repeats))

    # Per-point access: n round trips through the wrapper. This is the
    # pathological case a binding cannot make fast; it exists to quantify
    # how much the bulk path above is worth.
    def per_point_sum():
        sx = sy = sz = 0.0
        for i in range(n):
            x, y, z = cloud[i]
            sx += x
            sy += y
            sz += z
        return sx + sy + sz

    report("per_point_sum", bench(per_point_sum, repeats))

    # Centroid computed the idiomatic way: one bulk transfer, then numpy.
    def centroid():
        arr = cloud.to_array()
        return arr.mean(axis=0)

    report("centroid", bench(centroid, repeats))

    print(f"points {n}  repeats {repeats}", file=sys.stderr)
    if os.path.exists(path):
        os.remove(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
