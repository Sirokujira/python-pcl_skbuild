# Binding overhead vs native C++

`bench.cpp` and `bench.py` run identical workloads — same point counts,
same PCL entry points, same "minimum of R repetitions" estimator — so the
difference between them is the cost of going through the Cython wrapper.

```sh
sh bench/run.sh 1000000 7        # n_points, repeats
```

The C++ baseline is compiled against the same PCL install the extension
was built against (flags from `pkg-config`, or `PCL_ROOT` when set).

## Measured (PCL 1.14.0, gcc 13.3, Python 3.11, 1M points, best of 7)

| operation | C++ | Python | ratio |
|---|---|---|---|
| `from_array` | 1.81 ms | 1.79 ms | 0.99x |
| `to_array` | 1.16 ms | 1.13 ms | 0.97x |
| `save_pcd_binary` | 20.07 ms | 14.95 ms | 0.74x |
| `load_pcd_binary` | 9.72 ms | 10.03 ms | 1.03x |
| `per_point_sum` | 1.11 ms | 89.56 ms | **80.63x** |
| `centroid` | 1.10 ms | 17.29 ms | 15.78x |

Numbers reproduced across three runs; bulk rows vary by ±10% run to run,
`per_point_sum` is stable at 78–81x.

## Reading the table

**Bulk transfers cost nothing.** `from_array` / `to_array` are compiled
Cython loops with the same structure as the C++ loop they are compared
against, so they land within measurement noise of native. There is no
marshalling layer to pay for: the wrapper writes straight into
`PointCloud<PointXYZ>` storage through a typed memoryview.

**File I/O costs nothing.** Both sides call the same
`pcl::io::savePCDFile` / `loadPCDFile` and produce byte-identical 12 MB
files. The save row favours the binding by ~25%; that difference lives
inside PCL's writer, not the wrapper (rebuilding the baseline at `-O3`
changes nothing: 18.8 ms). Treat I/O as 1x — the binding is not on the
critical path.

**Per-point access from Python costs ~85 ns/point.** `cloud[i]` in a
Python loop is 80x slower than the equivalent C++ loop, because each
iteration is a Python-level call returning a freshly built tuple. This is
the interpreter, not the wrapper — no binding design can make it fast.

**`centroid` is a numpy trap, not binding overhead.** The breakdown:

```
to_array                 1.07 ms   <- the binding's share
a.mean(axis=0)          15.09 ms   <- numpy reducing along the strided axis
a.reshape(-1).sum()      0.71 ms   <- same data, contiguous
```

Every axis-0 formulation (`mean(axis=0)`, `sum(axis=0)/n`,
`np.add.reduce`, `a.T.sum(axis=1)`) costs ~15 ms on an `(n, 3)` float32
array, while the contiguous reduction of the same bytes takes 0.71 ms.
The binding hands over the data in 1 ms and numpy spends 15 ms on it.

## Consequences for the API

1. **Every wrapped operation must have a bulk form.** One call that moves
   n points is free; n calls that move one point each cost 85 ns apiece.
   This is why `from_array`/`to_array` exist and why `get_point` in a loop
   is documented as the slow path.
2. **Keep loops in C++.** Where PCL already has the loop
   (`compute3DCentroid`, filters, `KdTree` searches), wrap the PCL call
   rather than exposing points and reducing in Python. The `centroid` row
   is what "reduce in Python" costs even when the transfer is free.
3. **Do not optimize the transfer layer.** It is already at native speed;
   effort spent there buys nothing measurable.
