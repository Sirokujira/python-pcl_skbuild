#!/bin/sh
# Compile the native baseline against the same PCL the extension uses, run
# both sides, and print the comparison.
#
#     sh bench/run.sh [n_points] [repeats]
#
# PCL flags come from pkg-config (or PCL_ROOT when set) — never hard-coded.
set -e

N=${1:-1000000}
R=${2:-5}
HERE=$(dirname "$0")
OUT=${TMPDIR:-/tmp}/pcl_bench

mkdir -p "$OUT"

if [ -n "$PCL_ROOT" ]; then
    PKG_CONFIG_PATH="$PCL_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH"
    export PKG_CONFIG_PATH
fi

PCL_CFLAGS=$(pkg-config --cflags pcl_common pcl_io)
PCL_LIBS=$(pkg-config --libs pcl_common pcl_io)

echo "building native baseline..." >&2
# -O2 matches the extension's default Release flags; both sides therefore
# run the same optimization level on the same PCL build.
${CXX:-c++} -O2 -std=c++14 $PCL_CFLAGS "$HERE/bench.cpp" -o "$OUT/bench_cpp" $PCL_LIBS

echo "running native..." >&2
"$OUT/bench_cpp" "$N" "$R" "$OUT/bench_cpp.pcd" > "$OUT/cpp.tsv"

echo "running python binding..." >&2
${PYTHON:-python3} "$HERE/bench.py" "$N" "$R" "$OUT/bench_py.pcd" > "$OUT/py.tsv"

${PYTHON:-python3} "$HERE/compare.py" "$OUT/cpp.tsv" "$OUT/py.tsv" "$N"
