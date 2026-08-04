// Native C++ baseline for bench/bench.py.
//
// Both programs run the SAME workloads on the same point counts so the
// difference is exactly the cost of going through the Python/Cython
// wrapper. Timings are the minimum of R repetitions (least noisy
// estimator for short CPU-bound runs); output is `name<TAB>seconds` so
// bench/compare.py can diff the two runs mechanically.
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

static double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void report(const char* name, double best) {
    std::printf("%s\t%.9f\n", name, best);
}

int main(int argc, char** argv) {
    const std::size_t n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 1000000;
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 5;
    const std::string path = (argc > 3) ? argv[3] : "/tmp/bench_cpp.pcd";

    // Deterministic source data, identical formula to bench.py.
    std::vector<float> src(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        src[i * 3 + 0] = static_cast<float>(i) * 0.001f;
        src[i * 3 + 1] = static_cast<float>(i) * 0.002f;
        src[i * 3 + 2] = static_cast<float>(i) * 0.003f;
    }

    // --- from_array: build a cloud from a flat float buffer ---------------
    Cloud::Ptr cloud(new Cloud);
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        Cloud::Ptr c(new Cloud);
        auto t0 = Clock::now();
        c->width = static_cast<std::uint32_t>(n);
        c->height = 1;
        c->is_dense = true;
        c->points.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            c->points[i].x = src[i * 3 + 0];
            c->points[i].y = src[i * 3 + 1];
            c->points[i].z = src[i * 3 + 2];
        }
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
        cloud = c;
    }
    report("from_array", best);

    // --- to_array: copy the cloud back out into a flat float buffer -------
    best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        std::vector<float> out(n * 3);
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            out[i * 3 + 0] = cloud->points[i].x;
            out[i * 3 + 1] = cloud->points[i].y;
            out[i * 3 + 2] = cloud->points[i].z;
        }
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
    }
    report("to_array", best);

    // --- save (binary PCD) -----------------------------------------------
    best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        pcl::io::savePCDFile(path, *cloud, true);
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
    }
    report("save_pcd_binary", best);

    // --- load (binary PCD) ------------------------------------------------
    best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        Cloud::Ptr c(new Cloud);
        auto t0 = Clock::now();
        pcl::io::loadPCDFile(path, *c);
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
    }
    report("load_pcd_binary", best);

    // --- per-point access: sum every coordinate one point at a time -------
    // The pathological case for a binding: n round trips through the
    // wrapper instead of one bulk transfer.
    best = 1e300;
    double checksum = 0.0;
    for (int r = 0; r < repeats; ++r) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const pcl::PointXYZ& p = cloud->points[i];
            sx += p.x;
            sy += p.y;
            sz += p.z;
        }
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
        checksum = sx + sy + sz;
    }
    report("per_point_sum", best);

    // --- centroid: the same result computed the idiomatic way -------------
    best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sx += cloud->points[i].x;
            sy += cloud->points[i].y;
            sz += cloud->points[i].z;
        }
        volatile double cx = sx / static_cast<double>(n);
        volatile double cy = sy / static_cast<double>(n);
        volatile double cz = sz / static_cast<double>(n);
        (void)cx; (void)cy; (void)cz;
        auto t1 = Clock::now();
        best = std::min(best, elapsed(t0, t1));
    }
    report("centroid", best);

    std::fprintf(stderr, "checksum %.6f  points %zu  repeats %d\n",
                 checksum, n, repeats);
    std::remove(path.c_str());
    return 0;
}
