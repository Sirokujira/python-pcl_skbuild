// Mirror header: self-contained stand-in reproducing the API surface of
// <pcl/point_types.h> for the point types wrapped so far.
//
// Why a mirror and not the real header?  pxd generation must run on machines
// without PCL/Eigen installed (CI, sdist builds), and the generated pxd only
// needs *names and types* — the real memory layout always comes from the
// actual PCL headers at C++ compile time, because the generated pxd says
// `cdef extern from "pcl/point_types.h"`.
//
// Keep this file in sync with the PCL API you wrap.  When a field or type
// is missing here, add it and re-run:  python pxdgen/generate.py
#pragma once

#include <cstdint>

namespace pcl {

struct PointXYZ {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
};

struct PointXYZI {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        struct {
            float intensity;
        };
        float data_c[4];
    };
};

struct PointXYZRGB {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        union {
            struct {
                std::uint8_t b;
                std::uint8_t g;
                std::uint8_t r;
                std::uint8_t a;
            };
            float rgb;
        };
        std::uint32_t rgba;
    };
};

struct PointXYZRGBA {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        union {
            struct {
                std::uint8_t b;
                std::uint8_t g;
                std::uint8_t r;
                std::uint8_t a;
            };
            float rgb;
        };
        std::uint32_t rgba;
    };
};

struct Normal {
    union {
        struct {
            float normal_x;
            float normal_y;
            float normal_z;
        };
        float data_n[4];
    };
    union {
        struct {
            float curvature;
        };
        float data_c[4];
    };
};

struct PointNormal {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        struct {
            float normal_x;
            float normal_y;
            float normal_z;
        };
        float data_n[4];
    };
    union {
        struct {
            float curvature;
        };
        float data_c[4];
    };
};

// A point that remembers where it was seen from (vp_x, vp_y, vp_z).
struct PointWithViewpoint {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        struct {
            float vp_x;
            float vp_y;
            float vp_z;
        };
        float data_c[4];
    };
};


// Principal curvatures (PrincipalCurvaturesEstimation output): the
// direction of the largest curvature and the two magnitudes.
struct PrincipalCurvatures {
    union {
        struct {
            float principal_curvature_x;
            float principal_curvature_y;
            float principal_curvature_z;
        };
        float principal_curvature[3];
    };
    float pc1;
    float pc2;
};

// A 125-bin Point Feature Histogram: 3 angular features x 5 bins, one
// per point (PFHEstimation output).
struct PFHSignature125 {
    float histogram[125];
};

// Radius-based Surface Descriptor (RSDEstimation output): the smallest
// and largest surface radius around a point.
struct PrincipalRadiiRSD {
    float r_min;
    float r_max;
};

// Globally Aligned Spatial Distribution, shape only (GASDEstimation).
struct GASDSignature512 {
    float histogram[512];
};

// GASD with colour histograms (GASDColorEstimation).
struct GASDSignature984 {
    float histogram[984];
};

// Global RSD (GRSDEstimation): one 21-bin descriptor for a whole cloud.
struct GRSDSignature21 {
    float histogram[21];
};

// The three moment invariants of a point's neighbourhood.
struct MomentInvariants {
    float j1;
    float j2;
    float j3;
};

// A keypoint with the scale it was found at (SIFTKeypoint output).
struct PointWithScale {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    union {
        struct {
            float scale;
            float angle;
            float response;
            int octave;
        };
        float data_c[4];
    };
};

// A per-point label (OrganizedMultiPlaneSegmentation output).
struct Label {
    std::uint32_t label;
};

// A point with a label (supervoxel / LCCP output).
struct PointXYZL {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float data[4];
    };
    std::uint32_t label;
};

// A 308-bin Viewpoint Feature Histogram: one descriptor for a whole
// cloud, which is why VFHEstimation's output cloud holds a single point.
struct VFHSignature308 {
    float histogram[308];
};

// One flag per point (BoundaryEstimation output): 1 on the edge of the
// surface, 0 inside it.
struct Boundary {
    std::uint8_t boundary_point;
};

// A 640-bin Ensemble of Shape Functions: one descriptor for a whole
// cloud, computed from point pairs and triples -- no normals needed.
struct ESFSignature640 {
    float histogram[640];
};

// A Fast Point Feature Histogram: 3 angular features x 11 bins, one
// descriptor per point.
struct FPFHSignature33 {
    float histogram[33];
};

// A SHOT descriptor plus the local reference frame it was computed in.
struct SHOT352 {
    float descriptor[352];
    float rf[9];
};

}  // namespace pcl
