// Mirror header: self-contained stand-in for
// <pcl/sample_consensus/method_types.h>. The real header declares these
// as `constexpr int` in namespace pcl.
#pragma once

namespace pcl {

const int SAC_RANSAC = 0;
const int SAC_LMEDS = 1;
const int SAC_MSAC = 2;
const int SAC_RRANSAC = 3;
const int SAC_RMSAC = 4;
const int SAC_MLESAC = 5;
const int SAC_PROSAC = 6;

}  // namespace pcl
