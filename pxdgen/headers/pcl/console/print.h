// Mirror header: self-contained stand-in for <pcl/console/print.h>.
//
// Only the verbosity switch: PCL_INFO / PCL_WARN / PCL_DEBUG messages
// (GrabCut's per-iteration counts, CPC's cutting levels, ...) print at
// or below the current level. The print functions themselves stay C++.
#pragma once

namespace pcl {
namespace console {

enum VERBOSITY_LEVEL {
    L_ALWAYS,
    L_ERROR,
    L_WARN,
    L_INFO,
    L_DEBUG,
    L_VERBOSE
};

void setVerbosityLevel(VERBOSITY_LEVEL level);
VERBOSITY_LEVEL getVerbosityLevel();
bool isVerbosityLevelEnabled(VERBOSITY_LEVEL severity);

}  // namespace console
}  // namespace pcl
