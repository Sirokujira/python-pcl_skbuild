// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/version_args.h (reached as "pcl/compat/version_args.h"
// because CMake adds src/ to the includes). PCL's version macros as
// constants.
#pragma once

namespace pclcompat {

const int major_version = 1;
const int minor_version = 0;
const int revision_version = 0;

}  // namespace pclcompat
