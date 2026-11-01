// The PCL version the package was built against, which pcl_config.h
// states only as macros.
#pragma once

#include <pcl/pcl_config.h>

namespace pclcompat {

const int major_version = PCL_MAJOR_VERSION;
const int minor_version = PCL_MINOR_VERSION;
const int revision_version = PCL_REVISION_VERSION;

}  // namespace pclcompat
