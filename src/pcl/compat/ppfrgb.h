// PPFRGBEstimation, instantiable.
//
// PCL precompiles only PPFRGBRegionEstimation; PPFRGBEstimation's
// members live in the impl header, which the public header includes only
// under PCL_NO_PRECOMPILE. Including it here lets this translation unit
// instantiate the class for the point types the wrapper uses. The pxd's
// extern_from points here.
//
// The impl calls pcl::computeRGBPairFeatures without including
// pfh_tools.h, where it is declared (PCL's own .cpp gets it from another
// include), so that comes first.
#pragma once

#include <pcl/features/pfh_tools.h>
#include <pcl/features/ppfrgb.h>
#include <pcl/features/impl/ppfrgb.hpp>
