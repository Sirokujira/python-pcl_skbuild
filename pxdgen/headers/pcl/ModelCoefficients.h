// Mirror header: self-contained stand-in for <pcl/ModelCoefficients.h>.
#pragma once

#include <memory>
#include <vector>

namespace pcl {

struct ModelCoefficients {
    std::vector<float> values;

    typedef std::shared_ptr<ModelCoefficients> Ptr;
    typedef std::shared_ptr<const ModelCoefficients> ConstPtr;
};

}  // namespace pcl
