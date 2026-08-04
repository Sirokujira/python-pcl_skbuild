// Mirror header: self-contained stand-in for <pcl/Vertices.h>.
//
// One polygon of a mesh, as indices into the cloud it was built from.
#pragma once

#include <memory>
#include <vector>

namespace pcl {

struct Vertices {
    Vertices();

    std::vector<int> vertices;

    typedef std::shared_ptr<Vertices> Ptr;
    typedef std::shared_ptr<const Vertices> ConstPtr;
};

}  // namespace pcl
