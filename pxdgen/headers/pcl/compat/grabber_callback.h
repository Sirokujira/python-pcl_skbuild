// Mirror header: self-contained stand-in for the C++ shim that lives at
// src/pcl/compat/grabber_callback.h (which is on the include path as
// "pcl/compat/grabber_callback.h" because CMake adds src/ to the includes).
//
// Unlike the other mirrors, the real header here is OURS rather than
// PCL's. The mirror still exists for the same reason: the real one pulls
// in <boost/signals2/connection.hpp> and PCL's grabber headers, and pxd
// generation has to work on a machine with neither installed.
#pragma once

#include <memory>

#include <pcl/io/hdl_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

// The C callback Cython supplies: a `cdef ... noexcept nogil` function
// plus an opaque pointer to the Python callable.
typedef void (*CloudCallbackFn)(
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud, void* user_data);

class CloudCallback {
public:
    CloudCallback();

    void connect(pcl::PCDGrabber<pcl::PointXYZ>* grabber,
                 CloudCallbackFn fn, void* user_data);
    void connect(pcl::HDLGrabber* grabber,
                 CloudCallbackFn fn, void* user_data);

    void disconnect();
    bool connected() const;
};

}  // namespace pclcompat
