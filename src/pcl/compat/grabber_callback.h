// Bridge between PCL's grabber callbacks and a plain C function pointer.
//
// This is the C++ shim rung of the version policy (see
// .claude/rules/pipeline.md): it exposes ONE version-independent name to
// the pxd layer and keeps everything Cython cannot express on this side of
// the boundary.
//
// Two things need hiding:
//
//  * `pcl::Grabber::registerCallback` takes a `std::function<T>` (it took
//    `boost::function` before PCL 1.11). Cython cannot build a
//    std::function from a Python callable, so the shim builds one from a
//    C function pointer plus an opaque user-data pointer -- both of which
//    Cython CAN provide, the first as a `cdef ... noexcept nogil`
//    function and the second as the Python callable itself.
//  * `registerCallback` returns a `boost::signals2::connection`. Nothing
//    above this layer needs to know that type, so CloudCallback owns it
//    and exposes connect/disconnect/connected instead.
//
// OWNERSHIP: PCL hands the slot a `ConstPtr` to a cloud it allocated for
// that frame and drops its own reference once the signal returns, so the
// const_pointer_cast below hands the caller what becomes the sole owner.
// The cast is safe for that reason -- but if several callbacks are
// registered on one grabber they share the frame, so a callback must
// treat the cloud as read-only unless it is the only registration.
//
// THREADING: PCL invokes callbacks on the grabber's own thread. The
// function pointer therefore runs without the GIL held, and the Cython
// side is responsible for acquiring it (see src/pcl/_grabber.pyx).
#pragma once

#include <functional>
#include <memory>

#include <boost/signals2/connection.hpp>

#include <pcl/io/grabber.h>
#include <pcl/io/hdl_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using CloudPtr = std::shared_ptr<Cloud>;

/// Signature of the C callback Cython supplies.
typedef void (*CloudCallbackFn)(CloudPtr cloud, void* user_data);

/// Owns one registration on a grabber's point-cloud signal.
class CloudCallback {
public:
    CloudCallback() = default;

    // Disconnecting on destruction means a dropped wrapper can never leave
    // the grabber calling into freed user data.
    ~CloudCallback() { disconnect(); }

    CloudCallback(const CloudCallback&) = delete;
    CloudCallback& operator=(const CloudCallback&) = delete;

    // One overload per concrete grabber rather than a single
    // `pcl::Grabber*`: that spares Cython an upcast it cannot express
    // safely, and adding a grabber is one more three-line overload.
    void connect(pcl::PCDGrabber<pcl::PointXYZ>* grabber,
                 CloudCallbackFn fn, void* user_data) {
        connectBase(grabber, fn, user_data);
    }

    void connect(pcl::HDLGrabber* grabber,
                 CloudCallbackFn fn, void* user_data) {
        connectBase(grabber, fn, user_data);
    }

    void disconnect() {
        if (connection_.connected()) {
            connection_.disconnect();
        }
    }

    bool connected() const { return connection_.connected(); }

private:
    void connectBase(pcl::Grabber* grabber, CloudCallbackFn fn,
                     void* user_data) {
        disconnect();
        std::function<void(const Cloud::ConstPtr&)> slot =
            [fn, user_data](const Cloud::ConstPtr& cloud) {
                fn(std::const_pointer_cast<Cloud>(cloud), user_data);
            };
        connection_ = grabber->registerCallback(slot);
    }

    boost::signals2::connection connection_;
};

}  // namespace pclcompat
