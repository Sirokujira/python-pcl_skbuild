// Mirror header: self-contained stand-in for <pcl/io/grabber.h>.
//
// Only the device-control half of the interface is declared. The callback
// half (`registerCallback`, which takes a std::function and returns a
// boost::signals2::connection) is deliberately absent: Cython cannot build
// a std::function from a Python callable, so registration goes through the
// C++ shim instead -- see pcl/compat/grabber_callback.h.
//
// The real methods are pure virtual; a mirror only claims the names and
// types exist, so declaring them as ordinary methods is enough.
#pragma once

#include <string>

namespace pcl {

class Grabber {
public:
    void start();
    void stop();
    bool isRunning() const;
    std::string getName() const;
    float getFramesPerSecond() const;
};

}  // namespace pcl
