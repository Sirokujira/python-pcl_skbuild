// Mirror header: self-contained stand-in for <pcl/io/hdl_grabber.h>
// (Velodyne HDL / VLP LiDAR).
//
// Only the file/pcap constructor is declared. The other real constructor
// takes a `boost::asio::ip::address`, which a mirror header may not name
// (self-contained: std headers only) and which Cython has no way to build
// -- a live-network grabber therefore needs a shim of its own before it
// can be wrapped. Replaying a .pcap capture exercises the same signal.
//
// See filters/voxel_grid.h for why the inherited Grabber methods are
// re-declared here instead of modelling the base class.
#pragma once

#include <string>

namespace pcl {

class HDLGrabber {
public:
    HDLGrabber(const std::string& correctionsFile = "",
               const std::string& pcapFile = "");

    // Inherited entry points (Grabber).
    void start();
    void stop();
    bool isRunning() const;
    std::string getName() const;
    float getFramesPerSecond() const;

    void setMinimumDistanceThreshold(float& minThreshold);
    void setMaximumDistanceThreshold(float& maxThreshold);
    float getMinimumDistanceThreshold();
    float getMaximumDistanceThreshold();
    unsigned char getMaximumNumberOfLasers() const;
};

}  // namespace pcl
