// Mirror header: self-contained stand-in for <pcl/io/pcd_grabber.h>.
//
// PCDGrabber replays a list of .pcd files as a stream, which makes it the
// grabber you can actually test without hardware attached: same Grabber
// interface, same point-cloud signal as a real sensor.
//
// See filters/voxel_grid.h for why the inherited Grabber methods are
// re-declared here instead of modelling the base class.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pcl {

template <typename PointT>
class PCDGrabber {
public:
    PCDGrabber(const std::string& pcd_path,
               float frames_per_second = 0,
               bool repeat = false);
    PCDGrabber(const std::vector<std::string>& pcd_files,
               float frames_per_second = 0,
               bool repeat = false);

    // Inherited entry points (Grabber).
    void start();
    void stop();
    bool isRunning() const;
    std::string getName() const;
    float getFramesPerSecond() const;

    // Publishes the next cloud once; how a trigger-based grabber is driven
    // when frames_per_second is 0.
    void trigger();
    void rewind();
    std::size_t numFrames() const;
};

}  // namespace pcl
