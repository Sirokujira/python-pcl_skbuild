// Mirror header: self-contained stand-in for <pcl/io/debayer.h>.
//
// Bayer (GRBG, as the Kinect delivers it) to interleaved RGB. The line
// steps default to the tight layout.
#pragma once

namespace pcl {
namespace io {

class DeBayer {
public:
    DeBayer();

    void debayerBilinear(const unsigned char* bayer_pixel,
                         unsigned char* rgb_buffer, unsigned int width,
                         unsigned int height) const;
    void debayerEdgeAware(const unsigned char* bayer_pixel,
                          unsigned char* rgb_buffer, unsigned int width,
                          unsigned int height) const;
    void debayerEdgeAwareWeighted(const unsigned char* bayer_pixel,
                                  unsigned char* rgb_buffer,
                                  unsigned int width,
                                  unsigned int height) const;
};

}  // namespace io
}  // namespace pcl
