// Mirror header: self-contained stand-in for <pcl/io/lzf.h>.
//
// PCL's copy of the LZF codec (behind binary_compressed PCD and the
// .pclzf images). Both return the bytes written, 0 when out_len is too
// small (or the input is corrupt).
#pragma once

namespace pcl {

unsigned int lzfCompress(const void* in_data, unsigned int in_len,
                         void* out_data, unsigned int out_len);
unsigned int lzfDecompress(const void* in_data, unsigned int in_len,
                           void* out_data, unsigned int out_len);

}  // namespace pcl
