// SeededHueSegmentation without the DLL export.
//
// pcl::SeededHueSegmentation is declared without PCL_EXPORTS
// (seeded_hue_segmentation.h, class line), so on Windows the symbol
// `SeededHueSegmentation::segment` is compiled into pcl_segmentation.dll
// but never exported, and linking against the import library fails with
// an unresolved external. Linux and macOS export every symbol by default,
// which is why only the Windows build noticed.
//
// The method is non-template and defined in the impl header, which PCL
// itself only pulls in under PCL_NO_PRECOMPILE. Including it here gives
// this translation unit its own definition: on Windows that is the only
// one, on ELF/Mach-O it coexists with the library's (same code, ODR-
// identical). The pxd's extern_from points here so nothing else changes.
#pragma once

#include <pcl/segmentation/seeded_hue_segmentation.h>
#include <pcl/segmentation/impl/seeded_hue_segmentation.hpp>
