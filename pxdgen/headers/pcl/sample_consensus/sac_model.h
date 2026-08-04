// Mirror header: self-contained stand-in for
// <pcl/sample_consensus/sac_model.h>.
//
// Opaque on purpose: nothing above this layer calls a method on a model.
// It exists so the RANSAC shim's signatures have a type to name, and so
// the concrete models (plane, sphere, ...) can be built through
// pcl/compat/sac_models.h and handed straight back to PCL.
#pragma once

namespace pcl {

template <typename PointT>
class SampleConsensusModel {};

}  // namespace pcl
