// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/sac_models.h.
//
// SampleConsensusModel and RandomSampleConsensus appear only as opaque
// handles here — nothing above this layer calls a method on either, so
// neither needs a mirror of its own.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>

namespace pclcompat {

std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>> makeSacModel(
    int model_type,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);

bool sacModelIsNull(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model);

std::shared_ptr<pcl::RandomSampleConsensus<pcl::PointXYZ>> makeRansac(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model);

bool ransacComputeModel(pcl::RandomSampleConsensus<pcl::PointXYZ>& ransac);

void ransacInliers(pcl::RandomSampleConsensus<pcl::PointXYZ>& ransac,
                   std::vector<int>& out);

void ransacCoefficients(pcl::RandomSampleConsensus<pcl::PointXYZ>& ransac,
                        std::vector<float>& out);

}  // namespace pclcompat
