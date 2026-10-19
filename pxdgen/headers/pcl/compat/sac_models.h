// Mirror header for src/pcl/compat/sac_models.h: everything about
// PCL's sample-consensus API that a pxd cannot state. The models are
// built by type number into a base-class shared_ptr; the per-family
// settings (normals, axis, radius limits, a registration target) go
// through dynamic casts; the methods are built by SAC_* number into the
// SampleConsensus base; and the coefficients come back from an
// Eigen::VectorXf as a plain vector.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac.h>
#include <pcl/sample_consensus/sac_model.h>

namespace pclcompat {

std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>> makeSacModel(
    int model_type,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud);

bool sacModelIsNull(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model);

bool sacModelSetNormals(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    double distance_weight);

bool sacModelSetAxis(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    int model_type, float x, float y, float z);

bool sacModelSetEpsAngle(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    int model_type, double eps_angle);

void sacModelSetRadiusLimits(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    double min_radius, double max_radius);

bool sacModelSetConeOpeningAngle(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    double min_angle, double max_angle);

bool sacModelSetTarget(
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model,
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& target);

std::shared_ptr<pcl::SampleConsensus<pcl::PointXYZ>> makeSac(
    int method,
    const std::shared_ptr<pcl::SampleConsensusModel<pcl::PointXYZ>>& model);

bool sacIsNull(
    const std::shared_ptr<pcl::SampleConsensus<pcl::PointXYZ>>& sac);

bool sacComputeModel(pcl::SampleConsensus<pcl::PointXYZ>& sac);

void sacInliers(pcl::SampleConsensus<pcl::PointXYZ>& sac,
                std::vector<int>& out);

void sacCoefficients(pcl::SampleConsensus<pcl::PointXYZ>& sac,
                     std::vector<float>& out);

}  // namespace pclcompat
