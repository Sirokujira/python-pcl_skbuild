// C++ shim: PCL's sample-consensus API as Cython can reach it. See
// pxdgen/headers/pcl/compat/sac_models.h for the mirror.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/lmeds.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/mlesac.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/msac.h>
#include <pcl/sample_consensus/prosac.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/rmsac.h>
#include <pcl/sample_consensus/rransac.h>
#include <pcl/sample_consensus/sac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/sac_model_cone.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_normal_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/sample_consensus/sac_model_normal_sphere.h>
#include <pcl/sample_consensus/sac_model_parallel_line.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_stick.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;
using SacModel = pcl::SampleConsensusModel<Point>;
using SacModelPtr = SacModel::Ptr;
using Sac = pcl::SampleConsensus<Point>;
using SacPtr = std::shared_ptr<Sac>;

inline SacModelPtr makeSacModel(int model_type,
                                const std::shared_ptr<Cloud>& cloud) {
    switch (model_type) {
    case pcl::SACMODEL_PLANE:
        return SacModelPtr(new pcl::SampleConsensusModelPlane<Point>(cloud));
    case pcl::SACMODEL_LINE:
        return SacModelPtr(new pcl::SampleConsensusModelLine<Point>(cloud));
    case pcl::SACMODEL_CIRCLE2D:
        return SacModelPtr(
            new pcl::SampleConsensusModelCircle2D<Point>(cloud));
    case pcl::SACMODEL_CIRCLE3D:
        return SacModelPtr(
            new pcl::SampleConsensusModelCircle3D<Point>(cloud));
    case pcl::SACMODEL_SPHERE:
        return SacModelPtr(new pcl::SampleConsensusModelSphere<Point>(cloud));
    case pcl::SACMODEL_STICK:
        return SacModelPtr(new pcl::SampleConsensusModelStick<Point>(cloud));
    case pcl::SACMODEL_CYLINDER:
        return SacModelPtr(
            new pcl::SampleConsensusModelCylinder<Point, pcl::Normal>(cloud));
    case pcl::SACMODEL_CONE:
        return SacModelPtr(
            new pcl::SampleConsensusModelCone<Point, pcl::Normal>(cloud));
    case pcl::SACMODEL_NORMAL_PLANE:
        return SacModelPtr(
            new pcl::SampleConsensusModelNormalPlane<Point, pcl::Normal>(cloud));
    case pcl::SACMODEL_NORMAL_SPHERE:
        return SacModelPtr(
            new pcl::SampleConsensusModelNormalSphere<Point, pcl::Normal>(cloud));
    case pcl::SACMODEL_NORMAL_PARALLEL_PLANE:
        return SacModelPtr(
            new pcl::SampleConsensusModelNormalParallelPlane<Point, pcl::Normal>(cloud));
    case pcl::SACMODEL_PARALLEL_PLANE:
        return SacModelPtr(
            new pcl::SampleConsensusModelParallelPlane<Point>(cloud));
    case pcl::SACMODEL_PERPENDICULAR_PLANE:
        return SacModelPtr(
            new pcl::SampleConsensusModelPerpendicularPlane<Point>(cloud));
    case pcl::SACMODEL_PARALLEL_LINE:
        return SacModelPtr(
            new pcl::SampleConsensusModelParallelLine<Point>(cloud));
    case pcl::SACMODEL_REGISTRATION:
        return SacModelPtr(
            new pcl::SampleConsensusModelRegistration<Point>(cloud));
    default:
        return SacModelPtr();
    }
}

inline bool sacModelIsNull(const SacModelPtr& model) {
    return !model;
}

inline bool sacModelSetNormals(const SacModelPtr& model,
                               const std::shared_ptr<NormalCloud>& normals,
                               double distance_weight) {
    auto with_normals = std::dynamic_pointer_cast<
        pcl::SampleConsensusModelFromNormals<Point, pcl::Normal>>(model);
    if (!with_normals) {
        return false;
    }
    with_normals->setInputNormals(normals);
    with_normals->setNormalDistanceWeight(distance_weight);
    return true;
}

inline bool sacModelSetAxis(const SacModelPtr& model, int model_type,
                            float x, float y, float z) {
    const Eigen::Vector3f axis(x, y, z);
    switch (model_type) {
    case pcl::SACMODEL_CYLINDER:
        std::static_pointer_cast<
            pcl::SampleConsensusModelCylinder<Point, pcl::Normal>>(model)
            ->setAxis(axis);
        return true;
    case pcl::SACMODEL_CONE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelCone<Point, pcl::Normal>>(model)
            ->setAxis(axis);
        return true;
    case pcl::SACMODEL_NORMAL_PARALLEL_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelNormalParallelPlane<Point, pcl::Normal>>(model)
            ->setAxis(axis);
        return true;
    case pcl::SACMODEL_PARALLEL_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelParallelPlane<Point>>(model)
            ->setAxis(axis);
        return true;
    case pcl::SACMODEL_PERPENDICULAR_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelPerpendicularPlane<Point>>(model)
            ->setAxis(axis);
        return true;
    case pcl::SACMODEL_PARALLEL_LINE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelParallelLine<Point>>(model)
            ->setAxis(axis);
        return true;
    default:
        return false;
    }
}

inline bool sacModelSetEpsAngle(const SacModelPtr& model, int model_type,
                                double eps_angle) {
    switch (model_type) {
    case pcl::SACMODEL_CYLINDER:
        std::static_pointer_cast<
            pcl::SampleConsensusModelCylinder<Point, pcl::Normal>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    case pcl::SACMODEL_CONE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelCone<Point, pcl::Normal>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    case pcl::SACMODEL_NORMAL_PARALLEL_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelNormalParallelPlane<Point, pcl::Normal>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    case pcl::SACMODEL_PARALLEL_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelParallelPlane<Point>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    case pcl::SACMODEL_PERPENDICULAR_PLANE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelPerpendicularPlane<Point>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    case pcl::SACMODEL_PARALLEL_LINE:
        std::static_pointer_cast<
            pcl::SampleConsensusModelParallelLine<Point>>(model)
            ->setEpsAngle(eps_angle);
        return true;
    default:
        return false;
    }
}

inline void sacModelSetRadiusLimits(const SacModelPtr& model,
                                    double min_radius, double max_radius) {
    model->setRadiusLimits(min_radius, max_radius);
}

inline bool sacModelSetConeOpeningAngle(const SacModelPtr& model,
                                        double min_angle, double max_angle) {
    auto cone = std::dynamic_pointer_cast<
        pcl::SampleConsensusModelCone<Point, pcl::Normal>>(model);
    if (!cone) {
        return false;
    }
    cone->setMinMaxOpeningAngle(min_angle, max_angle);
    return true;
}

inline bool sacModelSetTarget(const SacModelPtr& model,
                              const std::shared_ptr<Cloud>& target) {
    auto registration = std::dynamic_pointer_cast<
        pcl::SampleConsensusModelRegistration<Point>>(model);
    if (!registration) {
        return false;
    }
    registration->setInputTarget(target);
    return true;
}

inline SacPtr makeSac(int method, const SacModelPtr& model) {
    switch (method) {
    case pcl::SAC_RANSAC:
        return SacPtr(new pcl::RandomSampleConsensus<Point>(model));
    case pcl::SAC_LMEDS:
        return SacPtr(new pcl::LeastMedianSquares<Point>(model));
    case pcl::SAC_MSAC:
        return SacPtr(new pcl::MEstimatorSampleConsensus<Point>(model));
    case pcl::SAC_RRANSAC:
        return SacPtr(new pcl::RandomizedRandomSampleConsensus<Point>(model));
    case pcl::SAC_RMSAC:
        return SacPtr(
            new pcl::RandomizedMEstimatorSampleConsensus<Point>(model));
    case pcl::SAC_MLESAC:
        return SacPtr(new pcl::MaximumLikelihoodSampleConsensus<Point>(model));
    case pcl::SAC_PROSAC:
        return SacPtr(new pcl::ProgressiveSampleConsensus<Point>(model));
    default:
        return SacPtr();
    }
}

inline bool sacIsNull(const SacPtr& sac) {
    return !sac;
}

inline bool sacComputeModel(Sac& sac) {
    return sac.computeModel();
}

inline void sacInliers(Sac& sac, std::vector<int>& out) {
    pcl::Indices inliers;
    sac.getInliers(inliers);
    out.assign(inliers.begin(), inliers.end());
}

inline void sacCoefficients(Sac& sac, std::vector<float>& out) {
    Eigen::VectorXf coefficients;
    sac.getModelCoefficients(coefficients);
    out.assign(coefficients.data(),
               coefficients.data() + coefficients.size());
}

}  // namespace pclcompat
