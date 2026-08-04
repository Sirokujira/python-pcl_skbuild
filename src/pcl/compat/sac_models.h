// Build and run RANSAC from Cython.
//
// Every step of pcl::RandomSampleConsensus's API is something Cython
// cannot express:
//
//  * its constructor takes a `SampleConsensusModel<PointT>::Ptr`, and the
//    concrete models are subclasses — Cython cannot do the
//    derived-to-base shared_ptr conversion that produces one.
//  * `getModelCoefficients` fills an `Eigen::VectorXf`, which a mirror
//    header may not name.
//
// So the model factories return the base-typed handle already, and the
// coefficients come back as a std::vector<float>. The model constants
// are PCL's own SacModel enumerators, so a caller picks a model with the
// same `pcl.SACMODEL_*` value they pass to SACSegmentation.
#pragma once

#include <memory>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_stick.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using Cloud = pcl::PointCloud<Point>;
using SacModel = pcl::SampleConsensusModel<Point>;
using SacModelPtr = SacModel::Ptr;
using Ransac = pcl::RandomSampleConsensus<Point>;

/// Build a model of the type named by a pcl::SacModel enumerator (the
/// same `SACMODEL_*` values SACSegmentation takes). Returns an empty
/// handle for a model this shim does not build, which the wrapper turns
/// into a ValueError naming the ones it does.
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
    default:
        return SacModelPtr();
    }
}

inline bool sacModelIsNull(const SacModelPtr& model) {
    return !model;
}

inline std::shared_ptr<Ransac> makeRansac(const SacModelPtr& model) {
    return std::shared_ptr<Ransac>(new Ransac(model));
}

inline bool ransacComputeModel(Ransac& ransac) {
    return ransac.computeModel();
}

inline void ransacInliers(Ransac& ransac, std::vector<int>& out) {
    pcl::Indices inliers;
    ransac.getInliers(inliers);
    out.assign(inliers.begin(), inliers.end());
}

inline void ransacCoefficients(Ransac& ransac, std::vector<float>& out) {
    Eigen::VectorXf coefficients;
    ransac.getModelCoefficients(coefficients);
    out.assign(coefficients.data(),
               coefficients.data() + coefficients.size());
}

}  // namespace pclcompat
