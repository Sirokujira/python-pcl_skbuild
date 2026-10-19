// Mirror header: self-contained stand-in for
// <pcl/sample_consensus/model_types.h>. Common subset of the supported
// PCL versions only: SACMODEL_ELLIPSE3D arrived in 1.13 and comes from
// the compat/sac_models.h shim instead.
#pragma once

namespace pcl {

enum SacModel {
    SACMODEL_PLANE,
    SACMODEL_LINE,
    SACMODEL_CIRCLE2D,
    SACMODEL_CIRCLE3D,
    SACMODEL_SPHERE,
    SACMODEL_CYLINDER,
    SACMODEL_CONE,
    SACMODEL_TORUS,
    SACMODEL_PARALLEL_LINE,
    SACMODEL_PERPENDICULAR_PLANE,
    SACMODEL_PARALLEL_LINES,
    SACMODEL_NORMAL_PLANE,
    SACMODEL_NORMAL_SPHERE,
    SACMODEL_REGISTRATION,
    SACMODEL_REGISTRATION_2D,
    SACMODEL_PARALLEL_PLANE,
    SACMODEL_NORMAL_PARALLEL_PLANE,
    SACMODEL_STICK
};

}  // namespace pcl
