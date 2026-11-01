// C++ shim: the organized, labelled and graph-cut segmenters whose PCL
// API Cython cannot state -- comparators that take shared_ptr-to-const
// clouds, a per-point plane-d vector and a raw distance-map pointer; a
// std::set of labels; PlanarRegion vectors with an Eigen allocator; a
// vector-of-vectors output PCL never sizes; and UnaryClassifier's
// vector of FPFH cloud pointers. Each segmenter is configured and run
// here and its result comes back as flat vectors. See
// pxdgen/headers/pcl/compat/segmentation_args.h for the mirror.
//
// Absorbed upstream bugs (verified in plain C++ against PCL 1.14):
//  * EuclideanClusterComparator::setExcludeLabels keeps ONLY the labels
//    in the set (compare() returns false when a label is NOT found; the
//    1.11 rewrite from vector<bool> inverted it; still so on master).
//    comparatorExcludesLabels() probes which way the installed PCL
//    reads the set, and the right set is passed either way.
//  * extractLabeledEuclideanClusters writes labeled_clusters[label]
//    without ever sizing the vector -- a segfault on an empty output
//    (still so on master). labeledEuclideanClusters() sizes it first.
//  * PlaneCoefficientComparator's constructor stores 2 degrees in
//    radians where compare() expects the cosine, so an unset angular
//    threshold accepts normals up to ~88 degrees apart. The angle is
//    always set here, through the setter that takes the cosine.
//  * GrabCut::setBackgroundPointsIndices marks the GIVEN indices as the
//    possible foreground and everything else as background -- the
//    opposite of its name and comment. grabCut() takes the foreground
//    region and says so.
//  * OrganizedMultiPlaneSegmentation::segmentAndRefine traces each
//    plane's contour from the LAST inlier, assuming raster order, but
//    refine() appends grown pixels out of order; the refined wall of a
//    test scene ended on the image corner and got an empty contour.
//    multiPlaneSegmentAndRefine re-traces from the raster-last inlier.
//
// randomWalker works on any boost graph; randomWalkerSegment builds one
// from an edge list, so the caller never needs a boost type.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/extract_labeled_clusters.h>
#include <pcl/segmentation/grabcut_segmentation.h>
#include <pcl/segmentation/ground_plane_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/plane_refinement_comparator.h>
#include <pcl/segmentation/random_walker.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/unary_classifier.h>

namespace pclcompat {

// --- organized connected components ---------------------------------------

const int COMPARATOR_EUCLIDEAN_CLUSTER = 0;
const int COMPARATOR_PLANE_COEFFICIENT = 1;
const int COMPARATOR_EUCLIDEAN_PLANE_COEFFICIENT = 2;
const int COMPARATOR_EDGE_AWARE_PLANE = 3;
const int COMPARATOR_GROUND_PLANE = 4;
const int COMPARATOR_RGB_PLANE_COEFFICIENT = 5;

/// The label OrganizedConnectedComponentSegmentation gives invalid points.
const unsigned int INVALID_LABEL = std::numeric_limits<unsigned int>::max();

using LabelCloud = pcl::PointCloud<pcl::Label>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;

/// True when the installed PCL's EuclideanClusterComparator treats its
/// "exclude" set as an exclusion; false when it keeps only the labels
/// in it (PCL 1.11 through at least 1.14). Two coincident points with
/// the same label, and that label in the set: an exclusion refuses to
/// join them, the inverted reading joins them.
inline bool comparatorExcludesLabels() {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(2, 1);
    (*cloud)[0] = (*cloud)[1] = pcl::PointXYZ(0.0f, 0.0f, 1.0f);
    auto labels = std::make_shared<LabelCloud>(2, 1);
    (*labels)[0].label = (*labels)[1].label = 0;
    auto set = std::make_shared<std::set<std::uint32_t>>();
    set->insert(0);
    pcl::EuclideanClusterComparator<pcl::PointXYZ, pcl::Label> probe;
    probe.setInputCloud(cloud);
    probe.setDistanceThreshold(1.0f, false);
    probe.setLabels(labels);
    probe.setExcludeLabels(set);
    return !probe.compare(0, 1);
}

template <typename PointT>
inline std::shared_ptr<pcl::Comparator<PointT>> euclideanClusterComparator(
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud,
    const std::vector<float>& params, const std::vector<unsigned int>& labels,
    const std::vector<unsigned int>& exclude) {
    auto comparator =
        std::make_shared<pcl::EuclideanClusterComparator<PointT, pcl::Label>>();
    comparator->setInputCloud(cloud);
    comparator->setDistanceThreshold(params.at(0), params.at(1) != 0.0f);
    if (!labels.empty()) {
        auto label_cloud =
            std::make_shared<LabelCloud>(cloud->width, cloud->height);
        std::set<std::uint32_t> present;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            (*label_cloud)[i].label = labels[i];
            present.insert(labels[i]);
        }
        auto excluded = std::make_shared<std::set<std::uint32_t>>(
            exclude.begin(), exclude.end());
        auto set = excluded;
        if (!comparatorExcludesLabels()) {
            // this PCL keeps only what is in the set: pass the complement
            set = std::make_shared<std::set<std::uint32_t>>();
            for (std::uint32_t label : present) {
                if (excluded->count(label) == 0) {
                    set->insert(label);
                }
            }
        }
        comparator->setLabels(label_cloud);
        comparator->setExcludeLabels(set);
    }
    return comparator;
}

/// d of each point's tangent plane, as OrganizedMultiPlaneSegmentation
/// computes it: p . n.
template <typename PointT>
inline std::shared_ptr<std::vector<float>> planeCoefficientD(
    const pcl::PointCloud<PointT>& cloud, const NormalCloud& normals) {
    auto d = std::make_shared<std::vector<float>>(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        (*d)[i] = cloud[i].getVector3fMap().dot(
            normals[i].getNormalVector3fMap());
    }
    return d;
}

template <typename Comparator, typename PointT>
inline void configurePlaneComparator(
    Comparator& comparator,
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud,
    const std::shared_ptr<NormalCloud>& normals,
    const std::vector<float>& params) {
    comparator.setInputCloud(cloud);
    comparator.setInputNormals(normals);
    auto d = planeCoefficientD(*cloud, *normals);
    comparator.setPlaneCoeffD(d);
    comparator.setAngularThreshold(params.at(0));
    comparator.setDistanceThreshold(params.at(1), params.at(2) != 0.0f);
}

/// RGBPlaneCoefficientComparator reads .r .g .b: only the colour cloud
/// has them, so the XYZ overload refuses instead of failing to compile.
inline std::shared_ptr<pcl::Comparator<pcl::PointXYZ>> rgbPlaneComparator(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>&,
    const std::shared_ptr<NormalCloud>&, const std::vector<float>&) {
    throw std::invalid_argument(
        "RGBPlaneCoefficientComparator needs a PointCloud_PointXYZRGBA");
}

inline std::shared_ptr<pcl::Comparator<pcl::PointXYZRGBA>> rgbPlaneComparator(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    const std::shared_ptr<NormalCloud>& normals,
    const std::vector<float>& params) {
    auto comparator = std::make_shared<
        pcl::RGBPlaneCoefficientComparator<pcl::PointXYZRGBA, pcl::Normal>>();
    configurePlaneComparator(*comparator, cloud, normals, params);
    comparator->setColorThreshold(params.at(3));
    return comparator;
}

/// params by kind (angles in radians, flags as 0 / 1):
///   EUCLIDEAN_CLUSTER        distance, depth_dependent
///   PLANE_COEFFICIENT,
///   EUCLIDEAN_PLANE_COEFFICIENT  angular, distance, depth_dependent
///   EDGE_AWARE_PLANE         ... + curvature, distance_map_threshold,
///                            euclidean_distance
///   GROUND_PLANE             ... + ground_angular, ground normal x y z
///   RGB_PLANE_COEFFICIENT    ... + colour distance
/// `labels` / `exclude` only for EUCLIDEAN_CLUSTER (empty: none);
/// `distance_map` only for EDGE_AWARE_PLANE (one value per point).
/// out_labels: one per point, INVALID_LABEL where PCL assigned none.
template <typename PointT>
inline void organizedConnectedComponents(
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud, int kind,
    const std::vector<float>& params,
    const std::shared_ptr<NormalCloud>& normals,
    const std::vector<unsigned int>& labels,
    const std::vector<unsigned int>& exclude,
    const std::vector<float>& distance_map,
    std::vector<unsigned int>& out_labels,
    std::vector<pcl::PointIndices>& clusters) {
    std::shared_ptr<pcl::Comparator<PointT>> comparator;
    switch (kind) {
    case COMPARATOR_EUCLIDEAN_CLUSTER:
        comparator = euclideanClusterComparator(cloud, params, labels, exclude);
        break;
    case COMPARATOR_PLANE_COEFFICIENT: {
        auto c = std::make_shared<
            pcl::PlaneCoefficientComparator<PointT, pcl::Normal>>();
        configurePlaneComparator(*c, cloud, normals, params);
        comparator = c;
        break;
    }
    case COMPARATOR_EUCLIDEAN_PLANE_COEFFICIENT: {
        auto c = std::make_shared<
            pcl::EuclideanPlaneCoefficientComparator<PointT, pcl::Normal>>();
        configurePlaneComparator(*c, cloud, normals, params);
        comparator = c;
        break;
    }
    case COMPARATOR_EDGE_AWARE_PLANE: {
        auto c = std::make_shared<
            pcl::EdgeAwarePlaneComparator<PointT, pcl::Normal>>();
        configurePlaneComparator(*c, cloud, normals, params);
        c->setCurvatureThreshold(params.at(3));
        c->setDistanceMapThreshold(params.at(4));
        c->setEuclideanDistanceThreshold(params.at(5));
        c->setDistanceMap(distance_map.data());
        comparator = c;
        break;
    }
    case COMPARATOR_GROUND_PLANE: {
        auto c = std::make_shared<
            pcl::GroundPlaneComparator<PointT, pcl::Normal>>();
        configurePlaneComparator(*c, cloud, normals, params);
        c->setGroundAngularThreshold(params.at(3));
        c->setExpectedGroundNormal(
            Eigen::Vector3f(params.at(4), params.at(5), params.at(6)));
        comparator = c;
        break;
    }
    case COMPARATOR_RGB_PLANE_COEFFICIENT:
        comparator = rgbPlaneComparator(cloud, normals, params);
        break;
    default:
        throw std::invalid_argument("unknown comparator kind");
    }
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> segmenter(
        comparator);
    segmenter.setInputCloud(cloud);
    LabelCloud result;
    segmenter.segment(result, clusters);
    out_labels.resize(result.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        out_labels[i] = result[i].label;
    }
}

/// The boundary of the labelled region that `start_index` belongs to.
inline void labeledRegionBoundary(int start_index,
                                  const std::vector<unsigned int>& labels,
                                  unsigned int width, unsigned int height,
                                  std::vector<int>& boundary) {
    auto cloud = std::make_shared<LabelCloud>(width, height);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        (*cloud)[i].label = labels[i];
    }
    pcl::PointIndices indices;
    pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ, pcl::Label>::
        findLabeledRegionBoundary(start_index, cloud, indices);
    boundary.assign(indices.indices.begin(), indices.indices.end());
}

// --- OrganizedMultiPlaneSegmentation::segmentAndRefine ---------------------

using MultiPlane =
    pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label>;

/// segment + refine with a PlaneRefinementComparator at `distance`
/// (point-to-plane, metres; scaled by z^2 when depth_dependent).
/// centroids 3 per plane, covariances 9 per plane (row-major), labels
/// one per point (INVALID_LABEL where none).
inline void multiPlaneSegmentAndRefine(
    MultiPlane& segmenter, float distance, bool depth_dependent,
    std::vector<pcl::ModelCoefficients>& coefficients,
    std::vector<pcl::PointIndices>& inliers,
    std::vector<unsigned int>& labels,
    std::vector<pcl::PointIndices>& boundaries,
    std::vector<float>& centroids, std::vector<float>& covariances) {
    auto refinement = std::make_shared<
        pcl::PlaneRefinementComparator<pcl::PointXYZ, pcl::Normal, pcl::Label>>();
    refinement->setDistanceThreshold(distance, depth_dependent);
    segmenter.setRefinementComparator(refinement);
    std::vector<pcl::PlanarRegion<pcl::PointXYZ>,
                Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZ>>>
        regions;
    auto label_cloud = std::make_shared<LabelCloud>();
    std::vector<pcl::PointIndices> label_indices;
    segmenter.segmentAndRefine(regions, coefficients, inliers, label_cloud,
                               label_indices, boundaries);
    // segmentAndRefine traces each contour from inliers.back(), assuming
    // raster order, but refine() appends the grown pixels out of order:
    // the trace can start inside the region and return nothing. Trace
    // again from the raster-last inlier, which is on the outer edge.
    for (std::size_t i = 0; i < inliers.size(); ++i) {
        const auto& idx = inliers[i].indices;
        if (idx.empty()) {
            continue;
        }
        const int start = *std::max_element(idx.begin(), idx.end());
        pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZ, pcl::Label>::
            findLabeledRegionBoundary(start, label_cloud, boundaries[i]);
    }
    labels.resize(label_cloud->size());
    for (std::size_t i = 0; i < label_cloud->size(); ++i) {
        labels[i] = (*label_cloud)[i].label;
    }
    centroids.clear();
    covariances.clear();
    for (const auto& region : regions) {
        const Eigen::Vector3f c = region.getCentroid();
        const Eigen::Matrix3f m = region.getCovariance();
        for (int i = 0; i < 3; ++i) {
            centroids.push_back(c[i]);
        }
        for (int r = 0; r < 3; ++r) {
            for (int k = 0; k < 3; ++k) {
                covariances.push_back(m(r, k));
            }
        }
    }
}

// --- LabeledEuclideanClusterExtraction -------------------------------------

/// Clusters per label: out_labels[i] is the label of out_clusters[i].
inline void labeledEuclideanClusters(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    const std::vector<unsigned int>& labels, double tolerance,
    int min_size, int max_size, std::vector<unsigned int>& out_labels,
    std::vector<pcl::PointIndices>& out_clusters) {
    auto labelled = std::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
    labelled->resize(cloud.size());
    unsigned int max_label = 0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        pcl::PointXYZL& p = (*labelled)[i];
        p.x = cloud[i].x;
        p.y = cloud[i].y;
        p.z = cloud[i].z;
        p.label = labels[i];
        if (labels[i] > max_label) {
            max_label = labels[i];
        }
    }
    pcl::LabeledEuclideanClusterExtraction<pcl::PointXYZL> extraction;
    extraction.setInputCloud(labelled);
    extraction.setClusterTolerance(tolerance);
    extraction.setMinClusterSize(min_size);
    extraction.setMaxClusterSize(max_size);
    // PCL indexes this by label without sizing it
    std::vector<std::vector<pcl::PointIndices>> per_label(
        static_cast<std::size_t>(max_label) + 1);
    extraction.extract(per_label);
    out_labels.clear();
    out_clusters.clear();
    for (std::size_t label = 0; label < per_label.size(); ++label) {
        for (const auto& cluster : per_label[label]) {
            out_labels.push_back(static_cast<unsigned int>(label));
            out_clusters.push_back(cluster);
        }
    }
}

// --- GrabCut ---------------------------------------------------------------

/// `region` holds the indices that MAY be foreground (GrabCut's
/// rectangle); everything else is fixed as background. Returns the
/// foreground indices.
inline void grabCut(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    const std::vector<int>& region, unsigned int k, float lambda,
    int neighbours, std::vector<int>& foreground) {
    pcl::GrabCut<pcl::PointXYZRGBA> grabcut(k, lambda);
    grabcut.setInputCloud(cloud);
    grabcut.setNumberOfNeighbours(neighbours);
    auto indices = std::make_shared<pcl::PointIndices>();
    indices->indices.assign(region.begin(), region.end());
    grabcut.setBackgroundPointsIndices(indices);
    std::vector<pcl::PointIndices> clusters;
    grabcut.extract(clusters);
    foreground.assign(clusters.at(1).indices.begin(),
                      clusters.at(1).indices.end());
}

// --- UnaryClassifier -------------------------------------------------------

inline void fpfhToFlat(const pcl::PointCloud<pcl::FPFHSignature33>& cloud,
                       std::vector<float>& flat) {
    for (const auto& f : cloud) {
        flat.insert(flat.end(), f.histogram, f.histogram + 33);
    }
}

template <typename PointT>
inline void configureUnary(pcl::UnaryClassifier<PointT>& classifier,
                           unsigned int k, float normal_radius,
                           float fpfh_radius) {
    classifier.setClusterSize(k);
    classifier.setNormalRadiusSearch(normal_radius);
    classifier.setFPFHRadiusSearch(fpfh_radius);
}

/// k FPFH cluster centres (33 values each) of the whole cloud.
inline void unaryTrain(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    unsigned int k, float normal_radius, float fpfh_radius,
    std::vector<float>& centres) {
    pcl::UnaryClassifier<pcl::PointXYZ> classifier;
    configureUnary(classifier, k, normal_radius, fpfh_radius);
    classifier.setInputCloud(cloud);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr out(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    classifier.train(out);
    centres.clear();
    fpfhToFlat(*out, centres);
}

/// k centres per label, in order of first appearance: out_labels[i] is
/// the label of centres[33 k i ...]; out_counts[i] how many centres
/// k-means produced for it.
inline void unaryTrainWithLabels(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    const std::vector<unsigned int>& labels, unsigned int k,
    float normal_radius, float fpfh_radius,
    std::vector<unsigned int>& out_labels, std::vector<int>& out_counts,
    std::vector<float>& centres) {
    auto labelled = std::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
    labelled->resize(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        pcl::PointXYZL& p = (*labelled)[i];
        p.x = cloud[i].x;
        p.y = cloud[i].y;
        p.z = cloud[i].z;
        p.label = labels[i];
    }
    pcl::UnaryClassifier<pcl::PointXYZL> classifier;
    configureUnary(classifier, k, normal_radius, fpfh_radius);
    classifier.setInputCloud(labelled);
    std::vector<pcl::PointCloud<pcl::FPFHSignature33>,
                Eigen::aligned_allocator<pcl::PointCloud<pcl::FPFHSignature33>>>
        per_label;
    classifier.trainWithLabel(per_label);
    // trainWithLabel walks the labels in order of first appearance
    out_labels.clear();
    std::set<unsigned int> seen;
    for (unsigned int label : labels) {
        if (seen.insert(label).second) {
            out_labels.push_back(label);
        }
    }
    out_counts.clear();
    centres.clear();
    for (const auto& one : per_label) {
        out_counts.push_back(static_cast<int>(one.size()));
        fpfhToFlat(one, centres);
    }
}

/// Assign each point to the trained set whose nearest centre (chi-square
/// over FPFH) is within `threshold`: out[i] is that set's position in
/// `centres`, or -1. Every set must hold the same number of centres --
/// PCL strides through them by the first set's size.
inline void unarySegment(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
    const std::vector<float>& centres, int sets, float threshold,
    float normal_radius, float fpfh_radius, std::vector<int>& out) {
    const std::size_t per_set = centres.size() / 33 / sets;
    std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> trained;
    for (int s = 0; s < sets; ++s) {
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr one(
            new pcl::PointCloud<pcl::FPFHSignature33>);
        one->resize(per_set);
        for (std::size_t i = 0; i < per_set; ++i) {
            const float* src = &centres[33 * (s * per_set + i)];
            std::copy(src, src + 33, (*one)[i].histogram);
        }
        trained.push_back(one);
    }
    pcl::UnaryClassifier<pcl::PointXYZ> classifier;
    configureUnary(classifier, static_cast<unsigned int>(per_set),
                   normal_radius, fpfh_radius);
    classifier.setInputCloud(cloud);
    classifier.setTrainedFeatures(trained);
    classifier.setFeatureThreshold(threshold);
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr labelled(
        new pcl::PointCloud<pcl::PointXYZRGBL>);
    classifier.segment(labelled);
    // PCL writes 1 for "no match" and set + 2 otherwise
    out.resize(labelled->size());
    for (std::size_t i = 0; i < labelled->size(); ++i) {
        out[i] = static_cast<int>((*labelled)[i].label) - 2;
    }
}

// --- random walker ---------------------------------------------------------

/// pcl::segmentation::randomWalker (Grady 2006) on an undirected graph of
/// `colours.size()` vertices: `edges` flat (source, target) pairs,
/// `weights` one positive value per edge, `colours` 0 for an unseeded
/// vertex and the seed's colour otherwise. On return `colours` holds
/// every vertex's colour (still 0 for an isolated unseeded vertex),
/// `columns` the colours in potential-column order and `potentials`
/// the vertices x columns probabilities, row-major. False when the
/// linear solve failed.
inline bool randomWalkerSegment(const std::vector<int>& edges,
                                const std::vector<float>& weights,
                                std::vector<unsigned int>& colours,
                                std::vector<unsigned int>& columns,
                                std::vector<float>& potentials) {
    using Graph = boost::adjacency_list<
        boost::vecS, boost::vecS, boost::undirectedS,
        boost::property<boost::vertex_color_t, std::uint32_t,
                        boost::property<boost::vertex_degree_t, float>>,
        boost::property<boost::edge_weight_t, float>>;
    Graph graph(colours.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        boost::add_edge(edges[2 * i], edges[2 * i + 1], weights[i], graph);
    }
    auto colour_map = boost::get(boost::vertex_color, graph);
    for (std::size_t v = 0; v < colours.size(); ++v) {
        colour_map[v] = colours[v];
    }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix;
    std::map<std::uint32_t, std::size_t> colour_to_column;
    const bool solved = pcl::segmentation::randomWalker(
        graph, boost::get(boost::edge_weight, graph), colour_map, matrix,
        colour_to_column);
    for (std::size_t v = 0; v < colours.size(); ++v) {
        colours[v] = colour_map[v];
    }
    columns.assign(colour_to_column.size(), 0);
    for (const auto& item : colour_to_column) {
        columns[item.second] = item.first;
    }
    potentials.resize(static_cast<std::size_t>(matrix.size()));
    for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
        for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
            potentials[r * matrix.cols() + c] = matrix(r, c);
        }
    }
    return solved;
}

}  // namespace pclcompat
