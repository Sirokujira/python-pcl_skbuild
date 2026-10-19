// Mirror header for src/pcl/compat/segmentation_args.h: organized
// connected components with each PCL comparator, multi-plane
// segment-and-refine, labelled euclidean clusters, GrabCut, the FPFH
// unary classifier and the random walker, configured and run in C++
// with flat results.
// The real shim's templates appear here once per point type.
#pragma once

#include <memory>
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

namespace pclcompat {

const int COMPARATOR_EUCLIDEAN_CLUSTER = 0;
const int COMPARATOR_PLANE_COEFFICIENT = 1;
const int COMPARATOR_EUCLIDEAN_PLANE_COEFFICIENT = 2;
const int COMPARATOR_EDGE_AWARE_PLANE = 3;
const int COMPARATOR_GROUND_PLANE = 4;
const int COMPARATOR_RGB_PLANE_COEFFICIENT = 5;

// The label PCL gives a point no component claims.
const unsigned int INVALID_LABEL = 4294967295u;

// Whether the installed PCL reads EuclideanClusterComparator's exclude
// set as an exclusion (false through at least 1.14: it keeps ONLY those).
bool comparatorExcludesLabels();

// Labels (one per point, INVALID_LABEL for none) and components. params
// by kind, labels / exclude and distance_map as in the real shim.
void organizedConnectedComponents(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud, int kind,
    const std::vector<float>& params,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    const std::vector<unsigned int>& labels,
    const std::vector<unsigned int>& exclude,
    const std::vector<float>& distance_map,
    std::vector<unsigned int>& out_labels,
    std::vector<pcl::PointIndices>& clusters);

void organizedConnectedComponents(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    int kind, const std::vector<float>& params,
    const std::shared_ptr<pcl::PointCloud<pcl::Normal>>& normals,
    const std::vector<unsigned int>& labels,
    const std::vector<unsigned int>& exclude,
    const std::vector<float>& distance_map,
    std::vector<unsigned int>& out_labels,
    std::vector<pcl::PointIndices>& clusters);

// The boundary pixels of the region `start_index` belongs to.
void labeledRegionBoundary(int start_index,
                           const std::vector<unsigned int>& labels,
                           unsigned int width, unsigned int height,
                           std::vector<int>& boundary);

// segmentAndRefine with a PlaneRefinementComparator at `distance`.
void multiPlaneSegmentAndRefine(
    pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZ, pcl::Normal, pcl::Label>&
        segmenter,
    float distance, bool depth_dependent,
    std::vector<pcl::ModelCoefficients>& coefficients,
    std::vector<pcl::PointIndices>& inliers,
    std::vector<unsigned int>& labels,
    std::vector<pcl::PointIndices>& boundaries,
    std::vector<float>& centroids, std::vector<float>& covariances);

// out_labels[i] is the label of out_clusters[i].
void labeledEuclideanClusters(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                              const std::vector<unsigned int>& labels,
                              double tolerance, int min_size, int max_size,
                              std::vector<unsigned int>& out_labels,
                              std::vector<pcl::PointIndices>& out_clusters);

// `region`: the indices that may be foreground; returns the foreground.
void grabCut(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
    const std::vector<int>& region, unsigned int k, float lambda,
    int neighbours, std::vector<int>& foreground);

// FPFH k-means centres, 33 floats each.
void unaryTrain(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                unsigned int k, float normal_radius, float fpfh_radius,
                std::vector<float>& centres);

void unaryTrainWithLabels(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                          const std::vector<unsigned int>& labels,
                          unsigned int k, float normal_radius,
                          float fpfh_radius,
                          std::vector<unsigned int>& out_labels,
                          std::vector<int>& out_counts,
                          std::vector<float>& centres);

// out[i]: index of the matching trained set, or -1.
void unarySegment(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud,
                  const std::vector<float>& centres, int sets,
                  float threshold, float normal_radius, float fpfh_radius,
                  std::vector<int>& out);

// randomWalker over an edge list; colours in/out (0: unseeded), columns
// the colour of each potential column, potentials vertices x columns.
bool randomWalkerSegment(const std::vector<int>& edges,
                         const std::vector<float>& weights,
                         std::vector<unsigned int>& colours,
                         std::vector<unsigned int>& columns,
                         std::vector<float>& potentials);

}  // namespace pclcompat
