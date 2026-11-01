// Build ConditionalRemoval's condition tree from Cython.
//
// The tree is made of shared_ptrs to abstract bases:
// `ConditionBase::addComparison` takes a `ComparisonBase<PointT>::ConstPtr`,
// `addCondition` a `ConditionBase<PointT>::Ptr`, and
// `ConditionalRemoval::setCondition` a `ConditionBase<PointT>::Ptr`.
// Cython can hold a `shared_ptr<T>` but cannot perform the derived-to-base
// conversion between two of them, and a tree built for one point type
// cannot filter another.
//
// So the Python side keeps the tree as a flat prefix encoding, and
// conditionalRemoval() rebuilds it for the point type of the cloud it is
// filtering. CompareOp is an enum in namespace ComparisonOps; its values
// are re-exported here so they stay tied to the header.
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/filters/conditional_removal.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int COMPARE_GT = pcl::ComparisonOps::GT;
const int COMPARE_GE = pcl::ComparisonOps::GE;
const int COMPARE_LT = pcl::ComparisonOps::LT;
const int COMPARE_LE = pcl::ComparisonOps::LE;
const int COMPARE_EQ = pcl::ComparisonOps::EQ;

// Node kinds of the encoding, in prefix order. Per node:
//   AND / OR      ints: child count
//   FIELD         names: field; ints: op; numbers: value
//   TF_QUADRATIC  ints: op; numbers: A (9, row-major), v (3), c
//                 (keeps p where p'Ap + 2v'p + c <op> 0)
//   PACKED_RGB    names: "r" / "g" / "b"; ints: op; numbers: value
//   PACKED_HSI    names: "h" / "s" / "i"; ints: op; numbers: value
const int CONDITION_AND = 0;
const int CONDITION_OR = 1;
const int COMPARISON_FIELD = 2;
const int COMPARISON_TF_QUADRATIC = 3;
const int COMPARISON_PACKED_RGB = 4;
const int COMPARISON_PACKED_HSI = 5;

struct ConditionReader {
    const std::vector<int>& kinds;
    const std::vector<int>& ints;
    const std::vector<double>& numbers;
    const std::vector<std::string>& names;
    std::size_t kind = 0, i = 0, n = 0, s = 0;

    int nextKind() { return kinds.at(kind++); }
    int nextInt() { return ints.at(i++); }
    double nextNumber() { return numbers.at(n++); }
    const std::string& nextName() { return names.at(s++); }
};

inline pcl::ComparisonOps::CompareOp compareOp(int op) {
    if (op < COMPARE_GT || op > COMPARE_EQ) {
        throw std::invalid_argument("unknown comparison operator");
    }
    return static_cast<pcl::ComparisonOps::CompareOp>(op);
}

template <typename PointT>
inline typename pcl::ConditionBase<PointT>::Ptr readCondition(
    ConditionReader& reader, int kind) {
    typename pcl::ConditionBase<PointT>::Ptr condition;
    if (kind == CONDITION_AND) {
        condition.reset(new pcl::ConditionAnd<PointT>());
    } else if (kind == CONDITION_OR) {
        condition.reset(new pcl::ConditionOr<PointT>());
    } else {
        throw std::invalid_argument("a condition must be AND or OR");
    }
    const int children = reader.nextInt();
    for (int c = 0; c < children; ++c) {
        const int child = reader.nextKind();
        switch (child) {
        case CONDITION_AND:
        case CONDITION_OR:
            condition->addCondition(readCondition<PointT>(reader, child));
            break;
        case COMPARISON_FIELD: {
            const std::string field = reader.nextName();
            const auto op = compareOp(reader.nextInt());
            condition->addComparison(
                typename pcl::FieldComparison<PointT>::ConstPtr(
                    new pcl::FieldComparison<PointT>(field, op,
                                                     reader.nextNumber())));
            break;
        }
        case COMPARISON_TF_QUADRATIC: {
            const auto op = compareOp(reader.nextInt());
            Eigen::Matrix3f a;
            for (int r = 0; r < 3; ++r) {
                for (int k = 0; k < 3; ++k) {
                    a(r, k) = static_cast<float>(reader.nextNumber());
                }
            }
            Eigen::Vector3f v;
            for (int k = 0; k < 3; ++k) {
                v[k] = static_cast<float>(reader.nextNumber());
            }
            const float scalar = static_cast<float>(reader.nextNumber());
            condition->addComparison(
                typename pcl::TfQuadraticXYZComparison<PointT>::ConstPtr(
                    new pcl::TfQuadraticXYZComparison<PointT>(op, a, v,
                                                              scalar)));
            break;
        }
        case COMPARISON_PACKED_RGB: {
            const std::string component = reader.nextName();
            const auto op = compareOp(reader.nextInt());
            condition->addComparison(
                typename pcl::PackedRGBComparison<PointT>::ConstPtr(
                    new pcl::PackedRGBComparison<PointT>(
                        component, op, reader.nextNumber())));
            break;
        }
        case COMPARISON_PACKED_HSI: {
            const std::string component = reader.nextName();
            const auto op = compareOp(reader.nextInt());
            condition->addComparison(
                typename pcl::PackedHSIComparison<PointT>::ConstPtr(
                    new pcl::PackedHSIComparison<PointT>(
                        component, op, reader.nextNumber())));
            break;
        }
        default:
            throw std::invalid_argument("unknown condition node");
        }
    }
    return condition;
}

/// Filter `cloud` through the encoded tree (its root is the first node).
/// With keep_organized, rejected points become `user_value` (NaN by
/// default) instead of being dropped.
template <typename PointT>
inline void conditionalRemoval(
    const std::shared_ptr<pcl::PointCloud<PointT>>& cloud,
    const std::vector<int>& kinds, const std::vector<int>& ints,
    const std::vector<double>& numbers, const std::vector<std::string>& names,
    bool keep_organized, float user_value, pcl::PointCloud<PointT>& out) {
    ConditionReader reader{kinds, ints, numbers, names};
    auto root = readCondition<PointT>(reader, reader.nextKind());
    pcl::ConditionalRemoval<PointT> filter;
    filter.setCondition(root);
    filter.setInputCloud(cloud);
    filter.setKeepOrganized(keep_organized);
    filter.setUserFilterValue(user_value);
    filter.filter(out);
}

}  // namespace pclcompat
