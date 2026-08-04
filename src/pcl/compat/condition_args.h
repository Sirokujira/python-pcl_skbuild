// Build ConditionalRemoval's condition tree from Cython.
//
// The tree is made of shared_ptrs to abstract bases:
// `ConditionAnd::addComparison` takes a
// `ComparisonBase<PointT>::ConstPtr` and `ConditionalRemoval::setCondition`
// a `ConditionBase<PointT>::Ptr`. Cython can hold a `shared_ptr<T>` but
// cannot perform the derived-to-base conversion between two of them, so
// every step of building the tree is a conversion it cannot express.
//
// The shim keeps the whole tree on the C++ side: the caller gets an
// opaque handle, adds comparisons to it by name, and hands it to a
// filter. CompareOp is also an enum (in namespace ComparisonOps) whose
// values are re-exported here so they stay tied to the header.
#pragma once

#include <memory>
#include <string>

#include <pcl/filters/conditional_removal.h>
#include <pcl/point_types.h>

namespace pclcompat {

using Point = pcl::PointXYZ;
using ConditionAndXYZ = pcl::ConditionAnd<Point>;
using ConditionalRemovalXYZ = pcl::ConditionalRemoval<Point>;

const int COMPARE_GT = pcl::ComparisonOps::GT;
const int COMPARE_GE = pcl::ComparisonOps::GE;
const int COMPARE_LT = pcl::ComparisonOps::LT;
const int COMPARE_LE = pcl::ComparisonOps::LE;
const int COMPARE_EQ = pcl::ComparisonOps::EQ;

inline std::shared_ptr<ConditionAndXYZ> makeConditionAnd() {
    return std::shared_ptr<ConditionAndXYZ>(new ConditionAndXYZ());
}

/// Add `field <op> value` to the condition, e.g. ("z", COMPARE_LT, 2.0).
inline void addFieldComparison(ConditionAndXYZ& condition,
                               const std::string& field_name,
                               int op, double value) {
    condition.addComparison(
        pcl::FieldComparison<Point>::ConstPtr(new pcl::FieldComparison<Point>(
            field_name, static_cast<pcl::ComparisonOps::CompareOp>(op),
            value)));
}

inline void setCondition(ConditionalRemovalXYZ& filter,
                         std::shared_ptr<ConditionAndXYZ> condition) {
    filter.setCondition(condition);
}

}  // namespace pclcompat
