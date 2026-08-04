// Mirror header: self-contained stand-in for the C++ shim at
// src/pcl/compat/condition_args.h.
#pragma once

#include <memory>
#include <string>

#include <pcl/filters/conditional_removal.h>
#include <pcl/point_types.h>

namespace pclcompat {

const int COMPARE_GT = 0;
const int COMPARE_GE = 1;
const int COMPARE_LT = 2;
const int COMPARE_LE = 3;
const int COMPARE_EQ = 4;

std::shared_ptr<pcl::ConditionAnd<pcl::PointXYZ>> makeConditionAnd();

void addFieldComparison(pcl::ConditionAnd<pcl::PointXYZ>& condition,
                        const std::string& field_name,
                        int op, double value);

void setCondition(pcl::ConditionalRemoval<pcl::PointXYZ>& filter,
                  std::shared_ptr<pcl::ConditionAnd<pcl::PointXYZ>> condition);

}  // namespace pclcompat
