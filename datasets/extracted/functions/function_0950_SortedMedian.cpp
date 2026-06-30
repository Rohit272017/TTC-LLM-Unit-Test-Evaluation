#include "tensorflow/core/grappler/costs/robust_stats.h"
#include <algorithm>
#include <cmath>
#include <utility>
namespace tensorflow {
namespace grappler {
static double SortedMedian(const std::vector<double> &values) {
  const int n = values.size();
  if (n == 0) return 0.0;
  if (n & 1) {
    return values[n / 2];
  } else {
    return (values[n / 2] + values[n / 2 - 1]) / 2.0;
  }
}
static double Median(std::vector<double> &&values) {
  const size_t n = values.size();
  if (n == 0) return 0;
  const auto middle = values.begin() + (n / 2);
  std::nth_element(values.begin(), middle, values.end());
  if (n & 1) {
    return *middle;
  }
  const auto lower_middle = std::max_element(values.begin(), middle);
  if (*lower_middle <= 0 && *middle >= 0) {
    return (*lower_middle + *middle) / 2;
  }
  return *lower_middle + (*middle - *lower_middle) / 2;
}
static std::pair<double, double> ScaledMedianAbsoluteDeviation(
    const std::vector<double> &sorted_values) {
  double median = SortedMedian(sorted_values);
  std::vector<double> deviations;
  deviations.reserve(sorted_values.size());
  for (double d : sorted_values) {
    deviations.push_back(std::abs(d - median));
  }
  double mad = Median(std::move(deviations)) * 1.4826;
  return std::pair<double, double>(median, mad);
}
RobustStats::RobustStats(const std::vector<double> &values)
    : RobustStats(std::vector<double>(values)) {}
RobustStats::RobustStats(std::vector<double> &&values) {
  std::sort(values.begin(), values.end());
  lo_ = values[0];
  hi_ = values.back();
  HuberMAD(values);
}
double UpdateHuberMean(const std::vector<double> &sorted_values, double mean,
                       double margin) {
  int num_within = 0;
  double sum = 0.0;
  for (double d : sorted_values) {
    if (d < mean - margin) {
      sum -= margin;
    } else if (d > mean + margin) {
      sum += margin;
    } else {
      sum += d;
      ++num_within;
    }
  }
  if (num_within > 0) {
    return sum / num_within;
  } else {
    return mean;
  }
}
void RobustStats::HuberMAD(const std::vector<double> &sorted_values) {
  const std::pair<double, double> median_mad =
      ScaledMedianAbsoluteDeviation(sorted_values);
  mean_ = median_mad.first;
  stddev_ = median_mad.second;
  const double c = 1.5;
  const double margin = c * stddev_;
  if (margin > 0.0) {
    for (int k = 0; k < 10; ++k) {
      double old_mean = mean_;
      mean_ = UpdateHuberMean(sorted_values, mean_, margin);
      if (mean_ == old_mean) break;
    }
  }
}
}  
}  