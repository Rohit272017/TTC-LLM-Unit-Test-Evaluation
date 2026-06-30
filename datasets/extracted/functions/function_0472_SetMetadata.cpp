#include "tensorstore/internal/metrics/protobuf.h"
#include <stdint.h>
#include <string>
#include <variant>
#include "absl/log/absl_log.h"
namespace tensorstore {
namespace internal_metrics {
namespace {
void SetMetadata(const MetricMetadata& metadata,
                 metrics_proto::Metadata& proto) {
  proto.set_description(metadata.description.data(),
                        metadata.description.size());
}
template <typename T>
void AddFields(const T& metric, metrics_proto::MetricInstance& proto) {
  for (auto& x : metric.fields) proto.add_field(x);
}
void AddValue(const CollectedMetric::Value& metric,
              metrics_proto::MetricInstance& proto) {
  if (std::holds_alternative<std::monostate>(metric.value) &&
      std::holds_alternative<std::monostate>(metric.max_value)) {
    return;
  }
  AddFields(metric, proto);
  if (std::holds_alternative<double>(metric.value) ||
      std::holds_alternative<double>(metric.max_value)) {
    auto* dest = proto.mutable_double_value();
    if (std::holds_alternative<double>(metric.value)) {
      dest->set_value(std::get<double>(metric.value));
    }
    if (std::holds_alternative<double>(metric.max_value)) {
      dest->set_max_value(std::get<double>(metric.max_value));
    }
  } else if (std::holds_alternative<int64_t>(metric.value) ||
             std::holds_alternative<int64_t>(metric.max_value)) {
    auto* dest = proto.mutable_int_value();
    if (std::holds_alternative<int64_t>(metric.value)) {
      dest->set_value(std::get<int64_t>(metric.value));
    }
    if (std::holds_alternative<int64_t>(metric.max_value)) {
      dest->set_max_value(std::get<int64_t>(metric.max_value));
    }
  } else if (std::holds_alternative<std::string>(metric.value)) {
    auto* dest = proto.mutable_string_value();
    dest->set_value(std::get<std::string>(metric.value));
  } else {
    ABSL_LOG(FATAL) << "Unsupported value";
  }
}
void AddHistogram(const CollectedMetric::Histogram& metric,
                  metrics_proto::MetricInstance& proto) {
  AddFields(metric, proto);
  auto* hist = proto.mutable_histogram();
  hist->set_count(metric.count);
  hist->set_mean(metric.mean);
  if (metric.count > 1) {
    hist->set_sum_of_squared_deviation(metric.sum_of_squared_deviation);
  }
  int n_zeros = 0;
  for (auto x : metric.buckets) {
    if (x == 0) {
      n_zeros++;
    } else {
      if (n_zeros > 0) hist->add_bucket(-n_zeros);
      n_zeros = 0;
      hist->add_bucket(x);
    }
  }
}
}  
void CollectedMetricToProto(const CollectedMetric& metric,
                            metrics_proto::Metric& proto) {
  proto.set_metric_name(metric.metric_name.data(), metric.metric_name.size());
  proto.set_tag(metric.tag.data(), metric.tag.size());
  for (auto& x : metric.field_names) {
    proto.add_field_name(x.data(), x.size());
  }
  SetMetadata(metric.metadata, *proto.mutable_metadata());
  for (auto& x : metric.values) {
    AddValue(x, *proto.add_instance());
  }
  for (auto& x : metric.histograms) {
    AddHistogram(x, *proto.add_instance());
  }
}
void CollectedMetricToProtoCollection(span<const CollectedMetric> metrics,
                                      metrics_proto::MetricCollection& proto) {
  for (auto& metric : metrics) {
    CollectedMetricToProto(metric, *proto.add_metric());
  }
}
void SortProtoCollection(metrics_proto::MetricCollection& proto) {
  std::sort(
      proto.mutable_metric()->pointer_begin(),
      proto.mutable_metric()->pointer_end(),
      [](const metrics_proto::Metric* p1, const metrics_proto::Metric* p2) {
        return p1->metric_name() < p2->metric_name();
      });
  for (int i = 0; i < proto.metric_size(); i++) {
    auto& metric = *proto.mutable_metric(i);
    std::sort(
        metric.mutable_instance()->pointer_begin(),
        metric.mutable_instance()->pointer_end(),
        [](const metrics_proto::MetricInstance* p1,
           const metrics_proto::MetricInstance* p2) {
          int n = std::min(p1->field_size(), p2->field_size());
          for (int i = 0; i < n; i++) {
            if (p1->field(i) != p2->field(i)) {
              return p1->field(i) < p2->field(i);
            }
          }
          return std::make_tuple(p1->field_size(), p1->has_int_value(),
                                 p1->has_double_value(), p1->has_string_value(),
                                 p1->has_histogram(),
                                 reinterpret_cast<uintptr_t>(p1)) <
                 std::make_tuple(p2->field_size(), p2->has_int_value(),
                                 p2->has_double_value(), p2->has_string_value(),
                                 p2->has_histogram(),
                                 reinterpret_cast<uintptr_t>(p2));
        });
  }
}
}  
}  