#ifndef TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_
#define TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_
#include <array>
#include <functional>
#include <string>
#include <vector>
#include "xla/tsl/lib/monitoring/metric_def.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace monitoring {
using tsl::monitoring::MetricDef;
using tsl::monitoring::MetricKind;
using tsl::monitoring::ValueType;
}  
}  
#endif  