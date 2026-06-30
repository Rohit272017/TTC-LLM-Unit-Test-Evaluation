#ifndef TENSORFLOW_CORE_LIB_MONITORING_GAUGE_H_
#define TENSORFLOW_CORE_LIB_MONITORING_GAUGE_H_
#include "xla/tsl/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
namespace tensorflow {
namespace monitoring {
using tsl::monitoring::Gauge;
using tsl::monitoring::GaugeCell;
}  
}  
#endif  