#ifndef TENSORFLOW_CORE_LIB_MONITORING_COUNTER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_COUNTER_H_
#include "xla/tsl/lib/monitoring/counter.h"
#ifdef IS_MOBILE_PLATFORM
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#else
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#endif
namespace tensorflow {
namespace monitoring {
using tsl::monitoring::Counter;
using tsl::monitoring::CounterCell;
}  
}  
#endif  