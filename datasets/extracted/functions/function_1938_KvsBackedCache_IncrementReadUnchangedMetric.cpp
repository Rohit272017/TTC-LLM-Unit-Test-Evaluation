#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include <cstdint>
#include <string>
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
namespace tensorstore {
namespace internal {
namespace {
auto& kvs_cache_read = internal_metrics::Counter<int64_t, std::string>::New(
    "/tensorstore/cache/kvs_cache_read", "category",
    internal_metrics::MetricMetadata(
        "Count of kvs_backed_cache reads by category. A large number of "
        "'unchanged' reads indicates that the dataset is relatively "
        "quiescent."));
}
void KvsBackedCache_IncrementReadUnchangedMetric() {
  static auto& cell = kvs_cache_read.GetCell("unchanged");
  cell.Increment();
}
void KvsBackedCache_IncrementReadChangedMetric() {
  static auto& cell = kvs_cache_read.GetCell("changed");
  cell.Increment();
}
void KvsBackedCache_IncrementReadErrorMetric() {
  static auto& cell = kvs_cache_read.GetCell("error");
  cell.Increment();
}
}  
}  