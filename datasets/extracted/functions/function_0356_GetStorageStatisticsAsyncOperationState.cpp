#include "tensorstore/internal/storage_statistics.h"
#include <stdint.h>
#include <atomic>
#include <utility>
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/util/future.h"
namespace tensorstore {
namespace internal {
GetStorageStatisticsAsyncOperationState::
    GetStorageStatisticsAsyncOperationState(
        Future<ArrayStorageStatistics>& future,
        const GetArrayStorageStatisticsOptions& options)
    : options(options) {
  auto p = PromiseFuturePair<ArrayStorageStatistics>::Make(std::in_place);
  this->promise = std::move(p.promise);
  future = std::move(p.future);
}
void GetStorageStatisticsAsyncOperationState::MaybeStopEarly() {
  if (options.mask & ArrayStorageStatistics::query_not_stored) {
    if (chunks_present.load() == 0) {
      return;
    }
  }
  if (options.mask & ArrayStorageStatistics::query_fully_stored) {
    if (chunk_missing.load() == false) {
      return;
    }
  }
  SetDeferredResult(promise, ArrayStorageStatistics{});
}
GetStorageStatisticsAsyncOperationState::
    ~GetStorageStatisticsAsyncOperationState() {
  auto& r = promise.raw_result();
  if (!r.ok()) return;
  r->mask = options.mask;
  int64_t num_present = chunks_present.load(std::memory_order_relaxed);
  if (options.mask & ArrayStorageStatistics::query_not_stored) {
    r->not_stored = (num_present == 0);
  }
  if (options.mask & ArrayStorageStatistics::query_fully_stored) {
    r->fully_stored = num_present == total_chunks;
  }
}
}  
}  