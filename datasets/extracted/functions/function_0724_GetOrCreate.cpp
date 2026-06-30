#include "xla/stream_executor/executor_cache.h"
#include <memory>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace stream_executor {
ExecutorCache::ExecutorCache() = default;
ExecutorCache::~ExecutorCache() = default;
absl::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    int ordinal, const ExecutorFactory& factory) {
  if (auto fast_result = Get(ordinal); fast_result.ok()) {
    return fast_result;
  }
  VLOG(2) << "building executor";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<StreamExecutor> result, factory());
  auto returned_executor = result.get();
  absl::MutexLock lock(&mutex_);
  cache_.emplace(ordinal, std::move(result));
  return returned_executor;
}
absl::StatusOr<StreamExecutor*> ExecutorCache::Get(int ordinal) {
  absl::ReaderMutexLock lock{&mutex_};
  if (auto it = cache_.find(ordinal); it != cache_.end()) {
    return it->second.get();
  }
  return absl::NotFoundError(
      absl::StrFormat("No executors registered for ordinal %d", ordinal));
}
}  