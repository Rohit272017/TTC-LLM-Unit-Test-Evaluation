#include "tensorflow/core/util/work_sharder.h"
#include <algorithm>
#include <functional>
#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"
namespace tensorflow {
namespace {
bool UseEigenParallelFor() {
  static bool result = []() {
    bool result = true;
    if (auto status =
            tsl::ReadBoolFromEnvVar("TF_USE_EIGEN_PARALLEL_FOR_IN_WORK_SHARDER",
                                    true, &result);
        status.ok()) {
      return result;
    }
    return true;
  }();
  return result;
}
}  
 thread_local int per_thread_max_parallelism = 1000000;
void SetPerThreadMaxParallelism(int max_parallelism) {
  CHECK_LE(0, max_parallelism);
  per_thread_max_parallelism = max_parallelism;
}
int GetPerThreadMaxParallelism() { return per_thread_max_parallelism; }
void Shard(int max_parallelism, thread::ThreadPool* workers, int64_t total,
           int64_t cost_per_unit, std::function<void(int64_t, int64_t)> work) {
  CHECK_GE(total, 0);
  if (total == 0) {
    return;
  }
  max_parallelism = std::min(max_parallelism, GetPerThreadMaxParallelism());
  if (max_parallelism <= 1) {
    work(0, total);
    return;
  }
  if (UseEigenParallelFor() && max_parallelism >= workers->NumThreads()) {
    tsl::profiler::TraceMe trace_me([=, num_threads = workers->NumThreads()]() {
      return tsl::profiler::TraceMeEncode("ParallelFor",
                                          {{"cost_per_unit", cost_per_unit},
                                           {"total", total},
                                           {"max_parallelism", max_parallelism},
                                           {"num_threads", num_threads}});
    });
    workers->ParallelFor(total, cost_per_unit, work);
    return;
  }
  Sharder::Do(
      total, cost_per_unit, work,
      [&workers](Sharder::Closure c) { workers->Schedule(c); },
      max_parallelism);
}
void Sharder::Do(int64_t total, int64_t cost_per_unit, const Work& work,
                 const Runner& runner, int max_parallelism) {
  tsl::profiler::TraceMe trace_me([=]() {
    return tsl::profiler::TraceMeEncode("Sharder::Do",
                                        {{"cost_per_unit", cost_per_unit},
                                         {"total", total},
                                         {"max_parallelism", max_parallelism}});
  });
  cost_per_unit = std::max(int64_t{1}, cost_per_unit);
  static const int64_t kMinCostPerShard = 10000;
  const int num_shards =
      std::max<int>(1, std::min(static_cast<int64_t>(max_parallelism),
                                total * cost_per_unit / kMinCostPerShard));
  const int64_t block_size = (total + num_shards - 1) / num_shards;
  CHECK_GT(block_size, 0);  
  if (block_size >= total) {
    work(0, total);
    return;
  }
  const int num_shards_used = (total + block_size - 1) / block_size;
  BlockingCounter counter(num_shards_used - 1);
  for (int64_t start = block_size; start < total; start += block_size) {
    auto limit = std::min(start + block_size, total);
    runner([&work, &counter, start, limit]() {
      work(start, limit);        
      counter.DecrementCount();  
    });
  }
  work(0, std::min(block_size, total));
  counter.Wait();
}
}  