#include "tensorflow/core/common_runtime/process_util.h"
#if defined(ENABLE_MKL) && defined(ENABLE_ONEDNN_OPENMP)
#ifdef _OPENMP
#include <omp.h>
#endif  
#endif  
#include <string.h>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"
#include "tsl/platform/tracing.h"
namespace tensorflow {
namespace {
int32 GetEnvNumInterOpThreads() {
  static int32_t env_num_threads = NumInterOpThreadsFromEnvironment();
  return env_num_threads;
}
int32 DefaultNumInterOpThreads() {
#ifndef __ANDROID__
  int32_t env_num_threads = GetEnvNumInterOpThreads();
  if (env_num_threads > 0) {
    return env_num_threads;
  }
  return port::MaxParallelism();
#else
  return 1;
#endif
}
static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
  int32_t inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    inter_op_parallelism_threads = DefaultNumInterOpThreads();
  }
  return new thread::ThreadPool(
      Env::Default(), ThreadOptions(), "Compute", inter_op_parallelism_threads,
      !options.config.experimental().disable_thread_spinning(),
      nullptr);
}
}  
thread::ThreadPool* ComputePool(const SessionOptions& options) {
  static thread::ThreadPool* compute_pool = InitComputePool(options);
  return compute_pool;
}
int32 NumInterOpThreadsFromEnvironment() {
  int32_t num;
  const char* val = std::getenv("TF_NUM_INTEROP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}
int32 NumIntraOpThreadsFromEnvironment() {
  int32_t num;
  const char* val = std::getenv("TF_NUM_INTRAOP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}
#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 OMPThreadsFromEnvironment() {
  int32 num;
  const char* val = std::getenv("OMP_NUM_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}
int32 DefaultNumIntraOpThreads() {
  static int env_num_threads = NumIntraOpThreadsFromEnvironment();
  if (env_num_threads > 0) {
    return env_num_threads;
  }
  return port::MaxParallelism();
}
#endif  
int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32_t inter_op = options.config.inter_op_parallelism_threads();
  if (inter_op > 0) return inter_op;
  const int32_t env_inter_op = GetEnvNumInterOpThreads();
  if (env_inter_op > 0) return env_inter_op;
#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  if (IsMKLEnabled()) {
    const int32 intra_op = options.config.intra_op_parallelism_threads();
    const int32 omp_max_threads = OMPThreadsFromEnvironment();
    const int32 mkl_intra_op =
        (omp_max_threads > 0)
            ? omp_max_threads
            : (intra_op > 0) ? intra_op : DefaultNumIntraOpThreads();
    DCHECK_GE(mkl_intra_op, 1);
    const int32 mkl_inter_op = std::max(
        (DefaultNumInterOpThreads() + mkl_intra_op - 1) / mkl_intra_op, 2);
    VLOG(0)
        << "Creating new thread pool with default inter op setting: "
        << mkl_inter_op
        << ". Tune using inter_op_parallelism_threads for best performance.";
    return mkl_inter_op;
  }
#endif  
  return DefaultNumInterOpThreads();
}
thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options, int32_t num_threads) {
  const int32_t num_threads_real =
      num_threads > 0 ? num_threads
                      : NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Session inter op parallelism threads: " << num_threads_real;
  return new thread::ThreadPool(
      options.env, ThreadOptions(), "Compute", num_threads_real,
      !options.config.experimental().disable_thread_spinning(),
      nullptr);
}
void SchedClosure(absl::AnyInvocable<void()> closure) {
  if (!tsl::tracing::EventCollector::IsEnabled()) {
    return Env::Default()->SchedClosure(std::move(closure));
  }
  uint64 id = tsl::tracing::GetUniqueArg();
  tsl::tracing::RecordEvent(tsl::tracing::EventCategory::kScheduleClosure, id);
  Env::Default()->SchedClosure([id, closure = std::move(closure)]() mutable {
    tsl::tracing::ScopedRegion region(tsl::tracing::EventCategory::kRunClosure,
                                      id);
    closure();
  });
}
void SchedNonBlockingClosureAfter(int64_t micros,
                                  absl::AnyInvocable<void()> closure) {
  Env::Default()->SchedClosureAfter(micros, std::move(closure));
}
}  