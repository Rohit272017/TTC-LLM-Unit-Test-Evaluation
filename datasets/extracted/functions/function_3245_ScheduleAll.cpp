#ifndef XLA_BACKENDS_CPU_RUNTIME_CONCURRENCY_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONCURRENCY_H_
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include "tsl/platform/logging.h"
#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"
namespace xla::cpu {
template <typename F,
          std::enable_if_t<std::is_invocable_v<F, int64_t>>* = nullptr>
void ScheduleAll(const Eigen::ThreadPoolDevice* intra_op_threadpool, int64_t n,
                 F&& f) {
  DCHECK(n >= 0) << "n must be non-negative";
  if (n == 0) return;
  if (n == 1) {
    f(0);
    return;
  }
  struct State {
    State(const Eigen::ThreadPoolDevice* intra_op_threadpool, F&& f)
        : intra_op_threadpool(intra_op_threadpool), f(std::forward<F>(f)) {}
    void Execute(std::shared_ptr<State> self, int64_t start, int64_t end) {
      while (end - start > 1) {
        uint64_t mid = (start + end) / 2;
        intra_op_threadpool->getPool()->Schedule(
            std::bind(&State::Execute, this, self, mid, end));
        end = mid;
      }
      f(start);
    }
    const Eigen::ThreadPoolDevice* intra_op_threadpool;
    F f;
  };
  auto s = std::make_shared<State>(intra_op_threadpool, std::forward<F>(f));
  s->Execute(std::move(s), 0, n);
}
}  
#endif  