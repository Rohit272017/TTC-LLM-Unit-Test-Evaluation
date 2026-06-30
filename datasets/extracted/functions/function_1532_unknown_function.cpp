#ifndef AROLLA_UTIL_REFCOUNT_H_
#define AROLLA_UTIL_REFCOUNT_H_
#include <atomic>
#include <cstdint>
namespace arolla {
class Refcount {
 public:
  constexpr Refcount() noexcept : count_{1} {}
  void increment() noexcept { count_.fetch_add(1, std::memory_order_relaxed); }
  [[nodiscard]] bool decrement() noexcept {
    return count_.fetch_sub(1, std::memory_order_acq_rel) != 1;
  }
  [[nodiscard]] bool skewed_decrement() noexcept {
    auto refcount = count_.load(std::memory_order_acquire);
    return refcount != 1 && count_.fetch_sub(1, std::memory_order_acq_rel) != 1;
  }
  struct TestOnly {};
  constexpr Refcount(TestOnly, int initial_count) noexcept
      : count_{initial_count} {}
 private:
  std::atomic<int32_t> count_;
};
}  
#endif  