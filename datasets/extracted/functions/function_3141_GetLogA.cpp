#include "tensorstore/internal/rate_limiter/scaling_rate_limiter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include "absl/log/absl_check.h"
#include "absl/time/time.h"
#include "tensorstore/internal/rate_limiter/token_bucket_rate_limiter.h"
namespace tensorstore {
namespace internal {
namespace {
double GetLogA(absl::Duration doubling_time) {
  if (doubling_time <= absl::ZeroDuration() ||
      doubling_time == absl::InfiniteDuration()) {
    return 0;
  }
  return 0.69314718055994530941723212145817656 /
         absl::ToDoubleSeconds(doubling_time);
}
double GetMaxAvailable(double initial_rate) {
  return std::min(initial_rate * 1000.0, 2000.0);
}
}  
DoublingRateLimiter::DoublingRateLimiter(double initial_rate,
                                         absl::Duration doubling_time)
    : TokenBucketRateLimiter(GetMaxAvailable(initial_rate)),
      initial_rate_(initial_rate),
      doubling_time_(doubling_time),
      a_(GetLogA(doubling_time)) {
  ABSL_CHECK_GT(initial_rate, std::numeric_limits<double>::min());
  ABSL_CHECK_GT(a_, 0);
}
DoublingRateLimiter::DoublingRateLimiter(double initial_rate,
                                         absl::Duration doubling_time,
                                         std::function<absl::Time()> clock)
    : TokenBucketRateLimiter(GetMaxAvailable(initial_rate), std::move(clock)),
      initial_rate_(initial_rate),
      doubling_time_(doubling_time),
      a_(GetLogA(doubling_time)) {
  ABSL_CHECK_GT(initial_rate, std::numeric_limits<double>::min());
  ABSL_CHECK_GT(a_, 0);
}
double DoublingRateLimiter::TokensToAdd(absl::Time current,
                                        absl::Time previous) const {
  double int_current =
      std::exp(a_ * absl::ToDoubleSeconds(current - start_time_));
  double int_prev =
      std::exp(a_ * absl::ToDoubleSeconds(previous - start_time_));
  return initial_rate_ * (int_current - int_prev) / a_;
}
absl::Duration DoublingRateLimiter::GetSchedulerDelay() const {
  return absl::Milliseconds(10);
}
ConstantRateLimiter::ConstantRateLimiter(double initial_rate)
    : TokenBucketRateLimiter(GetMaxAvailable(initial_rate)),
      initial_rate_(initial_rate),
      r_(absl::Seconds(1.0 / initial_rate)) {
  ABSL_CHECK_GT(initial_rate, std::numeric_limits<double>::min());
}
ConstantRateLimiter::ConstantRateLimiter(double initial_rate,
                                         std::function<absl::Time()> clock)
    : TokenBucketRateLimiter(GetMaxAvailable(initial_rate), std::move(clock)),
      initial_rate_(initial_rate),
      r_(absl::Seconds(1.0 / initial_rate)) {
  ABSL_CHECK_GT(initial_rate, std::numeric_limits<double>::min());
}
double ConstantRateLimiter::TokensToAdd(absl::Time current,
                                        absl::Time previous) const {
  return initial_rate_ * absl::ToDoubleSeconds(current - previous);
}
absl::Duration ConstantRateLimiter::GetSchedulerDelay() const {
  return std::max(r_, absl::Milliseconds(10));
}
}  
}  