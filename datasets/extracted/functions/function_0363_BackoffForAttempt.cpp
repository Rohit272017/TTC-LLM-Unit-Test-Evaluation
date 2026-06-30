#include "tensorstore/internal/retry.h"
#include <stdint.h>
#include <cassert>
#include "absl/random/random.h"
#include "absl/time/time.h"
namespace tensorstore {
namespace internal {
absl::Duration BackoffForAttempt(int attempt, absl::Duration initial_delay,
                                 absl::Duration max_delay,
                                 absl::Duration jitter) {
  assert(initial_delay > absl::ZeroDuration());
  assert(max_delay >= initial_delay);
  assert(attempt >= 0);
  int64_t multiple = int64_t{1} << (attempt > 62 ? 62 : attempt);
  auto delay = initial_delay * multiple;
  int64_t jitter_us = absl::ToInt64Microseconds(jitter);
  if (jitter_us > 0) {
    delay += absl::Microseconds(absl::Uniform(
        absl::IntervalClosed, absl::InsecureBitGen{}, 0, jitter_us));
  }
  if (delay > max_delay) delay = max_delay;
  return delay;
}
}  
}  