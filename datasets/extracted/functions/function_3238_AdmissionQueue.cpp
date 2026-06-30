#include "tensorstore/internal/rate_limiter/admission_queue.h"
#include <stddef.h>
#include <cassert>
#include <limits>
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
namespace tensorstore {
namespace internal {
AdmissionQueue::AdmissionQueue(size_t limit)
    : limit_(limit == 0 ? std::numeric_limits<size_t>::max() : limit) {}
void AdmissionQueue::Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;
  {
    absl::MutexLock lock(&mutex_);
    if (in_flight_++ >= limit_) {
      internal::intrusive_linked_list::InsertBefore(RateLimiterNodeAccessor{},
                                                    &head_, node);
      return;
    }
  }
  RunStartFunction(node);
}
void AdmissionQueue::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);
  RateLimiterNode* next_node = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    in_flight_--;
    next_node = head_.next_;
    if (next_node == &head_) return;
    internal::intrusive_linked_list::Remove(RateLimiterNodeAccessor{},
                                            next_node);
  }
  RunStartFunction(next_node);
}
}  
}  