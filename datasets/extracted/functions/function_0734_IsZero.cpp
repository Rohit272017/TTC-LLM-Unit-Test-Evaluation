#include "tensorstore/internal/multi_barrier.h"
#include <cassert>
namespace tensorstore {
namespace internal {
namespace {
bool IsZero(void* arg) { return *reinterpret_cast<int*>(arg) == 0; }
}  
MultiBarrier::MultiBarrier(int num_threads)
    : blocking_{num_threads, 0}, asleep_(0), num_threads_(num_threads << 1) {
  assert(num_threads > 0);
}
MultiBarrier::~MultiBarrier() {
  absl::MutexLock l(&lock_);
  lock_.Await(absl::Condition(IsZero, &asleep_));
}
bool MultiBarrier::Block() {
  absl::MutexLock l(&lock_);
  int& num_to_block = blocking_[num_threads_ & 1];
  num_to_block--;
  assert(num_to_block >= 0);
  if (num_to_block == 0) {
    int num_threads = num_threads_ >> 1;
    num_threads_ ^= 1;
    blocking_[num_threads_ & 1] = num_threads;
    asleep_ = num_threads;
  } else {
    lock_.Await(absl::Condition(IsZero, &num_to_block));
  }
  asleep_--;
  return asleep_ == 0;
}
}  
}  