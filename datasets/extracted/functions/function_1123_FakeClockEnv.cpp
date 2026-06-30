#include "tensorflow/core/util/fake_clock_env.h"
#include <string>
namespace tensorflow {
FakeClockEnv::FakeClockEnv(Env* wrapped) : EnvWrapper(wrapped) {}
void FakeClockEnv::AdvanceByMicroseconds(int64_t micros) {
  {
    mutex_lock l(mu_);
    current_time_ += micros;
  }
}
uint64 FakeClockEnv::NowMicros() const {
  {
    mutex_lock l(mu_);
    return current_time_;
  }
}
}  