#ifndef TENSORFLOW_CORE_UTIL_EXEC_ON_STALL_H_
#define TENSORFLOW_CORE_UTIL_EXEC_ON_STALL_H_
#include <functional>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
class ExecuteOnStall {
 public:
  ExecuteOnStall(int delay_secs, std::function<void()> f,
                 int32_t poll_microseconds = 100)
      : disabled_(false),
        joined_(false),
        env_(Env::Default()),
        f_(f),
        poll_microseconds_(poll_microseconds) {
    deadline_ = env_->NowMicros() + 1000000 * delay_secs;
    env_->SchedClosure([this]() {
      while (env_->NowMicros() < deadline_) {
        {
          mutex_lock l(mu_);
          if (disabled_) {
            break;
          }
        }
        env_->SleepForMicroseconds(poll_microseconds_);
      }
      {
        mutex_lock l(mu_);
        if (!disabled_) {
          f_();
        }
        joined_ = true;
        cond_var_.notify_all();
      }
    });
  }
  ~ExecuteOnStall() {
    mutex_lock l(mu_);
    disabled_ = true;
    if (!joined_) {
      cond_var_.wait(l);
    }
  }
 private:
  mutex mu_;
  condition_variable cond_var_;
  bool disabled_ TF_GUARDED_BY(mu_);
  bool joined_ TF_GUARDED_BY(mu_);
  Env* env_;
  std::function<void()> f_;
  int64_t deadline_;
  int32 poll_microseconds_;
};
}  
#endif  