#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include <algorithm>
#include <utility>
#include "absl/functional/any_invocable.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace serving {
PeriodicFunction::PeriodicFunction(absl::AnyInvocable<void()> function,
                                   const int64_t interval_micros,
                                   const Options& options)
    : function_(std::move(function)),
      interval_micros_([interval_micros]() -> int64 {
        if (interval_micros < 0) {
          const string error = strings::StrCat(
              " The value of 'interval_micros' should be >= 0: ",
              interval_micros, ". ");
          DCHECK(false) << error;
          LOG(WARNING) << error << "Resetting it to 0.";
          return 0;
        }
        return interval_micros;
      }()),
      options_(options) {
  thread_.reset(options_.env->StartThread(
      options_.thread_options, options_.thread_name_prefix, [this]() {
        RunLoop(options_.env->NowMicros());
      }));
}
PeriodicFunction::~PeriodicFunction() {
  NotifyStop();
  thread_.reset();
}
void PeriodicFunction::NotifyStop() {
  if (!stop_thread_.HasBeenNotified()) {
    stop_thread_.Notify();
  }
}
void PeriodicFunction::RunLoop(const int64_t start) {
  {
    if (options_.startup_delay_micros > 0) {
      const int64_t deadline = start + options_.startup_delay_micros;
      options_.env->SleepForMicroseconds(deadline - start);
    }
    while (!stop_thread_.HasBeenNotified()) {
      VLOG(3) << "Running function.";
      const int64_t begin = options_.env->NowMicros();
      function_();
      const int64_t end =
          std::max(static_cast<int64_t>(options_.env->NowMicros()), begin);
      const int64_t deadline = begin + interval_micros_;
      if (deadline > end) {
        if (end > begin) {
          VLOG(3) << "Reducing interval_micros from " << interval_micros_
                  << " to " << (deadline - end);
        }
        options_.env->SleepForMicroseconds(deadline - end);
      } else {
        VLOG(3) << "Function took longer than interval_micros, so not sleeping";
      }
    }
  }
}
}  
}  