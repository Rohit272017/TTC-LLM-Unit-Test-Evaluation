#ifndef QUICHE_QUIC_CORE_QUIC_TIME_ACCUMULATOR_H_
#define QUICHE_QUIC_CORE_QUIC_TIME_ACCUMULATOR_H_
#include "quiche/quic/core/quic_time.h"
#include "quiche/quic/platform/api/quic_export.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
class QUICHE_EXPORT QuicTimeAccumulator {
  static constexpr QuicTime NotRunningSentinel() {
    return QuicTime::Infinite();
  }
 public:
  bool IsRunning() const { return last_start_time_ != NotRunningSentinel(); }
  void Start(QuicTime now) {
    QUICHE_DCHECK(!IsRunning());
    last_start_time_ = now;
    QUICHE_DCHECK(IsRunning());
  }
  void Stop(QuicTime now) {
    QUICHE_DCHECK(IsRunning());
    if (now > last_start_time_) {
      total_elapsed_ = total_elapsed_ + (now - last_start_time_);
    }
    last_start_time_ = NotRunningSentinel();
    QUICHE_DCHECK(!IsRunning());
  }
  QuicTime::Delta GetTotalElapsedTime() const { return total_elapsed_; }
  QuicTime::Delta GetTotalElapsedTime(QuicTime now) const {
    if (!IsRunning()) {
      return total_elapsed_;
    }
    if (now <= last_start_time_) {
      return total_elapsed_;
    }
    return total_elapsed_ + (now - last_start_time_);
  }
 private:
  QuicTime::Delta total_elapsed_ = QuicTime::Delta::Zero();
  QuicTime last_start_time_ = NotRunningSentinel();
};
}  
#endif  