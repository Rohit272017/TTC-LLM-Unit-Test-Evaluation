#ifndef QUICHE_QUIC_CORE_CONGESTION_CONTROL_WINDOWED_FILTER_H_
#define QUICHE_QUIC_CORE_CONGESTION_CONTROL_WINDOWED_FILTER_H_
#include "quiche/quic/core/quic_time.h"
namespace quic {
template <class T>
struct QUICHE_EXPORT MinFilter {
  bool operator()(const T& lhs, const T& rhs) const { return lhs <= rhs; }
};
template <class T>
struct QUICHE_EXPORT MaxFilter {
  bool operator()(const T& lhs, const T& rhs) const { return lhs >= rhs; }
};
template <class T, class Compare, typename TimeT, typename TimeDeltaT>
class QUICHE_EXPORT WindowedFilter {
 public:
  WindowedFilter(TimeDeltaT window_length, T zero_value, TimeT zero_time)
      : window_length_(window_length),
        zero_value_(zero_value),
        zero_time_(zero_time),
        estimates_{Sample(zero_value_, zero_time),
                   Sample(zero_value_, zero_time),
                   Sample(zero_value_, zero_time)} {}
  void SetWindowLength(TimeDeltaT window_length) {
    window_length_ = window_length;
  }
  void Update(T new_sample, TimeT new_time) {
    if (estimates_[0].sample == zero_value_ ||
        Compare()(new_sample, estimates_[0].sample) ||
        new_time - estimates_[2].time > window_length_) {
      Reset(new_sample, new_time);
      return;
    }
    if (Compare()(new_sample, estimates_[1].sample)) {
      estimates_[1] = Sample(new_sample, new_time);
      estimates_[2] = estimates_[1];
    } else if (Compare()(new_sample, estimates_[2].sample)) {
      estimates_[2] = Sample(new_sample, new_time);
    }
    if (new_time - estimates_[0].time > window_length_) {
      estimates_[0] = estimates_[1];
      estimates_[1] = estimates_[2];
      estimates_[2] = Sample(new_sample, new_time);
      if (new_time - estimates_[0].time > window_length_) {
        estimates_[0] = estimates_[1];
        estimates_[1] = estimates_[2];
      }
      return;
    }
    if (estimates_[1].sample == estimates_[0].sample &&
        new_time - estimates_[1].time > window_length_ >> 2) {
      estimates_[2] = estimates_[1] = Sample(new_sample, new_time);
      return;
    }
    if (estimates_[2].sample == estimates_[1].sample &&
        new_time - estimates_[2].time > window_length_ >> 1) {
      estimates_[2] = Sample(new_sample, new_time);
    }
  }
  void Reset(T new_sample, TimeT new_time) {
    estimates_[0] = estimates_[1] = estimates_[2] =
        Sample(new_sample, new_time);
  }
  void Clear() { Reset(zero_value_, zero_time_); }
  T GetBest() const { return estimates_[0].sample; }
  T GetSecondBest() const { return estimates_[1].sample; }
  T GetThirdBest() const { return estimates_[2].sample; }
 private:
  struct QUICHE_EXPORT Sample {
    T sample;
    TimeT time;
    Sample(T init_sample, TimeT init_time)
        : sample(init_sample), time(init_time) {}
  };
  TimeDeltaT window_length_;  
  T zero_value_;              
  TimeT zero_time_;           
  Sample estimates_[3];       
};
}  
#endif  