#ifndef XLA_TSL_PROFILER_UTILS_TIMESPAN_H_
#define XLA_TSL_PROFILER_UTILS_TIMESPAN_H_
#include <algorithm>
#include <string>
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace profiler {
class Timespan {
 public:
  static Timespan FromEndPoints(uint64 begin_ps, uint64 end_ps) {
    if (begin_ps > end_ps) {
      return Timespan(begin_ps, 0);
    }
    return Timespan(begin_ps, end_ps - begin_ps);
  }
  explicit Timespan(uint64 begin_ps = 0, uint64 duration_ps = 0)
      : begin_ps_(begin_ps), duration_ps_(duration_ps) {}
  uint64 begin_ps() const { return begin_ps_; }
  uint64 middle_ps() const { return begin_ps_ + duration_ps_ / 2; }
  uint64 end_ps() const { return begin_ps_ + duration_ps_; }
  uint64 duration_ps() const { return duration_ps_; }
  bool Instant() const { return duration_ps() == 0; }
  bool Empty() const { return begin_ps() == 0 && duration_ps() == 0; }
  bool Overlaps(const Timespan& other) const {
    return begin_ps() <= other.end_ps() && other.begin_ps() <= end_ps();
  }
  bool Includes(const Timespan& other) const {
    return begin_ps() <= other.begin_ps() && other.end_ps() <= end_ps();
  }
  bool Includes(uint64 time_ps) const { return Includes(Timespan(time_ps)); }
  uint64 OverlappedDurationPs(const Timespan& other) const {
    if (!Overlaps(other)) return 0;
    return std::min(end_ps(), other.end_ps()) -
           std::max(begin_ps(), other.begin_ps());
  }
  void ExpandToInclude(const Timespan& other) {
    *this = FromEndPoints(std::min(begin_ps(), other.begin_ps()),
                          std::max(end_ps(), other.end_ps()));
  }
  bool operator<(const Timespan& other) const {
    if (begin_ps_ < other.begin_ps_) return true;
    if (begin_ps_ > other.begin_ps_) return false;
    return duration_ps_ > other.duration_ps_;
  }
  bool operator==(const Timespan& other) const {
    return begin_ps_ == other.begin_ps_ && duration_ps_ == other.duration_ps_;
  }
  std::string DebugString() const {
    return absl::StrCat("[", begin_ps(), ", ", end_ps(), "]");
  }
  static bool ByDuration(const Timespan& a, const Timespan& b) {
    if (a.duration_ps_ < b.duration_ps_) return true;
    if (a.duration_ps_ > b.duration_ps_) return false;
    return a.begin_ps_ < b.begin_ps_;
  }
 private:
  uint64 begin_ps_;
  uint64 duration_ps_;  
};
inline Timespan PicoSpan(uint64 start_ps, uint64 end_ps) {
  return Timespan::FromEndPoints(start_ps, end_ps);
}
inline Timespan MilliSpan(double start_ms, double end_ms) {
  return PicoSpan(MilliToPico(start_ms), MilliToPico(end_ms));
}
}  
}  
#endif  