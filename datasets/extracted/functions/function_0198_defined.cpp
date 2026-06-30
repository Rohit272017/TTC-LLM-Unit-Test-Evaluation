#include "tsl/platform/setround.h"
#include "tsl/platform/logging.h"
namespace tsl {
namespace port {
#if defined(TF_BROKEN_CFENV)
ScopedSetRound::ScopedSetRound(const int mode) : original_mode_(mode) {
  DCHECK_EQ(mode, FE_TONEAREST);
}
ScopedSetRound::~ScopedSetRound() {}
#else
ScopedSetRound::ScopedSetRound(const int mode) {
  original_mode_ = std::fegetround();
  if (original_mode_ < 0) {
    original_mode_ = FE_TONEAREST;
  }
  std::fesetround(mode);
}
ScopedSetRound::~ScopedSetRound() { std::fesetround(original_mode_); }
#endif  
}  
}  