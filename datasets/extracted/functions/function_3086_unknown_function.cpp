#ifndef TENSORFLOW_TSL_PLATFORM_STRINGPIECE_H_
#define TENSORFLOW_TSL_PLATFORM_STRINGPIECE_H_
#include "absl/base/macros.h"
#include "absl/strings/string_view.h"  
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif
namespace tsl {
using StringPiece ABSL_DEPRECATE_AND_INLINE() = absl::string_view;
}  
#endif  