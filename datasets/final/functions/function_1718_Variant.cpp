#include "tensorflow/lite/core/async/interop/variant.h"
#include <cstring>
#include <utility>
namespace tflite {
namespace interop {
Variant::Variant() {
  type = kInvalid;
  val.i = 0;
}
bool Variant::operator==(const Variant& other) const {
  if (type != other.type) return false;
  switch (type) {
    case kInvalid:
      return true;
    case kInt:
      return val.i == other.val.i;
    case kSizeT:
      return val.s == other.val.s;
    case kString:
      return (val.c == other.val.c) || (strcmp(val.c, other.val.c) == 0);
    case kBool:
      return val.b == other.val.b;
  }
}
bool Variant::operator!=(const Variant& other) const {
  return !(*this == other);
}
}  
}  