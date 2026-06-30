#ifndef TENSORFLOW_CORE_PLATFORM_TSTRING_H_
#define TENSORFLOW_CORE_PLATFORM_TSTRING_H_
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/ctstring.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tsl/platform/tstring.h"
namespace tensorflow {
using tstring = tsl::tstring;
}
#endif  