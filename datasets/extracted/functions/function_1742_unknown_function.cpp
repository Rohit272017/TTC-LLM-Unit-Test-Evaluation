#ifndef TENSORFLOW_CORE_PLATFORM_INTRUSIVE_PTR_H_
#define TENSORFLOW_CORE_PLATFORM_INTRUSIVE_PTR_H_
#include <algorithm>
#include "tsl/platform/intrusive_ptr.h"
namespace tensorflow {
namespace core {
template <class T>
using IntrusivePtr = tsl::core::IntrusivePtr<T>;
}  
}  
#endif  