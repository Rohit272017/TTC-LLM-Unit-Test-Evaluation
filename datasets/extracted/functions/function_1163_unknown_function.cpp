#ifndef TENSORFLOW_CORE_LIB_GTL_MAP_UTIL_H_
#define TENSORFLOW_CORE_LIB_GTL_MAP_UTIL_H_
#include "xla/tsl/lib/gtl/map_util.h"
namespace tensorflow {
namespace gtl {
using ::tsl::gtl::EraseKeyReturnValuePtr;
using ::tsl::gtl::FindOrNull;
using ::tsl::gtl::FindPtrOrNull;
using ::tsl::gtl::FindWithDefault;
using ::tsl::gtl::InsertIfNotPresent;
using ::tsl::gtl::InsertOrUpdate;
using ::tsl::gtl::LookupOrInsert;
using ::tsl::gtl::ReverseMap;
}  
}  
#endif  