#include "xla/python/ifrt/index.h"
#include <ostream>
#include <string>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
namespace xla {
namespace ifrt {
std::string Index::DebugString() const {
  return absl::StrCat("[", absl::StrJoin(elements_, ","), "]");
}
std::ostream& operator<<(std::ostream& os, const Index& index) {
  return os << index.DebugString();
}
}  
}  