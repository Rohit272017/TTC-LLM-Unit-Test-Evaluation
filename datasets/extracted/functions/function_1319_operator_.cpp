#include "xla/python/ifrt/index_domain.h"
#include <ostream>
#include <string>
#include "absl/strings/str_cat.h"
namespace xla {
namespace ifrt {
std::string IndexDomain::DebugString() const {
  return absl::StrCat("IndexDomain(origin=", origin_.DebugString(),
                      ",shape=", shape_.DebugString(), ")");
}
std::ostream& operator<<(std::ostream& os, const IndexDomain& index_domain) {
  return os << index_domain.DebugString();
}
}  
}  