#ifndef XLA_TSL_PROFILER_CONVERT_XLA_OP_UTILS_H_
#define XLA_TSL_PROFILER_CONVERT_XLA_OP_UTILS_H_
#include <string>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
namespace tsl {
namespace profiler {
inline bool IsFusion(absl::string_view category) {
  return absl::EndsWith(category, " fusion");
}
inline std::string HloModuleNameWithProgramId(absl::string_view hlo_module_name,
                                              uint64_t program_id) {
  return absl::StrCat(hlo_module_name, "(", program_id, ")");
}
}  
}  
#endif  