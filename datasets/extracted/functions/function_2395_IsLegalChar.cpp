#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include <cctype>
#include <string>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/Location.h"  
#include "mlir/Support/LLVM.h"  
namespace mlir {
namespace {
bool IsLegalChar(char c, bool first_char) {
  if (isalpha(c)) return true;
  if (isdigit(c)) return true;
  if (c == '.') return true;
  if (c == '_') return true;
  if (first_char) return false;
  if (c == '/') return true;
  if (c == '-') return true;
  return false;
}
}  
void LegalizeNodeName(std::string& name) {
  if (name.empty()) return;
  if (!IsLegalChar(name[0], true)) name[0] = '.';
  for (char& c : llvm::drop_begin(name, 1))
    if (!IsLegalChar(c, false)) c = '.';
}
std::string GetNameFromLoc(Location loc) {
  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);
  bool names_is_nonempty = false;
  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();
    if (auto name_loc = mlir::dyn_cast<NameLoc>(curr_loc)) {
      auto name = name_loc.getName().strref().split('@').first;
      if (!name.ends_with(":")) {
        loc_names.push_back(name);
        if (!name.empty()) names_is_nonempty = true;
      }
      continue;
    } else if (auto call_loc = mlir::dyn_cast<CallSiteLoc>(curr_loc)) {
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = mlir::dyn_cast<FusedLoc>(curr_loc)) {
      auto reversed_fused_locs = llvm::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
      continue;
    }
    loc_names.push_back(llvm::StringRef());
  }
  if (names_is_nonempty)
    return llvm::join(loc_names.begin(), loc_names.end(), ";");
  return "";
}
}  