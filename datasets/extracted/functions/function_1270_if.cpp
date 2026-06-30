#include "arolla/util/bytes.h"
#include <cstddef>
#include <string>
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "arolla/util/repr.h"
namespace arolla {
ReprToken ReprTraits<Bytes>::operator()(const Bytes& value) const {
  constexpr size_t kBytesAbbrevLimit = 120;
  ReprToken result;
  absl::string_view bytes = value;
  if (bytes.size() <= kBytesAbbrevLimit) {
    result.str = absl::StrCat("b'", absl::CHexEscape(bytes), "'");
  } else {
    result.str =
        absl::StrCat("b'", absl::CHexEscape(bytes.substr(0, kBytesAbbrevLimit)),
                     "... (", bytes.size(), " bytes total)'");
  }
  return result;
}
}  