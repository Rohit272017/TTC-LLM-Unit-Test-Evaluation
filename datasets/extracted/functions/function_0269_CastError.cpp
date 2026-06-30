#include "tensorstore/static_cast.h"
#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_cast {
absl::Status CastError(std::string_view source_description,
                       std::string_view target_description) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Cannot cast ", source_description, " to ", target_description));
}
}  
}  