#include "tensorstore/index_space/index_vector_or_scalar.h"
#include <system_error>  
#include "absl/status/status.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_index_space {
absl::Status CheckIndexVectorSize(IndexVectorOrScalarView indices,
                                  DimensionIndex size) {
  if (indices.pointer && indices.size_or_scalar != size)
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of dimensions (", size, ") does not match number of indices (",
        indices.size_or_scalar, ")"));
  return absl::OkStatus();
}
}  
}  