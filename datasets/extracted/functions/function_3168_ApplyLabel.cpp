#include "tensorstore/index_space/internal/label_op.h"
#include <stddef.h>
#include <string>
#include <string_view>
#include <utility>
#include "absl/status/status.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/dimension_labels.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_index_space {
Result<IndexTransform<>> ApplyLabel(IndexTransform<> transform,
                                    DimensionIndexBuffer* dimensions,
                                    internal::StringLikeSpan labels,
                                    bool domain_only) {
  if (dimensions->size() != static_cast<size_t>(labels.size())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of dimensions (", dimensions->size(),
        ") does not match number of labels (", labels.size(), ")."));
  }
  auto rep = MutableRep(
      TransformAccess::rep_ptr<container>(std::move(transform)), domain_only);
  const DimensionIndex input_rank = rep->input_rank;
  span<std::string> input_labels = rep->input_labels().first(input_rank);
  for (DimensionIndex i = 0;
       i < static_cast<DimensionIndex>(dimensions->size()); ++i) {
    const DimensionIndex input_dim = (*dimensions)[i];
    std::string_view label = labels[i];
    input_labels[input_dim].assign(label.begin(), label.end());
  }
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateDimensionLabelsAreUnique(input_labels));
  internal_index_space::DebugCheckInvariants(rep.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}
}  
}  