#include "tensorstore/rank.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
std::string StaticCastTraits<DimensionIndex>::Describe(DimensionIndex value) {
  if (value == dynamic_rank) return "dynamic rank";
  return tensorstore::StrCat("rank of ", value);
}
absl::Status ValidateRank(DimensionIndex rank) {
  if (!IsValidRank(rank)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Rank ", rank, " is outside valid range [0, ", kMaxRank, "]"));
  }
  return absl::OkStatus();
}
}  