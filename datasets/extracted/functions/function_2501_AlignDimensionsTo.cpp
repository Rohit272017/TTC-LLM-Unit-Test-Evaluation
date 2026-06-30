#include "tensorstore/index_space/alignment.h"
#include <algorithm>
#include <numeric>
#include "absl/status/status.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
absl::Status AlignDimensionsTo(IndexDomainView<> source,
                               IndexDomainView<> target,
                               span<DimensionIndex> source_matches,
                               DomainAlignmentOptions options) {
  assert(source.valid());
  assert(target.valid());
  const DimensionIndex source_rank = source.rank();
  const DimensionIndex target_rank = target.rank();
  if (!(options & DomainAlignmentOptions::broadcast) &&
      source_rank != target_rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Aligning source domain of rank ", source_rank,
        " to target domain of rank ", target_rank, " requires broadcasting"));
  }
  assert(source_matches.size() == source_rank);
  const auto source_labels = source.labels();
  const auto target_labels = target.labels();
  if (!(options & DomainAlignmentOptions::permute) ||
      internal_index_space::IsUnlabeled(source_labels) ||
      internal_index_space::IsUnlabeled(target_labels)) {
    const DimensionIndex match_rank = std::min(source_rank, target_rank);
    const DimensionIndex source_match_start = source_rank - match_rank;
    const DimensionIndex target_match_start = target_rank - match_rank;
    std::fill_n(source_matches.begin(), source_match_start, DimensionIndex(-1));
    std::iota(source_matches.begin() + source_match_start, source_matches.end(),
              target_match_start);
  } else {
    DimensionIndex next_potentially_unlabeled_target = target_rank - 1;
    for (DimensionIndex i = source_rank - 1; i >= 0; --i) {
      std::string_view source_label = source_labels[i];
      DimensionIndex j;
      if (source_label.empty()) {
        while (true) {
          if (next_potentially_unlabeled_target < 0) {
            j = -1;
            break;
          }
          if (target_labels[next_potentially_unlabeled_target].empty()) {
            j = next_potentially_unlabeled_target--;
            break;
          }
          --next_potentially_unlabeled_target;
        }
      } else {
        for (j = target_rank - 1; j >= 0; --j) {
          if (target_labels[j] == source_label) break;
        }
      }
      source_matches[i] = j;
    }
  }
  std::string mismatch_error;
  const auto source_shape = source.shape();
  const auto target_shape = target.shape();
  for (DimensionIndex i = 0; i < source_rank; ++i) {
    DimensionIndex& j = source_matches[i];
    const DimensionIndex source_size = source_shape[i];
    if (j != -1) {
      if (!(options & DomainAlignmentOptions::translate)
              ? source[i] != target[j]
              : source_size != target_shape[j]) {
        if (!(options & DomainAlignmentOptions::broadcast) ||
            source_size != 1) {
          tensorstore::StrAppend(&mismatch_error, "source dimension ", i, " ",
                                 source[i], " mismatch with target dimension ",
                                 j, " ", target[j], ", ");
        }
        j = -1;
      }
    } else {
      if (!(options & DomainAlignmentOptions::broadcast)) {
        tensorstore::StrAppend(&mismatch_error, "unmatched source dimension ",
                               i, " ", source[i], ", ");
      }
      if (source_size != 1) {
        tensorstore::StrAppend(&mismatch_error, "unmatched source dimension ",
                               i, " ", source[i],
                               " does not have a size of 1, ");
      }
    }
  }
  if (!mismatch_error.empty()) {
    mismatch_error.resize(mismatch_error.size() - 2);
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Error aligning dimensions: ", mismatch_error));
  }
  return absl::OkStatus();
}
Result<IndexTransform<>> AlignDomainTo(IndexDomainView<> source,
                                       IndexDomainView<> target,
                                       DomainAlignmentOptions options) {
  using internal_index_space::TransformAccess;
  assert(source.valid());
  assert(target.valid());
  const DimensionIndex source_rank = source.rank();
  DimensionIndex source_matches[kMaxRank];
  TENSORSTORE_RETURN_IF_ERROR(AlignDimensionsTo(
      source, target, span(source_matches).first(source_rank), options));
  const DimensionIndex target_rank = target.rank();
  auto alignment =
      internal_index_space::TransformRep::Allocate(target_rank, source_rank);
  CopyTransformRepDomain(TransformAccess::rep(target), alignment.get());
  alignment->output_rank = source_rank;
  const auto maps = alignment->output_index_maps();
  span<const Index> source_origin = source.origin();
  span<const Index> target_origin = target.origin();
  for (DimensionIndex i = 0; i < source_rank; ++i) {
    auto& map = maps[i];
    const DimensionIndex j = source_matches[i];
    const Index source_origin_value = source_origin[i];
    if (j == -1) {
      map.SetConstant();
      map.offset() = source_origin_value;
      map.stride() = 0;
    } else {
      map.SetSingleInputDimension(j);
      map.offset() = source_origin_value - target_origin[j];
      map.stride() = 1;
    }
  }
  internal_index_space::DebugCheckInvariants(alignment.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(alignment));
}
Result<IndexTransform<>> AlignTransformTo(IndexTransform<> source_transform,
                                          IndexDomainView<> target,
                                          DomainAlignmentOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto alignment,
      AlignDomainTo(source_transform.domain(), target, options));
  return ComposeTransforms(source_transform, alignment);
}
}  