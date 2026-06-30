#include "tensorflow/core/util/tensor_slice_set.h"
#include <unordered_map>
#include <utility>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_slice_util.h"
namespace tensorflow {
namespace checkpoint {
TensorSliceSet::TensorSliceSet(const TensorShape& shape, DataType type)
    : shape_(shape), type_(type) {}
TensorSliceSet::~TensorSliceSet() = default;
Status TensorSliceSet::Register(const TensorSlice& slice, const string& tag) {
  TensorShape result_shape;
  TF_RETURN_IF_ERROR(slice.SliceTensorShape(shape_, &result_shape));
  string str = slice.DebugString();
  if (slices_.empty()) {
    slices_hull_ = slice;
  } else {
    if (slices_hull_.Overlaps(slice)) {
      for (const auto& x : slices_) {
        if (slice.Overlaps(x.second.slice)) {
          return errors::Internal("Overlapping slices: existing slice = ",
                                  x.first, ", new slice = ", str);
        }
      }
    }
    slices_hull_.UpdateToCover(slice);
  }
  TensorSliceSet::SliceInfo info = {slice, tag, result_shape.num_elements()};
  slices_.insert(std::make_pair(str, info));
  return absl::OkStatus();
}
bool TensorSliceSet::QueryMeta(
    const TensorSlice& slice,
    std::vector<std::pair<TensorSlice, string>>* results) const {
  results->clear();
  Status s;
  string str = slice.DebugString();
  const TensorSliceSet::SliceInfo* info = gtl::FindOrNull(slices_, str);
  if (info) {
    results->emplace_back(std::make_pair(info->slice, info->tag));
    return true;
  } else {
    TensorShape target_shape;
    Status s;
    s = slice.SliceTensorShape(shape_, &target_shape);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }
    int64_t total_size = target_shape.num_elements();
    int64_t overlap_size = 0;
    TensorSlice intersection;
    TensorShape inter_shape;
    for (const auto& x : slices_) {
      if (slice.Intersect(x.second.slice, &intersection)) {
        s = intersection.SliceTensorShape(shape_, &inter_shape);
        if (!s.ok()) {
          LOG(WARNING) << s;
          return false;
        }
        overlap_size += inter_shape.num_elements();
        results->emplace_back(std::make_pair(x.second.slice, x.second.tag));
      }
    }
    if (total_size == overlap_size) {
      return true;
    } else {
      results->clear();
      return false;
    }
  }
}
Status RegisterTensorSlice(
    const string& name, const TensorShape& shape, DataType type,
    const string& tag, const TensorSlice& slice,
    std::unordered_map<string, TensorSliceSet*>* tensor_slices) {
  DCHECK_NE(tensor_slices, nullptr);
  TensorSliceSet* tss = gtl::FindPtrOrNull(*tensor_slices, name);
  if (!tss) {
    tss = new TensorSliceSet(shape, type);
    tensor_slices->insert(std::make_pair(name, tss));
  } else {
    const TensorShape& tss_shape(tss->shape());
    if (!shape.IsSameSize(tss_shape)) {
      return errors::Internal("Incompatible tensor shapes detected for tensor ",
                              name, ": existing = ", tss_shape.DebugString(),
                              ", new = ", shape.DebugString());
    }
    if (type != tss->type()) {
      return errors::Internal("Incompatible tensor types detected for tensor ",
                              name,
                              ": existing = ", DataTypeString(tss->type()),
                              ", new = ", DataTypeString(type));
    }
  }
  return tss->Register(slice, tag);
}
}  
}  