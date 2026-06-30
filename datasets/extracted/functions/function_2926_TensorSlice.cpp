#include "tensorflow/core/framework/tensor_slice.h"
#include <limits>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
TensorSlice::TensorSlice(int dim) { SetFullSlice(dim); }
TensorSlice::TensorSlice(const TensorSliceProto& proto) {
  starts_.reserve(proto.extent_size());
  lengths_.reserve(proto.extent_size());
  for (const auto& e : proto.extent()) {
    starts_.push_back(e.start());
    lengths_.push_back(GetExtentLength(e));
  }
}
TensorSlice::TensorSlice(
    std::initializer_list<std::pair<int64_t, int64_t>> extents) {
  starts_.reserve(extents.size());
  lengths_.reserve(extents.size());
  for (const auto& e : extents) {
    starts_.push_back(e.first);
    lengths_.push_back(e.second);
  }
}
Status TensorSlice::BuildTensorSlice(const TensorSliceProto& proto,
                                     TensorSlice* output) {
  output->Clear();
  output->starts_.reserve(proto.extent_size());
  output->lengths_.reserve(proto.extent_size());
  for (const auto& e : proto.extent()) {
    int64_t l = GetExtentLength(e);
    if (e.start() != 0 || l != kFullExtent) {
      if (e.start() < 0 || l <= 0) {
        return errors::InvalidArgument(
            "Expected non-negative start and positive length but got start = ",
            e.start(), ", length = ", l, ": extent = ", e.ShortDebugString());
      }
      if (static_cast<uint64_t>(e.start()) + static_cast<uint64_t>(e.length()) >
          std::numeric_limits<int64_t>::max()) {
        return errors::InvalidArgument(
            "Extent end exceeds the maximum possible size: extent = ",
            e.ShortDebugString());
      }
    }
    output->starts_.push_back(e.start());
    output->lengths_.push_back(l);
  }
  return absl::OkStatus();
}
Status TensorSlice::Parse(const string& str, TensorSlice* slice) {
  std::vector<string> items = str_util::Split(str, ':', str_util::SkipEmpty());
  slice->starts_.reserve(items.size());
  slice->lengths_.reserve(items.size());
  for (const string& x : items) {
    int64_t s, l;
    if (x == "-") {
      s = 0;
      l = kFullExtent;
    } else {
      std::vector<string> sl = str_util::Split(x, ',', str_util::SkipEmpty());
      if (sl.size() != 2 || !strings::safe_strto64(sl[0], &s) ||
          !strings::safe_strto64(sl[1], &l)) {
        return errors::InvalidArgument(
            "Expected a pair of numbers or '-' "
            "but got '",
            x, "': string = ", str);
      }
      if (s < 0 || l <= 0) {
        return errors::InvalidArgument(
            "Expected non-negative start and "
            "positive length but got start = ",
            s, ", length = ", l, ": string = ", str);
      }
    }
    slice->starts_.push_back(s);
    slice->lengths_.push_back(l);
  }
  return absl::OkStatus();
}
void TensorSlice::Clear() {
  starts_.clear();
  lengths_.clear();
}
bool TensorSlice::IsFull() const {
  for (int d = 0; d < dims(); ++d) {
    if (!IsFullAt(d)) return false;
  }
  return true;
}
void TensorSlice::SetFullSlice(int dim) {
  Clear();
  starts_.reserve(dim);
  lengths_.reserve(dim);
  for (int d = 0; d < dim; ++d) {
    starts_.push_back(0);
    lengths_.push_back(kFullExtent);
  }
}
void TensorSlice::Extend(int dim) {
  int old_dim = dims();
  DCHECK_LE(old_dim, dim);
  starts_.resize(dim);
  lengths_.resize(dim);
  for (int d = old_dim; d < dim; ++d) {
    starts_[d] = 0;
    lengths_[d] = kFullExtent;
  }
}
void TensorSlice::AsProto(TensorSliceProto* proto) const {
  for (int d = 0; d < dims(); ++d) {
    TensorSliceProto::Extent* e = proto->add_extent();
    if (!IsFullAt(d)) {
      e->set_start(starts_[d]);
      e->set_length(lengths_[d]);
    }
  }
}
string TensorSlice::DebugString() const {
  string buffer;
  bool first = true;
  for (int d = 0; d < dims(); ++d) {
    if (!first) {
      buffer.append(":");
    }
    if (IsFullAt(d)) {
      buffer.append("-");
    } else {
      strings::StrAppend(&buffer, starts_[d], ",", lengths_[d]);
    }
    first = false;
  }
  return buffer;
}
bool TensorSlice::Intersect(const TensorSlice& other,
                            TensorSlice* result) const {
  if (dims() != other.dims()) {
    return false;
  }
  if (result) {
    result->SetFullSlice(dims());
  }
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      if (result) {
        result->set_start(d, other.start(d));
        result->set_length(d, other.length(d));
      }
    } else if (other.IsFullAt(d)) {
      if (result) {
        result->set_start(d, start(d));
        result->set_length(d, length(d));
      }
    } else {
      int64_t s = std::max(start(d), other.start(d));
      int64_t l = std::min(end(d), other.end(d)) - s;
      if (l > 0) {
        if (result) {
          result->set_start(d, s);
          result->set_length(d, l);
        }
      } else {
        if (result) {
          result->Clear();
        }
        return false;
      }
    }
  }
  return true;
}
bool TensorSlice::operator==(const TensorSlice& other) const {
  return dims() == other.dims() && starts_ == other.starts_ &&
         lengths_ == other.lengths_;
}
void TensorSlice::ComputeRelative(const TensorSlice& sub,
                                  TensorSlice* relative) const {
  DCHECK_EQ(dims(), sub.dims());
  relative->SetFullSlice(dims());
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      relative->set_start(d, sub.start(d));
      relative->set_length(d, sub.length(d));
    } else {
      relative->set_start(d, sub.start(d) - start(d));
      relative->set_length(d, sub.length(d));
    }
  }
}
void TensorSlice::UpdateToCover(const TensorSlice& other) {
  DCHECK_EQ(dims(), other.dims());
  for (int d = 0; d < dims(); ++d) {
    if (!IsFullAt(d)) {
      if (other.IsFullAt(d)) {
        starts_[d] = 0;
        lengths_[d] = kFullExtent;
      } else {
        const auto new_end = std::max(end(d), other.end(d));
        set_start(d, std::min(start(d), other.start(d)));
        set_length(d, new_end - start(d));
      }
    }
  }
}
bool TensorSlice::HasExtentLength(const TensorSliceProto::Extent& extent) {
  return extent.has_length_case() == TensorSliceProto::Extent::kLength;
}
int64_t TensorSlice::GetExtentLength(const TensorSliceProto::Extent& extent) {
  if (!HasExtentLength(extent)) return -1;
  return extent.length();
}
Status TensorSlice::SliceTensorShape(const TensorShape& shape,
                                     TensorShape* result_shape) const {
  result_shape->Clear();
  if (shape.dims() != dims()) {
    return errors::Internal("Mismatching ranks: shape = ", shape.DebugString(),
                            ", slice = ", DebugString());
  }
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      result_shape->AddDim(shape.dim_size(d));
    } else {
      if (end(d) <= shape.dim_size(d)) {
        result_shape->AddDim(length(d));
      } else {
        result_shape->Clear();
        return errors::Internal("Extent in dimension ", d,
                                " out of bounds: shape = ", shape.DebugString(),
                                ", slice = ", DebugString());
      }
    }
  }
  return absl::OkStatus();
}
const int64_t TensorSlice::kFullExtent = -1;
}  