#include "xla/python/ifrt_proxy/common/array_util.h"
#include <string>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
namespace proxy {
namespace {
std::string StridesAsStr(const ArrayMemRegion::ByteStrides& strides) {
  if (!strides.has_value()) return "strides{nullopt}";
  return absl::StrCat("strides{", absl::StrJoin(*strides, ","), "}");
}
}  
absl::StatusOr<std::vector<int64_t>> DefaultByteStrides(const DType dtype,
                                                        const Shape& shape) {
  if (!dtype.byte_size().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type to query byte-strides for: ",
                     dtype.DebugString()));
  }
  std::vector<int64_t> result(shape.dims().size());
  int64_t stride = *dtype.byte_size();
  for (int i = static_cast<int>(shape.dims().size()) - 1; i >= 0; --i) {
    result[i] = stride;
    stride *= shape.dims()[i];
  }
  return result;
}
absl::StatusOr<ArrayMemRegion> ArrayMemRegion::FromZerothElementPointer(
    const void* zeroth_element, const DType dtype, const Shape& shape,
    ByteStrides byte_strides) {
  if (!dtype.byte_size().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type to construct ArrayMemRegion: ",
                     dtype.DebugString()));
  }
  void* const mem_region_start = const_cast<void*>(zeroth_element);
  if (!byte_strides.has_value() ||
      (byte_strides->empty() && shape.dims().empty())) {
    return ArrayMemRegion(mem_region_start,
                          dtype.byte_size().value() * shape.num_elements());
  }
  if (shape.num_elements() == 0) {
    return ArrayMemRegion(mem_region_start, 0);
  }
  if (shape.dims().size() != byte_strides->size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shape has different dimensions from byte_strides: ",
                     shape.DebugString(), " vs ", StridesAsStr(byte_strides)));
  }
  uint64_t last_element_byte_offset = 0;
  for (int i = 0; i < byte_strides->size(); ++i) {
    int stride = (*byte_strides)[i];
    if (shape.dims()[i] < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("A shape dimension is negative: ", shape.DebugString()));
    } else if (shape.dims()[i] == 1) {
      continue;
    } else if (stride <= 0) {
      return absl::UnimplementedError(
          absl::StrCat("Negative or zero strides are not fully supported: ",
                       StridesAsStr(byte_strides)));
    } else if (stride % dtype.byte_size().value() != 0) {
      return absl::UnimplementedError(absl::StrCat(
          "byte_stride[", i, "] is not a multiple of the data-type's size: ",
          StridesAsStr(byte_strides), ", dtype=", dtype.DebugString()));
    } else {
      DCHECK_GT(shape.dims()[i], 0);
      last_element_byte_offset += (stride * (shape.dims()[i] - 1));
    }
  }
  return ArrayMemRegion(mem_region_start,
                        last_element_byte_offset + dtype.byte_size().value());
}
absl::StatusOr<ArrayMemRegion> ArrayMemRegion::FromMinimalMemRegion(
    absl::string_view mem_region, const DType dtype, const Shape& shape,
    ByteStrides byte_strides) {
  TF_ASSIGN_OR_RETURN(
      auto result,
      FromZerothElementPointer(mem_region.data(), dtype, shape, byte_strides));
  if (result.mem_region().size() != mem_region.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect size ", result.mem_region().size(), " vs ",
                     mem_region.size(), "; is provided memory region minimal? ",
                     dtype.DebugString(), " ", shape.DebugString(), " ",
                     StridesAsStr(byte_strides)));
  }
  CHECK_EQ(result.mem_region().data(), mem_region.data());
  return result;
}
absl::string_view ArrayMemRegion::mem_region() const {
  return absl::string_view(static_cast<char*>(mem_region_start_), nbytes_);
}
void* ArrayMemRegion::zeroth_element() const {
  return mem_region_start_;
}
}  
}  
}  