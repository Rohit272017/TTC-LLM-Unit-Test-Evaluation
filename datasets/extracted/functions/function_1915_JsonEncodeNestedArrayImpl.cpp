#include "tensorstore/internal/json/array.h"
#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_json {
::nlohmann::json JsonEncodeNestedArrayImpl(
    ArrayView<const void, dynamic_rank, offset_origin> array,
    absl::FunctionRef<::nlohmann::json(const void*)> encode_element) {
  if (array.rank() == 0) {
    assert(array.data());
    return encode_element(array.data());
  }
  using array_t = ::nlohmann::json::array_t;
  array_t* path[kMaxRank];
  DimensionIndex level = 0;
  array_t j_root;
  j_root.reserve(array.shape()[0]);
  path[0] = &j_root;
  if (array.shape()[0] == 0) {
    return j_root;
  }
  ByteStridedPointer<const void> pointer = array.byte_strided_origin_pointer();
  while (true) {
    array_t* j_parent = path[level];
    if (level == array.rank() - 1) {
      assert(pointer.get());
      j_parent->push_back(encode_element(pointer.get()));
    } else {
      const Index size = array.shape()[level + 1];
      array_t next_array;
      next_array.reserve(size);
      j_parent->emplace_back(std::move(next_array));
      j_parent = j_parent->back().get_ptr<array_t*>();
      if (size != 0) {
        path[++level] = j_parent;
        continue;
      }
    }
    while (true) {
      array_t* j_array = path[level];
      const Index i = j_array->size();
      const Index size = array.shape()[level];
      const Index byte_stride = array.byte_strides()[level];
      pointer += byte_stride;
      if (i != size) break;
      pointer -= i * byte_stride;
      if (level-- == 0) {
        return j_root;
      }
    }
  }
}
Result<SharedArray<void>> JsonParseNestedArrayImpl(
    const ::nlohmann::json& j_root, DataType dtype,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& v, void* out)>
        decode_element) {
  assert(dtype.valid());
  using array_t = ::nlohmann::json::array_t;
  SharedArray<void> array;
  ByteStridedPointer<void> pointer;
  const Index byte_stride = dtype->size;
  Index shape_or_position[kMaxRank];
  const array_t* path[kMaxRank];
  DimensionIndex nesting_level = 0;
  const ::nlohmann::json* j = &j_root;
  const auto allocate_array = [&] {
    array =
        AllocateArray(tensorstore::span(&shape_or_position[0], nesting_level),
                      c_order, default_init, dtype);
    pointer = array.byte_strided_origin_pointer();
    std::fill_n(&shape_or_position[0], nesting_level, static_cast<Index>(0));
  };
  while (true) {
    const array_t* j_array = j->get_ptr<const ::nlohmann::json::array_t*>();
    if (!j_array) {
      if (!array.data()) allocate_array();
      if (nesting_level != array.rank()) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected rank-", array.rank(),
            " array, but found non-array element ", j->dump(), " at position ",
            span(&shape_or_position[0], nesting_level), "."));
      }
      TENSORSTORE_RETURN_IF_ERROR(
          decode_element(*j, pointer.get()),
          MaybeAnnotateStatus(
              _,
              tensorstore::StrCat("Error parsing array element at position ",
                                  span(&shape_or_position[0], nesting_level))));
      pointer += byte_stride;
    } else {
      if (nesting_level == kMaxRank) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Nesting level exceeds maximum rank of ", kMaxRank));
      }
      path[nesting_level++] = j_array;
      const Index size = j_array->size();
      if (!array.data()) {
        shape_or_position[nesting_level - 1] = size;
        if (size == 0) {
          allocate_array();
          return array;
        }
      } else if (nesting_level > static_cast<size_t>(array.rank())) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected rank-", array.rank(), " array, but found array element ",
            j->dump(), " at position ",
            span(&shape_or_position[0], nesting_level - 1), "."));
      } else if (array.shape()[nesting_level - 1] != size) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected array of shape ", array.shape(),
            ", but found array element ", j->dump(), " of length ", size,
            " at position ", span(&shape_or_position[0], nesting_level - 1),
            "."));
      }
      j = &(*j_array)[0];
      continue;
    }
    while (true) {
      if (nesting_level == 0) {
        return array;
      }
      const array_t* j_array = path[nesting_level - 1];
      const Index size = j_array->size();
      const Index i = ++shape_or_position[nesting_level - 1];
      if (i != size) {
        j = &(*j_array)[i];
        break;
      }
      shape_or_position[nesting_level - 1] = 0;
      --nesting_level;
    }
  }
}
Result<::nlohmann::json> JsonEncodeNestedArray(ArrayView<const void> array) {
  auto convert = internal::GetDataTypeConverter(
      array.dtype(), dtype_v<::tensorstore::dtypes::json_t>);
  if (!(convert.flags & DataTypeConversionFlags::kSupported)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Conversion from ", array.dtype(), " to JSON is not implemented"));
  }
  bool error = false;
  absl::Status status;
  ::nlohmann::json j = JsonEncodeNestedArrayImpl(
      array, [&](const void* ptr) -> ::nlohmann::json {
        if ((convert.flags & DataTypeConversionFlags::kCanReinterpretCast) ==
            DataTypeConversionFlags::kCanReinterpretCast) {
          return *reinterpret_cast<const ::tensorstore::dtypes::json_t*>(ptr);
        }
        ::nlohmann::json value;
        if ((*convert.closure
                  .function)[internal::IterationBufferKind::kContiguous](
                convert.closure.context, {1, 1},
                internal::IterationBufferPointer(const_cast<void*>(ptr),
                                                 Index(0), Index(0)),
                internal::IterationBufferPointer(&value, Index(0), Index(0)),
                &status) != 1) {
          error = true;
          return nullptr;
        }
        return value;
      });
  if (error) return internal::GetElementCopyErrorStatus(std::move(status));
  return j;
}
Result<SharedArray<void>> JsonParseNestedArray(const ::nlohmann::json& j,
                                               DataType dtype,
                                               DimensionIndex rank_constraint) {
  auto convert = internal::GetDataTypeConverter(
      dtype_v<::tensorstore::dtypes::json_t>, dtype);
  if (!(convert.flags & DataTypeConversionFlags::kSupported)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Conversion from JSON to ", dtype, " is not implemented"));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array,
      JsonParseNestedArrayImpl(
          j, dtype, [&](const ::nlohmann::json& v, void* out) -> absl::Status {
            if ((convert.flags &
                 DataTypeConversionFlags::kCanReinterpretCast) ==
                DataTypeConversionFlags::kCanReinterpretCast) {
              *reinterpret_cast<::tensorstore::dtypes::json_t*>(out) = v;
              return absl::OkStatus();
            } else {
              absl::Status status;
              if ((*convert.closure
                        .function)[internal::IterationBufferKind::kContiguous](
                      convert.closure.context, {1, 1},
                      internal::IterationBufferPointer(
                          const_cast<::nlohmann::json*>(&v), Index(0),
                          Index(0)),
                      internal::IterationBufferPointer(out, Index(0), Index(0)),
                      &status) != 1) {
                return internal::GetElementCopyErrorStatus(std::move(status));
              }
              return absl::OkStatus();
            }
          }));
  if (rank_constraint != dynamic_rank && array.rank() != rank_constraint) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Array rank (", array.rank(), ") does not match expected rank (",
        rank_constraint, ")"));
  }
  return array;
}
}  
}  