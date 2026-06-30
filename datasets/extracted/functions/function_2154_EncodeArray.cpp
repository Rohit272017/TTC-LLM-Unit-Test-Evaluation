#include "tensorstore/internal/data_type_endian_conversion.h"
#include <cassert>
#include <complex>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
namespace tensorstore {
namespace internal {
void EncodeArray(ArrayView<const void> source, ArrayView<void> target,
                 endian target_endian) {
  const DataType dtype = source.dtype();
  assert(absl::c_equal(source.shape(), target.shape()));
  assert(dtype == target.dtype());
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  
  internal::IterateOverStridedLayouts<2>(
      {(target_endian == endian::native) ? functions.copy
                                                      : functions.swap_endian,
       nullptr},
      nullptr, source.shape(),
      {{const_cast<void*>(source.data()), target.data()}},
      {{source.byte_strides().data(), target.byte_strides().data()}},
      skip_repeated_elements, {{dtype.size(), dtype.size()}});
}
namespace {
static_assert(sizeof(bool) == 1);
struct DecodeBoolArray {
  void operator()(unsigned char* source, bool* output, void*) const {
    *output = static_cast<bool>(*source);
  }
};
struct DecodeBoolArrayInplace {
  void operator()(unsigned char* source, void*) const {
    *source = static_cast<bool>(*source);
  }
};
}  
void DecodeArray(ArrayView<const void> source, endian source_endian,
                 ArrayView<void> target) {
  const DataType dtype = source.dtype();
  assert(absl::c_equal(source.shape(), target.shape()));
  assert(dtype == target.dtype());
  if (dtype.id() != DataTypeId::bool_t) {
    EncodeArray(source, target, source_endian);
    return;
  }
  internal::IterateOverStridedLayouts<2>(
      {SimpleElementwiseFunction<
           DecodeBoolArray(unsigned char, bool), void*>(),
       nullptr},
      nullptr, source.shape(),
      {{const_cast<void*>(source.data()), target.data()}},
      {{source.byte_strides().data(), target.byte_strides().data()}},
      skip_repeated_elements, {{1, 1}});
}
void DecodeArray(SharedArrayView<void>* source, endian source_endian,
                 StridedLayoutView<> decoded_layout) {
  assert(source != nullptr);
  assert(absl::c_equal(source->shape(), decoded_layout.shape()));
  const DataType dtype = source->dtype();
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  
  if ((reinterpret_cast<std::uintptr_t>(source->data()) % dtype->alignment) ==
          0 &&
      std::all_of(source->byte_strides().begin(), source->byte_strides().end(),
                  [&](Index byte_stride) {
                    return (byte_stride % dtype->alignment) == 0;
                  })) {
    const ElementwiseFunction<1, void*>* convert_func = nullptr;
    if (dtype.id() == DataTypeId::bool_t) {
      convert_func =
          SimpleElementwiseFunction<DecodeBoolArrayInplace(unsigned char),
                                    void*>();
    } else if (source_endian != endian::native &&
               functions.swap_endian_inplace) {
      convert_func = functions.swap_endian_inplace;
    }
    if (convert_func) {
      internal::IterateOverStridedLayouts<1>(
          {convert_func,
           nullptr},
          nullptr, source->shape(), {{source->data()}},
          {{source->byte_strides().data()}},
          skip_repeated_elements, {{dtype.size()}});
    }
  } else {
    *source = CopyAndDecodeArray(*source, source_endian, decoded_layout);
  }
}
SharedArrayView<void> CopyAndDecodeArray(ArrayView<const void> source,
                                         endian source_endian,
                                         StridedLayoutView<> decoded_layout) {
  SharedArrayView<void> target(
      internal::AllocateAndConstructSharedElements(
          decoded_layout.num_elements(), default_init, source.dtype()),
      decoded_layout);
  DecodeArray(source, source_endian, target);
  return target;
}
SharedArrayView<const void> TryViewCordAsArray(const absl::Cord& source,
                                               Index offset, DataType dtype,
                                               endian source_endian,
                                               StridedLayoutView<> layout) {
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  
  if (source_endian != endian::native && functions.swap_endian_inplace) {
    return {};
  }
  auto maybe_flat = source.TryFlat();
  if (!maybe_flat) {
    return {};
  }
  ByteStridedPointer<const void> ptr = maybe_flat->data();
  ptr += offset;
  if ((reinterpret_cast<std::uintptr_t>(ptr.get()) % dtype->alignment) != 0 ||
      !std::all_of(layout.byte_strides().begin(), layout.byte_strides().end(),
                   [&](Index byte_stride) {
                     return (byte_stride % dtype->alignment) == 0;
                   })) {
    return {};
  }
  auto shared_cord = std::make_shared<absl::Cord>(source);
  if (auto shared_flat = shared_cord->TryFlat();
      !shared_flat || shared_flat->data() != maybe_flat->data()) {
    return {};
  }
  return SharedArrayView<const void>(
      SharedElementPointer<const void>(
          std::shared_ptr<const void>(std::move(shared_cord), ptr.get()),
          dtype),
      layout);
}
}  
}  