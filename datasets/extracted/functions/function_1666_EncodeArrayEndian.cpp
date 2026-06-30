#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <memory>
#include <string_view>
#include <utility>
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/base/chain.h"
#include "riegeli/bytes/copy_all.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
using ::tensorstore::internal_metrics::MetricMetadata;
namespace tensorstore {
namespace internal {
namespace {
auto& contiguous_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/internal/riegeli/contiguous_bytes",
    MetricMetadata("Endian codec bytes from contiguous buffers",
                   internal_metrics::Units::kBytes));
auto& noncontiguous_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/internal/riegeli/noncontiguous_bytes",
    MetricMetadata("Endian codec bytes from non-contiguous buffers",
                   internal_metrics::Units::kBytes));
}  
[[nodiscard]] bool EncodeArrayEndian(SharedArrayView<const void> decoded,
                                     endian encoded_endian,
                                     ContiguousLayoutOrder order,
                                     riegeli::Writer& writer) {
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(decoded.dtype().id())];
  assert(functions.copy != nullptr);  
  if ((encoded_endian == endian::native ||
       functions.swap_endian_inplace == nullptr) &&
      IsContiguousLayout(decoded, order)) {
    const size_t length = decoded.num_elements() * decoded.dtype().size();
    if (writer.PrefersCopying()) {
      return writer.Write(std::string_view(
          reinterpret_cast<const char*>(decoded.data()), length));
    }
    return writer.Write(
        internal::MakeCordFromSharedPtr(std::move(decoded.pointer()), length));
  }
  const internal::ElementwiseFunction<1, void*>* write_func =
      encoded_endian == endian::native ? &functions.write_native_endian
                                       : &functions.write_swapped_endian;
  return internal::IterateOverArrays(
      {write_func, &writer},
      nullptr, {order, include_repeated_elements}, decoded);
}
namespace {
class ContiguousBufferSinkWriter : public riegeli::Writer {
 public:
  std::shared_ptr<const void> data;
  size_t expected_length;
  size_t expected_alignment;
  void DoFail() { Fail(absl::UnimplementedError("")); }
  bool PushSlow(size_t min_length, size_t recommended_length) override {
    DoFail();
    return false;
  }
  bool ValidateContiguousBuffer(std::string_view buf) {
    if (buf.size() != expected_length ||
        (reinterpret_cast<uintptr_t>(buf.data()) % expected_alignment) != 0) {
      DoFail();
      return false;
    }
    return true;
  }
  template <typename T>
  bool WriteCordLike(T&& src) {
    if (this->data) {
      DoFail();
      return false;
    }
    auto buf = src.TryFlat();
    if (!buf) {
      DoFail();
      return false;
    }
    if (!ValidateContiguousBuffer(*buf)) return false;
    auto data = std::make_shared<absl::remove_cvref_t<T>>(std::forward<T>(src));
    buf = data->TryFlat();
    if (!buf) {
      DoFail();
      return false;
    }
    if (!ValidateContiguousBuffer(*buf)) return false;
    this->data = std::shared_ptr<const void>(std::move(data), buf->data());
    return true;
  }
  bool WriteSlow(const riegeli::Chain& src) override {
    return WriteCordLike(src);
  }
  bool WriteSlow(const absl::Cord& src) override { return WriteCordLike(src); }
};
}  
Result<SharedArray<const void>> DecodeArrayEndian(
    riegeli::Reader& reader, DataType dtype, span<const Index> decoded_shape,
    endian encoded_endian, ContiguousLayoutOrder order) {
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  
  size_t expected_length = dtype.size() * ProductOfExtents(decoded_shape);
  const auto may_be_contiguous = [&] {
    if (encoded_endian != endian::native &&
        functions.swap_endian_inplace != nullptr) {
      return false;
    }
    if (!reader.SupportsRewind()) {
      return false;
    }
    if (!reader.SupportsSize()) {
      return false;
    }
    auto size_opt = reader.Size();
    if (!size_opt) return false;
    if (*size_opt < expected_length ||
        *size_opt - expected_length != reader.pos()) {
      return false;
    }
    return true;
  };
  if (may_be_contiguous()) {
    auto pos = reader.pos();
    ContiguousBufferSinkWriter buffer_sink_writer;
    buffer_sink_writer.expected_length = expected_length;
    buffer_sink_writer.expected_alignment = dtype->alignment;
    if (riegeli::CopyAll(reader, buffer_sink_writer, expected_length).ok()) {
      absl::Status status;
      if (functions.validate) {
        if (!(*functions.validate)[IterationBufferKind::kContiguous](
                nullptr, {1, static_cast<Index>(expected_length)},
                IterationBufferPointer(
                    const_cast<void*>(buffer_sink_writer.data.get()), 0,
                    dtype.size()),
                &status)) {
          return status;
        }
      }
      contiguous_bytes.IncrementBy(expected_length);
      return tensorstore::SharedArray<const void>(
          SharedElementPointer<const void>(std::move(buffer_sink_writer.data),
                                           dtype),
          decoded_shape, order);
    }
    if (!reader.Seek(pos)) {   
      return reader.status();  
    }
  }
  auto decoded =
      tensorstore::AllocateArray(decoded_shape, order, default_init, dtype);
  TENSORSTORE_RETURN_IF_ERROR(
      DecodeArrayEndian(reader, encoded_endian, order, decoded));
  reader.VerifyEnd();
  if (!reader.ok()) {
    return reader.status();
  }
  noncontiguous_bytes.IncrementBy(expected_length);
  return decoded;
}
absl::Status DecodeArrayEndian(riegeli::Reader& reader, endian encoded_endian,
                               ContiguousLayoutOrder order,
                               ArrayView<void> decoded) {
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(decoded.dtype().id())];
  assert(functions.copy != nullptr);  
  riegeli::LimitingReader limiting_reader(
      &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                   decoded.dtype().size() * decoded.num_elements()));
  [[maybe_unused]] const auto unused_result = internal::IterateOverArrays(
      {encoded_endian == endian::native ? &functions.read_native_endian
                                        : &functions.read_swapped_endian,
       &limiting_reader},
      nullptr, {order, include_repeated_elements}, decoded);
  if (!limiting_reader.VerifyEndAndClose()) {
    return limiting_reader.status();
  }
  return absl::OkStatus();
}
}  
}  