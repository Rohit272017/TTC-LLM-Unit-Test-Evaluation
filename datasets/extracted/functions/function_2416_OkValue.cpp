#ifndef QUICHE_COMMON_WIRE_SERIALIZATION_H_
#define QUICHE_COMMON_WIRE_SERIALIZATION_H_
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/quiche_buffer_allocator.h"
#include "quiche/common/quiche_data_writer.h"
#include "quiche/common/quiche_status_utils.h"
namespace quiche {
template <typename T>
class QUICHE_NO_EXPORT SerializeIntoWriterStatus {
 public:
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) <= 32,
                "The types passed into SerializeInto() APIs are passed by "
                "value; if your type has non-trivial copy costs, it should be "
                "wrapped into a type that carries a pointer");
  using Type = decltype(std::declval<T>().SerializeIntoWriter(
      std::declval<QuicheDataWriter&>()));
  static constexpr bool kIsBool = std::is_same_v<Type, bool>;
  static constexpr bool kIsStatus = std::is_same_v<Type, absl::Status>;
  static_assert(
      kIsBool || kIsStatus,
      "SerializeIntoWriter() has to return either a bool or an absl::Status");
  static ABSL_ATTRIBUTE_ALWAYS_INLINE Type OkValue() {
    if constexpr (kIsStatus) {
      return absl::OkStatus();
    } else {
      return true;
    }
  }
};
inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool IsWriterStatusOk(bool status) {
  return status;
}
inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool IsWriterStatusOk(
    const absl::Status& status) {
  return status.ok();
}
template <typename T>
class QUICHE_EXPORT WireFixedSizeIntBase {
 public:
  using DataType = T;
  static_assert(std::is_integral_v<DataType>,
                "WireFixedSizeIntBase is only usable with integral types");
  explicit WireFixedSizeIntBase(T value) { value_ = value; }
  size_t GetLengthOnWire() const { return sizeof(T); }
  T value() const { return value_; }
 private:
  T value_;
};
class QUICHE_EXPORT WireUint8 : public WireFixedSizeIntBase<uint8_t> {
 public:
  using WireFixedSizeIntBase::WireFixedSizeIntBase;
  bool SerializeIntoWriter(QuicheDataWriter& writer) const {
    return writer.WriteUInt8(value());
  }
};
class QUICHE_EXPORT WireUint16 : public WireFixedSizeIntBase<uint16_t> {
 public:
  using WireFixedSizeIntBase::WireFixedSizeIntBase;
  bool SerializeIntoWriter(QuicheDataWriter& writer) const {
    return writer.WriteUInt16(value());
  }
};
class QUICHE_EXPORT WireUint32 : public WireFixedSizeIntBase<uint32_t> {
 public:
  using WireFixedSizeIntBase::WireFixedSizeIntBase;
  bool SerializeIntoWriter(QuicheDataWriter& writer) const {
    return writer.WriteUInt32(value());
  }
};
class QUICHE_EXPORT WireUint64 : public WireFixedSizeIntBase<uint64_t> {
 public:
  using WireFixedSizeIntBase::WireFixedSizeIntBase;
  bool SerializeIntoWriter(QuicheDataWriter& writer) const {
    return writer.WriteUInt64(value());
  }
};
class QUICHE_EXPORT WireVarInt62 {
 public:
  using DataType = uint64_t;
  explicit WireVarInt62(uint64_t value) { value_ = value; }
  template <typename T>
  explicit WireVarInt62(T value) {
    static_assert(std::is_enum_v<T> || std::is_convertible_v<T, uint64_t>);
    value_ = static_cast<uint64_t>(value);
  }
  size_t GetLengthOnWire() const {
    return QuicheDataWriter::GetVarInt62Len(value_);
  }
  bool SerializeIntoWriter(QuicheDataWriter& writer) const {
    return writer.WriteVarInt62(value_);
  }
 private:
  uint64_t value_;
};
class QUICHE_EXPORT WireBytes {
 public:
  using DataType = absl::string_view;
  explicit WireBytes(absl::string_view value) { value_ = value; }
  size_t GetLengthOnWire() { return value_.size(); }
  bool SerializeIntoWriter(QuicheDataWriter& writer) {
    return writer.WriteStringPiece(value_);
  }
 private:
  absl::string_view value_;
};
template <class LengthWireType>
class QUICHE_EXPORT WireStringWithLengthPrefix {
 public:
  using DataType = absl::string_view;
  explicit WireStringWithLengthPrefix(absl::string_view value) {
    value_ = value;
  }
  size_t GetLengthOnWire() {
    return LengthWireType(value_.size()).GetLengthOnWire() + value_.size();
  }
  absl::Status SerializeIntoWriter(QuicheDataWriter& writer) {
    if (!LengthWireType(value_.size()).SerializeIntoWriter(writer)) {
      return absl::InternalError("Failed to serialize the length prefix");
    }
    if (!writer.WriteStringPiece(value_)) {
      return absl::InternalError("Failed to serialize the string proper");
    }
    return absl::OkStatus();
  }
 private:
  absl::string_view value_;
};
using WireStringWithVarInt62Length = WireStringWithLengthPrefix<WireVarInt62>;
template <typename WireType, typename InnerType = typename WireType::DataType>
class QUICHE_EXPORT WireOptional {
 public:
  using DataType = std::optional<InnerType>;
  using Status = SerializeIntoWriterStatus<WireType>;
  explicit WireOptional(DataType value) { value_ = value; }
  size_t GetLengthOnWire() const {
    return value_.has_value() ? WireType(*value_).GetLengthOnWire() : 0;
  }
  typename Status::Type SerializeIntoWriter(QuicheDataWriter& writer) const {
    if (value_.has_value()) {
      return WireType(*value_).SerializeIntoWriter(writer);
    }
    return Status::OkValue();
  }
 private:
  DataType value_;
};
template <typename WireType,
          typename SpanElementType = typename WireType::DataType>
class QUICHE_EXPORT WireSpan {
 public:
  using DataType = absl::Span<const SpanElementType>;
  explicit WireSpan(DataType value) { value_ = value; }
  size_t GetLengthOnWire() const {
    size_t total = 0;
    for (const SpanElementType& value : value_) {
      total += WireType(value).GetLengthOnWire();
    }
    return total;
  }
  absl::Status SerializeIntoWriter(QuicheDataWriter& writer) const {
    for (size_t i = 0; i < value_.size(); i++) {
      auto status = WireType(value_[i]).SerializeIntoWriter(writer);
      if (IsWriterStatusOk(status)) {
        continue;
      }
      if constexpr (SerializeIntoWriterStatus<WireType>::kIsStatus) {
        return AppendToStatus(std::move(status),
                              " while serializing the value #", i);
      } else {
        return absl::InternalError(
            absl::StrCat("Failed to serialize vector value #", i));
      }
    }
    return absl::OkStatus();
  }
 private:
  DataType value_;
};
namespace wire_serialization_internal {
template <typename T>
auto SerializeIntoWriterWrapper(QuicheDataWriter& writer, int argno, T data) {
#if defined(NDEBUG)
  (void)argno;
  (void)data;
  return data.SerializeIntoWriter(writer);
#else
  const size_t initial_offset = writer.length();
  const size_t expected_size = data.GetLengthOnWire();
  auto result = data.SerializeIntoWriter(writer);
  const size_t final_offset = writer.length();
  if (IsWriterStatusOk(result)) {
    QUICHE_DCHECK_EQ(initial_offset + expected_size, final_offset)
        << "while serializing field #" << argno;
  }
  return result;
#endif
}
template <typename T>
std::enable_if_t<SerializeIntoWriterStatus<T>::kIsBool, absl::Status>
SerializeIntoWriterCore(QuicheDataWriter& writer, int argno, T data) {
  const bool success = SerializeIntoWriterWrapper(writer, argno, data);
  if (!success) {
    return absl::InternalError(
        absl::StrCat("Failed to serialize field #", argno));
  }
  return absl::OkStatus();
}
template <typename T>
std::enable_if_t<SerializeIntoWriterStatus<T>::kIsStatus, absl::Status>
SerializeIntoWriterCore(QuicheDataWriter& writer, int argno, T data) {
  return AppendToStatus(SerializeIntoWriterWrapper(writer, argno, data),
                        " while serializing field #", argno);
}
template <typename T1, typename... Ts>
absl::Status SerializeIntoWriterCore(QuicheDataWriter& writer, int argno,
                                     T1 data1, Ts... rest) {
  QUICHE_RETURN_IF_ERROR(SerializeIntoWriterCore(writer, argno, data1));
  return SerializeIntoWriterCore(writer, argno + 1, rest...);
}
inline absl::Status SerializeIntoWriterCore(QuicheDataWriter&, int) {
  return absl::OkStatus();
}
}  
template <typename... Ts>
absl::Status SerializeIntoWriter(QuicheDataWriter& writer, Ts... data) {
  return wire_serialization_internal::SerializeIntoWriterCore(
      writer, 0, data...);
}
template <typename T>
size_t ComputeLengthOnWire(T data) {
  return data.GetLengthOnWire();
}
template <typename T1, typename... Ts>
size_t ComputeLengthOnWire(T1 data1, Ts... rest) {
  return data1.GetLengthOnWire() + ComputeLengthOnWire(rest...);
}
inline size_t ComputeLengthOnWire() { return 0; }
template <typename... Ts>
absl::StatusOr<QuicheBuffer> SerializeIntoBuffer(
    QuicheBufferAllocator* allocator, Ts... data) {
  size_t buffer_size = ComputeLengthOnWire(data...);
  if (buffer_size == 0) {
    return QuicheBuffer();
  }
  QuicheBuffer buffer(allocator, buffer_size);
  QuicheDataWriter writer(buffer.size(), buffer.data());
  QUICHE_RETURN_IF_ERROR(SerializeIntoWriter(writer, data...));
  if (writer.remaining() != 0) {
    return absl::InternalError(absl::StrCat(
        "Excess ", writer.remaining(), " bytes allocated while serializing"));
  }
  return buffer;
}
template <typename... Ts>
absl::StatusOr<std::string> SerializeIntoString(Ts... data) {
  size_t buffer_size = ComputeLengthOnWire(data...);
  if (buffer_size == 0) {
    return std::string();
  }
  std::string buffer;
  buffer.resize(buffer_size);
  QuicheDataWriter writer(buffer.size(), buffer.data());
  QUICHE_RETURN_IF_ERROR(SerializeIntoWriter(writer, data...));
  if (writer.remaining() != 0) {
    return absl::InternalError(absl::StrCat(
        "Excess ", writer.remaining(), " bytes allocated while serializing"));
  }
  return buffer;
}
}  
#endif  