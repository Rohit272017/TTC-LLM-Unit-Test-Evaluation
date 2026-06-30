#ifndef AROLLA_UTIL_STRUCT_FIELD_H_
#define AROLLA_UTIL_STRUCT_FIELD_H_
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "arolla/util/demangle.h"
namespace arolla {
template <typename T, bool kIsSkipped = false>
struct StructField {
  static constexpr bool kIsIncludedToArollaQType = !kIsSkipped;
  static_assert(!kIsIncludedToArollaQType || !std::is_array_v<T>,
                "array field types are not supported");
  using field_type = T;
  size_t field_offset;  
  absl::string_view field_name;
};
template <class FieldType>
const FieldType* UnsafeGetStructFieldPtr(const StructField<FieldType>& field,
                                         const void* value) {
  return reinterpret_cast<const FieldType*>(static_cast<const char*>(value) +
                                            field.field_offset);
}
namespace struct_field_impl {
template <typename T, typename Enabled = void>
struct StructFieldTraits {
  static constexpr auto ArollaStructFields() { return std::tuple(); }
};
template <typename T>
struct StructFieldTraits<
    T, std::enable_if_t<std::is_invocable_v<decltype(T::ArollaStructFields)>>> {
  static auto ArollaStructFields() { return T::ArollaStructFields(); }
};
template <class T, class FieldTuple, size_t... Is>
absl::Status VerifyArollaStructFields(
    ABSL_ATTRIBUTE_UNUSED const FieldTuple& fields,
    std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) != 0) {
    auto offsets =
        std::array<size_t, sizeof...(Is)>{std::get<Is>(fields).field_offset...};
    auto alignments = std::array<size_t, sizeof...(Is)>{
        alignof(typename std::tuple_element_t<Is, FieldTuple>::field_type)...};
    auto sizes = std::array<size_t, sizeof...(Is)>{
        sizeof(typename std::tuple_element_t<Is, FieldTuple>::field_type)...};
    if (offsets[0] != 0) {
      return absl::FailedPreconditionError(
          "first struct field defined incorrectly");
    }
    if (!(((Is == 0) || (offsets[Is] > offsets[Is - 1])) && ...)) {
      return absl::FailedPreconditionError("struct fields are out of order");
    }
    auto align_offset = [](size_t offset, size_t alignment) constexpr {
      return offset +
             (offset % alignment == 0 ? 0 : alignment - offset % alignment);
    };
    if (!(((Is == 0) ||
           (offsets[Is] <=
            align_offset(offsets[Is - 1] + sizes[Is - 1], alignments[Is]))) &&
          ...)) {
      return absl::FailedPreconditionError(
          "struct field is missed in the middle");
    }
    if (align_offset(offsets.back() + sizes.back(), alignof(T)) != sizeof(T)) {
      return absl::FailedPreconditionError("struct field is missed at the end");
    }
  }
  return absl::OkStatus();
}
}  
template <class T>
const auto& GetStructFields() {
  ABSL_ATTRIBUTE_UNUSED static const bool once = [] {
    const auto fields =
        struct_field_impl::StructFieldTraits<T>::ArollaStructFields();
    constexpr size_t kSize = std::tuple_size_v<decltype(fields)>;
    CHECK_OK(struct_field_impl::VerifyArollaStructFields<T>(
        fields, std::make_index_sequence<kSize>()))
        << TypeName<T>();
    return true;
  }();
  auto filter_and_convert_to_tuple = [](auto struct_field) {
    using StructField = decltype(struct_field);
    if constexpr (StructField::kIsIncludedToArollaQType) {
      return std::tuple<StructField>{struct_field};
    } else {
      return std::tuple<>();
    }
  };
  static const auto filtered_fields = std::apply(
      [&](auto... struct_fields) {
        return std::tuple_cat(filter_and_convert_to_tuple(struct_fields)...);
      },
      struct_field_impl::StructFieldTraits<T>::ArollaStructFields());
  return filtered_fields;
}
template <class T>
constexpr size_t StructFieldCount() {
  return std::tuple_size_v<std::decay_t<decltype(GetStructFields<T>())>>;
}
template <class T>
constexpr bool HasStructFields() {
  return StructFieldCount<T>() != 0;
}
#define AROLLA_DECLARE_STRUCT_FIELD(NAME)                        \
  ::arolla::StructField<decltype(CppType::NAME)> {               \
    .field_offset = offsetof(CppType, NAME), .field_name = #NAME \
  }
#define AROLLA_SKIP_STRUCT_FIELD(NAME)                                  \
  ::arolla::StructField<decltype(CppType::NAME), true> { \
    .field_offset = offsetof(CppType, NAME), .field_name = #NAME        \
  }
}  
#endif  