#ifndef THIRD_PARTY_CEL_CPP_COMMON_INTERNAL_DATA_INTERFACE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_INTERNAL_DATA_INTERFACE_H_
#include <type_traits>
#include "absl/base/attributes.h"
#include "common/native_type.h"
namespace cel {
class TypeInterface;
class ValueInterface;
namespace common_internal {
class DataInterface;
class DataInterface {
 public:
  DataInterface(const DataInterface&) = delete;
  DataInterface(DataInterface&&) = delete;
  virtual ~DataInterface() = default;
  DataInterface& operator=(const DataInterface&) = delete;
  DataInterface& operator=(DataInterface&&) = delete;
 protected:
  DataInterface() = default;
 private:
  friend class cel::TypeInterface;
  friend class cel::ValueInterface;
  friend struct NativeTypeTraits<DataInterface>;
  virtual NativeTypeId GetNativeTypeId() const = 0;
};
}  
template <>
struct NativeTypeTraits<common_internal::DataInterface> final {
  static NativeTypeId Id(const common_internal::DataInterface& data_interface) {
    return data_interface.GetNativeTypeId();
  }
};
template <typename T>
struct NativeTypeTraits<
    T, std::enable_if_t<std::conjunction_v<
           std::is_base_of<common_internal::DataInterface, T>,
           std::negation<std::is_same<T, common_internal::DataInterface>>>>>
    final {
  static NativeTypeId Id(const common_internal::DataInterface& data_interface) {
    return NativeTypeTraits<common_internal::DataInterface>::Id(data_interface);
  }
};
}  
#endif  