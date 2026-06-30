#ifndef AROLLA_IO_ACCESSOR_HELPERS_H_
#define AROLLA_IO_ACCESSOR_HELPERS_H_
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
namespace arolla::accessor_helpers_impl {
template <class... NameAccessors>
class VariadicPackToNestedTupleImpl {
 private:
  template <size_t I>
  using AccessorType =
      std::remove_cv_t<std::remove_reference_t<typename std::tuple_element<
          I * 2 + 1, std::tuple<NameAccessors...>>::type>>;
  template <size_t I>
  AccessorType<I> Accessor() const {
    return std::get<I * 2 + 1>(name_accessors_);
  }
  template <size_t... Is>
  using NestedTupleType =
      std::tuple<std::pair<std::string, AccessorType<Is>>...>;
  template <size_t... Is>
  NestedTupleType<Is...> MakeNestedTupleImpl(std::index_sequence<Is...>) const {
    return std::make_tuple(std::make_pair(Name<Is>(), Accessor<Is>())...);
  }
  template <size_t I>
  std::string Name() const {
    return std::string(std::get<I * 2>(name_accessors_));
  }
 public:
  static_assert(
      sizeof...(NameAccessors) % 2 == 0,
      "NameAccessors must be formed as name, accessor, name, accessor, ...");
  static constexpr size_t kAccessorCount = sizeof...(NameAccessors) / 2;
  explicit VariadicPackToNestedTupleImpl(
      std::tuple<NameAccessors...> name_accessors)
      : name_accessors_(name_accessors) {}
  auto MakeNestedTuple() const
      -> decltype(MakeNestedTupleImpl(
          std::make_index_sequence<
              VariadicPackToNestedTupleImpl::kAccessorCount>())) {
    return MakeNestedTupleImpl(
        std::make_index_sequence<
            VariadicPackToNestedTupleImpl::kAccessorCount>());
  }
 private:
  std::tuple<NameAccessors...> name_accessors_;
};
template <class... NameAccessors>
auto ConvertNameAccessorsPackToNestedTuple(NameAccessors... name_accessors)
    -> decltype(VariadicPackToNestedTupleImpl<NameAccessors...>(
                    std::make_tuple(name_accessors...))
                    .MakeNestedTuple()) {
  return VariadicPackToNestedTupleImpl<NameAccessors...>(
             std::forward_as_tuple(name_accessors...))
      .MakeNestedTuple();
}
}  
#endif  