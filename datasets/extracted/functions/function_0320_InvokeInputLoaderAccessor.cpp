#ifndef AROLLA_IO_ACCESSORS_INPUT_LOADER_H_
#define AROLLA_IO_ACCESSORS_INPUT_LOADER_H_
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/io/accessor_helpers.h"
#include "arolla/io/input_loader.h"
#include "arolla/memory/frame.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/meta.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <class Accessor, class Input, class Output>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void InvokeInputLoaderAccessor(
    const Accessor& accessor, const Input& input, RawBufferFactory* factory,
    Output* output) {
  if constexpr (std::is_invocable_v<const Accessor&, const Input&,
                                    RawBufferFactory*, Output*>) {
    accessor(input, factory, output);
  } else if constexpr (std::is_invocable_v<const Accessor&, const Input&,
                                           Output*>) {
    ((void)(factory));
    accessor(input, output);
  } else if constexpr (std::is_invocable_v<const Accessor&, const Input&,
                                           RawBufferFactory*>) {
    *output = accessor(input, factory);
  } else if constexpr (std::is_invocable_v<const Accessor&, const Input&>) {
    ((void)(factory));
    *output = accessor(input);
  }
}
namespace input_loader_impl {
template <class Accessor, class Input>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline auto InvokeInputLoaderAccessorTypeMeta() {
  if constexpr (std::is_invocable_v<const Accessor&, const Input&,
                                    RawBufferFactory*>) {
    return std::invoke_result<const Accessor&, const Input&,
                              RawBufferFactory*>();
  } else if constexpr (std::is_invocable_v<const Accessor&, const Input&>) {
    return std::invoke_result<const Accessor&, const Input&>();
  } else {
    using info = meta::function_traits<std::decay_t<Accessor>>;
    if constexpr (info::arity == 2) {
      using Output = std::remove_pointer_t<
          std::tuple_element_t<1, typename info::arg_types::tuple>>;
      static_assert(std::is_invocable_v<const Accessor&, const Input&, Output*>,
                    "Unexpected accessor signature.");
      return meta::type<Output>();
    } else {
      using Output = std::remove_pointer_t<
          std::tuple_element_t<2, typename info::arg_types::tuple>>;
      static_assert(std::is_invocable_v<const Accessor&, const Input&,
                                        RawBufferFactory*, Output*>,
                    "Unexpected accessor signature.");
      return meta::type<Output>();
    }
  }
}
}  
template <class Accessor, class Input>
using InputLoaderAccessorResultType = std::decay_t<
    typename decltype(input_loader_impl::InvokeInputLoaderAccessorTypeMeta<
                      const Accessor&, const Input&>())::type>;
namespace input_loader_impl {
template <class Input, class NameAccessorsTuple>
class AccessorsInputLoader;
template <class Input, class Accessor>
class Setter {
 public:
  using ResultType = InputLoaderAccessorResultType<Accessor, Input>;
  Setter(std::optional<FrameLayout::Slot<ResultType>> slot, Accessor accessor)
      : slot_(slot), accessor_(std::move(accessor)) {}
  static absl::StatusOr<Setter> Build(std::optional<TypedSlot> slot,
                                      const Accessor& accessor) {
    if (slot.has_value()) {
      ASSIGN_OR_RETURN(auto specific_slot, slot->ToSlot<ResultType>());
      return {Setter({specific_slot}, accessor)};
    } else {
      return {Setter(std::nullopt, accessor)};
    }
  }
  void operator()(const Input& input, FramePtr frame,
                  RawBufferFactory* factory) const {
    if (slot_.has_value()) {
      InvokeInputLoaderAccessor(accessor_, input, factory,
                                frame.GetMutable(*slot_));
    }
  }
 private:
  std::optional<FrameLayout::Slot<ResultType>> slot_;
  Accessor accessor_;
};
template <class Input, class... Accessors>
class AccessorsInputLoader<Input,
                           std::tuple<std::pair<std::string, Accessors>...>>
    final : public StaticInputLoader<Input> {
  using NameAccessorsTuple = std::tuple<std::pair<std::string, Accessors>...>;
 public:
  static absl::StatusOr<InputLoaderPtr<Input>> Build(
      NameAccessorsTuple accessors) {
    auto output_types_in_order = CreateOutputTypesInOrder(
        accessors,
        std::make_index_sequence<std::tuple_size<NameAccessorsTuple>::value>{});
    RETURN_IF_ERROR(ValidateDuplicatedNames(output_types_in_order));
    return InputLoaderPtr<Input>(
        static_cast<InputLoader<Input>*>(new AccessorsInputLoader(
            std::move(accessors), std::move(output_types_in_order))));
  }
  absl::StatusOr<BoundInputLoader<Input>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& output_slots)
      const final {
    ASSIGN_OR_RETURN(auto slots, MaybeFindSlotsAndVerifyTypes(
                                     this->types_in_order(), output_slots));
    return BindImpl(
        std::move(slots),
        std::make_index_sequence<std::tuple_size<NameAccessorsTuple>::value>{});
  }
 private:
  explicit AccessorsInputLoader(
      NameAccessorsTuple accessors,
      std::vector<std::pair<std::string, QTypePtr>> output_types_in_order)
      : StaticInputLoader<Input>(std::move(output_types_in_order)),
        accessors_(std::move(accessors)) {}
  template <size_t I>
  using Accessor =
      std::tuple_element_t<1, std::tuple_element_t<I, NameAccessorsTuple>>;
  template <size_t I>
  using InputLoaderAccessorResultType =
      InputLoaderAccessorResultType<Accessor<I>, Input>;
  template <size_t I>
  const Accessor<I>& GetAccessor() const {
    return std::get<1>(std::get<I>(accessors_));
  }
  template <size_t I>
  static QTypePtr GetOutputType() {
    return GetQType<InputLoaderAccessorResultType<I>>();
  }
  template <size_t... Is>
  static std::vector<std::pair<std::string, QTypePtr>> CreateOutputTypesInOrder(
      const NameAccessorsTuple& accessors, std::index_sequence<Is...>) {
    return {{std::string(std::get<0>(std::get<Is>(accessors))),
             GetOutputType<Is>()}...};
  }
  template <size_t... Is>
  absl::StatusOr<BoundInputLoader<Input>> BindImpl(
      std::vector<std::optional<TypedSlot>> slots,
      std::index_sequence<Is...>) const {
    auto setters_or = LiftStatusUp(
        Setter<Input, Accessor<Is>>::Build(slots[Is], GetAccessor<Is>())...);
    ASSIGN_OR_RETURN(auto setters, setters_or);
    return BoundInputLoader<Input>(
        [setters_(std::move(setters))](
            const Input& input ABSL_ATTRIBUTE_UNUSED,
            FramePtr frame ABSL_ATTRIBUTE_UNUSED,
            RawBufferFactory* factory ABSL_ATTRIBUTE_UNUSED) {
          (std::get<Is>(setters_)(input, frame, factory), ...);
          return absl::OkStatus();
        });
  }
  NameAccessorsTuple accessors_;
};
}  
template <class Input, class NameAccessorsTuple>
using AccessorsInputLoader =
    input_loader_impl::AccessorsInputLoader<Input, NameAccessorsTuple>;
template <class Input, class... Accessors>
absl::StatusOr<InputLoaderPtr<Input>> CreateAccessorsInputLoaderFromTuple(
    std::tuple<std::pair<std::string, Accessors>...> name_accessors) {
  return AccessorsInputLoader<Input, decltype(name_accessors)>::Build(
      std::move(name_accessors));
}
template <class Input, class... NameAccessors>
absl::StatusOr<InputLoaderPtr<Input>> CreateAccessorsInputLoader(
    NameAccessors... name_accessors) {
  return CreateAccessorsInputLoaderFromTuple<Input>(
      accessor_helpers_impl::ConvertNameAccessorsPackToNestedTuple(
          name_accessors...));
}
}  
#endif  