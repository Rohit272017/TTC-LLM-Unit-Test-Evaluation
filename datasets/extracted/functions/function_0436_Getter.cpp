#ifndef AROLLA_IO_ACCESSORS_SLOT_LISTENER_H_
#define AROLLA_IO_ACCESSORS_SLOT_LISTENER_H_
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/io/accessor_helpers.h"
#include "arolla/io/input_loader.h"
#include "arolla/io/slot_listener.h"
#include "arolla/memory/frame.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/meta.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
namespace slot_listener_impl {
template <class Accessor, class Output>
using slot_listener_accessor_input_t = std::decay_t<meta::head_t<
    typename meta::function_traits<std::decay_t<Accessor>>::arg_types>>;
template <class Output, class Accessor>
class Getter {
 public:
  using InputType = slot_listener_accessor_input_t<Accessor, Output>;
  Getter(std::optional<FrameLayout::Slot<InputType>> slot, Accessor accessor)
      : slot_(slot), accessor_(std::move(accessor)) {}
  static absl::StatusOr<Getter> Build(std::optional<TypedSlot> slot,
                                      const Accessor& accessor) {
    if (slot.has_value()) {
      ASSIGN_OR_RETURN(auto specific_slot, slot->ToSlot<InputType>());
      return {Getter({specific_slot}, accessor)};
    } else {
      return {Getter(std::nullopt, accessor)};
    }
  }
  void operator()(ConstFramePtr frame, Output* output) const {
    if (slot_.has_value()) {
      accessor_(frame.Get(*slot_), output);
    }
  }
 private:
  std::optional<FrameLayout::Slot<InputType>> slot_;
  Accessor accessor_;
};
template <class Input, class NameAccessorsTuple>
class AccessorsSlotListener;
template <class Output, class... Accessors>
class AccessorsSlotListener<Output,
                            std::tuple<std::pair<std::string, Accessors>...>>
    final : public StaticSlotListener<Output> {
  using NameAccessorsTuple = std::tuple<std::pair<std::string, Accessors>...>;
 public:
  static absl::StatusOr<std::unique_ptr<SlotListener<Output>>> Build(
      NameAccessorsTuple accessors) {
    auto loader =
        absl::WrapUnique(new AccessorsSlotListener(std::move(accessors)));
    RETURN_IF_ERROR(ValidateDuplicatedNames(loader->types_in_order()));
    return {std::move(loader)};
  }
 private:
  absl::StatusOr<BoundSlotListener<Output>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& input_slots)
      const final {
    ASSIGN_OR_RETURN(auto slots, MaybeFindSlotsAndVerifyTypes(
                                     this->types_in_order(), input_slots));
    return BindImpl(
        std::move(slots),
        std::make_index_sequence<std::tuple_size<NameAccessorsTuple>::value>{});
  }
  explicit AccessorsSlotListener(NameAccessorsTuple accessors)
      : StaticSlotListener<Output>(
            AccessorsSlotListener::CreateOutputTypesInOrder(
                accessors, std::make_index_sequence<
                               std::tuple_size<NameAccessorsTuple>::value>{})),
        accessors_(std::move(accessors)) {}
  template <size_t I>
  using Accessor =
      std::tuple_element_t<1, std::tuple_element_t<I, NameAccessorsTuple>>;
  template <size_t I>
  using SlotListenerAccessorInputType =
      slot_listener_impl::slot_listener_accessor_input_t<Accessor<I>, Output>;
  template <size_t I>
  const Accessor<I>& GetAccessor() const {
    return std::get<1>(std::get<I>(accessors_));
  }
  template <size_t I>
  static QTypePtr GetOutputType() {
    return GetQType<SlotListenerAccessorInputType<I>>();
  }
  template <size_t... Is>
  static std::vector<std::pair<std::string, QTypePtr>> CreateOutputTypesInOrder(
      const NameAccessorsTuple& accessors, std::index_sequence<Is...>) {
    return {{std::string(std::get<0>(std::get<Is>(accessors))),
             GetOutputType<Is>()}...};
  }
  template <size_t... Is>
  absl::StatusOr<BoundSlotListener<Output>> BindImpl(
      std::vector<std::optional<TypedSlot>> slots,
      std::index_sequence<Is...>) const {
    auto getters_or =
        LiftStatusUp(slot_listener_impl::Getter<Output, Accessor<Is>>::Build(
            slots[Is], GetAccessor<Is>())...);
    ASSIGN_OR_RETURN(auto getters, getters_or);
    return BoundSlotListener<Output>(
        [getters_(std::move(getters))](
            ConstFramePtr frame ABSL_ATTRIBUTE_UNUSED,
            Output* output ABSL_ATTRIBUTE_UNUSED) {
          (std::get<Is>(getters_)(frame, output), ...);
          return absl::OkStatus();
        });
  }
  NameAccessorsTuple accessors_;
};
}  
template <class Input, class NameAccessorsTuple>
using AccessorsSlotListener =
    slot_listener_impl::AccessorsSlotListener<Input, NameAccessorsTuple>;
template <class Output, class... Accessors>
absl::StatusOr<std::unique_ptr<SlotListener<Output>>>
CreateAccessorsSlotListenerFromTuple(
    std::tuple<std::pair<std::string, Accessors>...> name_accessors) {
  return AccessorsSlotListener<Output, decltype(name_accessors)>::Build(
      std::move(name_accessors));
}
template <class Output, class... NameAccessors>
absl::StatusOr<std::unique_ptr<SlotListener<Output>>>
CreateAccessorsSlotListener(NameAccessors... name_accessors) {
  return CreateAccessorsSlotListenerFromTuple<Output>(
      accessor_helpers_impl::ConvertNameAccessorsPackToNestedTuple(
          name_accessors...));
}
}  
#endif  