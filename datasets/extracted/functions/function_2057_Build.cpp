#ifndef AROLLA_IO_DELEGATING_SLOT_LISTENER_H_
#define AROLLA_IO_DELEGATING_SLOT_LISTENER_H_
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "arolla/io/input_loader.h"
#include "arolla/io/slot_listener.h"
#include "arolla/memory/frame.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
namespace delegating_output_listener_impl {
template <class Output, class DelegateOutput>
class DelegatingSlotListener final : public SlotListener<Output> {
  struct PrivateConstructorTag {};
 public:
  template <class Accessor>
  static absl::StatusOr<std::unique_ptr<SlotListener<Output>>> Build(
      std::unique_ptr<SlotListener<DelegateOutput>> delegate_listener,
      const Accessor& accessor, std::string name_prefix) {
    static_assert(std::is_same_v<decltype(accessor(std::declval<Output*>())),
                                 DelegateOutput*>,
                  "Accessor must have `DelegateOutput*` result type.");
    return absl::make_unique<DelegatingSlotListener>(
        PrivateConstructorTag{}, std::move(delegate_listener), accessor,
        std::move(name_prefix));
  }
  absl::Nullable<const QType*> GetQTypeOf(
      absl::string_view name,
      absl::Nullable<const QType*> desired_qtype) const final {
    if (!absl::ConsumePrefix(&name, name_prefix_)) {
      return nullptr;
    }
    return delegate_listener_->GetQTypeOf(name, desired_qtype);
  }
  std::vector<std::string> SuggestAvailableNames() const final {
    std::vector<std::string> names =
        delegate_listener_->SuggestAvailableNames();
    for (auto& name : names) {
      name = absl::StrCat(name_prefix_, name);
    }
    return names;
  }
  DelegatingSlotListener(
      PrivateConstructorTag,
      std::unique_ptr<SlotListener<DelegateOutput>> delegate_listener,
      std::function<DelegateOutput*(Output*)> accessor, std::string name_prefix)
      : delegate_listener_(std::move(delegate_listener)),
        accessor_(accessor),
        name_prefix_(name_prefix) {}
 private:
  absl::StatusOr<BoundSlotListener<Output>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& input_slots)
      const final {
    absl::flat_hash_map<std::string, TypedSlot> delegate_input_slots;
    for (const auto& [name, slot] : input_slots) {
      absl::string_view name_view = name;
      if (absl::ConsumePrefix(&name_view, name_prefix_)) {
        delegate_input_slots.emplace(std::string(name_view), slot);
      }
    }
    ASSIGN_OR_RETURN(BoundSlotListener<DelegateOutput> bound_delegate_listener,
                     delegate_listener_->Bind(delegate_input_slots));
    return BoundSlotListener<Output>(
        [bound_delegate_listener(std::move(bound_delegate_listener)),
         accessor(accessor_)](ConstFramePtr frame,
                              Output* output) -> absl::Status {
          return bound_delegate_listener(frame, accessor(output));
        });
  }
  std::unique_ptr<SlotListener<DelegateOutput>> delegate_listener_;
  std::function<DelegateOutput*(Output*)> accessor_;
  absl::flat_hash_map<std::string, QTypePtr> types_;
  std::string name_prefix_;
};
}  
template <class Output, class DelegateOutput, class Accessor>
absl::StatusOr<std::unique_ptr<SlotListener<Output>>>
CreateDelegatingSlotListener(
    std::unique_ptr<SlotListener<DelegateOutput>> delegate_listener,
    const Accessor& accessor, std::string name_prefix = "") {
  return delegating_output_listener_impl::DelegatingSlotListener<
      Output, DelegateOutput>::Build(std::move(delegate_listener), accessor,
                                     std::move(name_prefix));
}
}  
#endif  