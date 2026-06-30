#ifndef AROLLA_IO_CHAIN_SLOT_LISTENER_H_
#define AROLLA_IO_CHAIN_SLOT_LISTENER_H_
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "arolla/io/slot_listener.h"
#include "arolla/memory/frame.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <class Output>
class ChainSlotListener final : public SlotListener<Output> {
  struct PrivateConstructorTag {};
 public:
  template <class... ListenerTs>
  static absl::StatusOr<std::unique_ptr<SlotListener<Output>>> Build(
      std::unique_ptr<ListenerTs>... listeners) {
    std::vector<std::unique_ptr<SlotListener<Output>>> listeners_vec;
    (listeners_vec.push_back(std::move(listeners)), ...);
    return Build(std::move(listeners_vec));
  }
  static absl::StatusOr<std::unique_ptr<SlotListener<Output>>> Build(
      std::vector<std::unique_ptr<SlotListener<Output>>> listeners) {
    return std::make_unique<ChainSlotListener>(PrivateConstructorTag{},
                                               std::move(listeners));
  }
  absl::Nullable<const QType*> GetQTypeOf(
      absl::string_view name,
      absl::Nullable<const QType*> desired_qtype) const final {
    for (const auto& listener : listeners_) {
      if (auto qtype = listener->GetQTypeOf(name, desired_qtype);
          qtype != nullptr) {
        return qtype;
      }
    }
    return nullptr;
  }
  std::vector<std::string> SuggestAvailableNames() const final {
    std::vector<std::string> names;
    for (const auto& listener : listeners_) {
      auto available = listener->SuggestAvailableNames();
      names.insert(names.end(), available.begin(), available.end());
    }
    return names;
  }
  explicit ChainSlotListener(
      PrivateConstructorTag,
      std::vector<std::unique_ptr<SlotListener<Output>>> listeners)
      : listeners_(std::move(listeners)) {}
 private:
  absl::StatusOr<BoundSlotListener<Output>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& slots) const final {
    std::vector<BoundSlotListener<Output>> bound_listeners;
    bound_listeners.reserve(listeners_.size());
    for (const auto& listener : listeners_) {
      ASSIGN_OR_RETURN(std::optional<BoundSlotListener<Output>> bound_listener,
                       listener->PartialBind(slots));
      if (bound_listener.has_value()) {
        bound_listeners.push_back(*std::move(bound_listener));
      }
    }
    if (bound_listeners.empty()) {
      return BoundSlotListener<Output>(
          [](ConstFramePtr, Output*) { return absl::OkStatus(); });
    }
    if (bound_listeners.size() == 1) {
      return std::move(bound_listeners[0]);
    }
    return BoundSlotListener<Output>(
        [bound_listeners(std::move(bound_listeners))](
            ConstFramePtr frame, Output* output) -> absl::Status {
          for (const auto& listener : bound_listeners) {
            RETURN_IF_ERROR(listener(frame, output));
          }
          return absl::OkStatus();
        });
  }
  std::vector<std::unique_ptr<SlotListener<Output>>> listeners_;
};
}  
#endif  