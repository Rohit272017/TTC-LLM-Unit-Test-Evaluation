#ifndef AROLLA_IO_TUPLE_INPUT_LOADER_H_
#define AROLLA_IO_TUPLE_INPUT_LOADER_H_
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "arolla/io/input_loader.h"
#include "arolla/memory/frame.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
namespace arolla {
template <typename Input>
class TupleInputLoader;
template <typename... Ts>
class TupleInputLoader<std::tuple<Ts...>> final
    : public StaticInputLoader<std::tuple<Ts...>> {
 public:
  using Input = std::tuple<Ts...>;
  static absl::StatusOr<InputLoaderPtr<Input>> Create(
      std::vector<std::string> arg_names) {
    if (arg_names.size() != sizeof...(Ts)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("tuple size doesn't match arg_names size: %d vs %d",
                          sizeof...(Ts), arg_names.size()));
    }
    return InputLoaderPtr<Input>(
        static_cast<InputLoader<Input>*>(new TupleInputLoader<Input>(
            std::move(arg_names), std::index_sequence_for<Ts...>{})));
  }
 private:
  template <size_t... Is>
  explicit TupleInputLoader(std::vector<std::string> arg_names,
                            std::index_sequence<Is...>)
      : StaticInputLoader<std::tuple<Ts...>>(
            {{arg_names[Is], ::arolla::GetQType<Ts>()}...}) {}
  absl::StatusOr<BoundInputLoader<Input>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& output_slots)
      const override {
    std::vector<TypedSlot> slots_in_order;
    slots_in_order.reserve(this->types_in_order().size());
    for (const auto& [name, _] : this->types_in_order()) {
      auto it = output_slots.find(name);
      if (it == output_slots.end()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "TupleInputLoader doesn't support unused arguments; no slot for: ",
            name));
      }
      slots_in_order.push_back(it->second);
    }
    return BoundInputLoader<Input>(
        [slots_in_order](const Input& input, FramePtr frame,
                         RawBufferFactory*) -> absl::Status {
          LoaderImpl(input, frame, slots_in_order,
                     std::index_sequence_for<Ts...>{});
          return absl::OkStatus();
        });
  }
  template <size_t... Is>
  static void LoaderImpl(const Input& input, FramePtr frame,
                         const std::vector<TypedSlot>& slots,
                         std::index_sequence<Is...>) {
    (frame.Set(slots[Is].UnsafeToSlot<Ts>(), std::get<Is>(input)), ...);
  }
};
}  
#endif  