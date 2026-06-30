#ifndef AROLLA_IO_INPLACE_LOADER_H_
#define AROLLA_IO_INPLACE_LOADER_H_
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "arolla/memory/frame.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_slot.h"
namespace arolla {
template <class Struct>
class InplaceSlotBuilder final {
  static_assert(
      std::is_standard_layout<Struct>::value,
      "Data must be standard layout to be used with InplaceSlotBuilder.");
 public:
  using value_type = Struct;
  absl::flat_hash_map<std::string, TypedSlot> OutputSlots(
      FrameLayout::Slot<Struct> slot) const {
    absl::flat_hash_map<std::string, TypedSlot> slots;
    slots.reserve(zero_based_slots_.size());
    for (const auto& name_slot : zero_based_slots_) {
      slots.insert({name_slot.first,
                    TypedSlot::UnsafeFromOffset(
                        name_slot.second.GetType(),
                        name_slot.second.byte_offset() + slot.byte_offset())});
    }
    return slots;
  }
  absl::Status AddUnsafeField(const std::string& name, QTypePtr type,
                              size_t field_offset) {
    if (!zero_based_slots_
             .insert({name, TypedSlot::UnsafeFromOffset(type, field_offset)})
             .second) {
      return absl::FailedPreconditionError(absl::StrCat(
          "InplaceLoaderBuilder: duplicated slot name: '", name, "'"));
    }
    return absl::OkStatus();
  }
 private:
  absl::flat_hash_map<std::string, TypedSlot> zero_based_slots_;
};
#define AROLLA_ADD_INPLACE_SLOT_FIELD(builder, field, name)                 \
  builder.AddUnsafeField(                                                   \
      name,                                                                 \
      ::arolla::GetQType<                                                   \
          decltype(std::declval<decltype(builder)::value_type>().field)>(), \
      offsetof(decltype(builder)::value_type, field))
}  
#endif  