#include "arolla/io/typed_refs_input_loader.h"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/io/input_loader.h"
#include "arolla/memory/frame.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_ref.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
namespace {
using Input = absl::Span<const TypedRef>;
class TypedRefsInputLoader : public StaticInputLoader<Input> {
 public:
  explicit TypedRefsInputLoader(
      std::vector<std::pair<std::string, QTypePtr>> args)
      : StaticInputLoader<Input>(std::move(args)) {}
 private:
  absl::StatusOr<BoundInputLoader<Input>> BindImpl(
      const absl::flat_hash_map<std::string, TypedSlot>& output_slots)
      const override {
    std::vector<size_t> element_ids;
    std::vector<TypedSlot> slots;
    element_ids.reserve(output_slots.size());
    slots.reserve(output_slots.size());
    for (size_t i = 0; i != types_in_order().size(); ++i) {
      if (auto it = output_slots.find(types_in_order()[i].first);
          it != output_slots.end()) {
        element_ids.push_back(i);
        slots.push_back(it->second);
      }
    }
    return BoundInputLoader<Input>(
        [slots = std::move(slots), element_ids = std::move(element_ids),
         expected_input_size = types_in_order().size()](
            const Input& input, FramePtr frame,
            RawBufferFactory*) -> absl::Status {
          if (input.size() != expected_input_size) {
            return absl::InvalidArgumentError(
                absl::StrFormat("unexpected input count: expected %d, got %d",
                                expected_input_size, input.size()));
          }
          for (size_t i = 0; i < slots.size(); ++i) {
            size_t id = element_ids[i];
            DCHECK_LT(id, input.size());
            RETURN_IF_ERROR(input[id].CopyToSlot(slots[i], frame));
          }
          return absl::OkStatus();
        });
  }
};
}  
std::unique_ptr<InputLoader<Input>> CreateTypedRefsInputLoader(
    const std::vector<std::pair<std::string, QTypePtr>>& args) {
  return std::make_unique<TypedRefsInputLoader>(args);
}
}  