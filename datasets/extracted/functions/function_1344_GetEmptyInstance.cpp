#ifndef THIRD_PARTY_CEL_CPP_EVAL_EVAL_COMPREHENSION_SLOTS_H_
#define THIRD_PARTY_CEL_CPP_EVAL_EVAL_COMPREHENSION_SLOTS_H_
#include <cstddef>
#include <utility>
#include <vector>
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/types/optional.h"
#include "common/value.h"
#include "eval/eval/attribute_trail.h"
namespace google::api::expr::runtime {
class ComprehensionSlots {
 public:
  struct Slot {
    cel::Value value;
    AttributeTrail attribute;
  };
  static ComprehensionSlots& GetEmptyInstance() {
    static absl::NoDestructor<ComprehensionSlots> instance(0);
    return *instance;
  }
  explicit ComprehensionSlots(size_t size) : size_(size), slots_(size) {}
  ComprehensionSlots(const ComprehensionSlots&) = delete;
  ComprehensionSlots& operator=(const ComprehensionSlots&) = delete;
  ComprehensionSlots(ComprehensionSlots&&) = default;
  ComprehensionSlots& operator=(ComprehensionSlots&&) = default;
  Slot* Get(size_t index) {
    ABSL_DCHECK_LT(index, slots_.size());
    auto& slot = slots_[index];
    if (!slot.has_value()) return nullptr;
    return &slot.value();
  }
  void Reset() {
    slots_.clear();
    slots_.resize(size_);
  }
  void ClearSlot(size_t index) {
    ABSL_DCHECK_LT(index, slots_.size());
    slots_[index] = absl::nullopt;
  }
  void Set(size_t index) {
    ABSL_DCHECK_LT(index, slots_.size());
    slots_[index].emplace();
  }
  void Set(size_t index, cel::Value value) {
    Set(index, std::move(value), AttributeTrail());
  }
  void Set(size_t index, cel::Value value, AttributeTrail attribute) {
    ABSL_DCHECK_LT(index, slots_.size());
    slots_[index] = Slot{std::move(value), std::move(attribute)};
  }
  size_t size() const { return slots_.size(); }
 private:
  size_t size_;
  std::vector<absl::optional<Slot>> slots_;
};
}  
#endif  