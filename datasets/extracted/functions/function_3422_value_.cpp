#ifndef TENSORSTORE_INDEX_SPACE_TRANSFORM_ARRAY_CONSTRAINTS_H_
#define TENSORSTORE_INDEX_SPACE_TRANSFORM_ARRAY_CONSTRAINTS_H_
#include "tensorstore/util/iterate.h"
namespace tensorstore {
enum class MustAllocateConstraint {
  may_allocate = 0,
  must_allocate = 1
};
constexpr MustAllocateConstraint may_allocate =
    MustAllocateConstraint::may_allocate;
constexpr MustAllocateConstraint must_allocate =
    MustAllocateConstraint::must_allocate;
class TransformArrayConstraints {
 public:
  constexpr TransformArrayConstraints(
      IterationConstraints iteration_constraint = {},
      MustAllocateConstraint allocate_constraint = may_allocate)
      : value_(iteration_constraint.value() |
               (static_cast<int>(allocate_constraint)
                << IterationConstraints::kNumBits)) {}
  constexpr TransformArrayConstraints(
      LayoutOrderConstraint order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      UnspecifiedLayoutOrder order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      ContiguousLayoutOrder order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      LayoutOrderConstraint order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      UnspecifiedLayoutOrder order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      ContiguousLayoutOrder order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      RepeatedElementsConstraint repeat_constraint,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(IterationConstraints(repeat_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints{}, allocate_constraint) {
  }
  explicit constexpr TransformArrayConstraints(int value) : value_(value) {}
  constexpr IterationConstraints iteration_constraints() const {
    return IterationConstraints(value() &
                                ((1 << IterationConstraints::kNumBits) - 1));
  }
  constexpr LayoutOrderConstraint order_constraint() const {
    return iteration_constraints().order_constraint();
  }
  constexpr RepeatedElementsConstraint repeated_elements_constraint() const {
    return iteration_constraints().repeated_elements_constraint();
  }
  constexpr MustAllocateConstraint allocate_constraint() const {
    return static_cast<MustAllocateConstraint>(value_ >>
                                               IterationConstraints::kNumBits);
  }
  constexpr int value() const { return value_; }
  constexpr static int kNumBits = IterationConstraints::kNumBits + 1;
  friend constexpr bool operator==(TransformArrayConstraints a,
                                   TransformArrayConstraints b) {
    return a.value() == b.value();
  }
  friend constexpr bool operator!=(TransformArrayConstraints a,
                                   TransformArrayConstraints b) {
    return a.value() != b.value();
  }
 private:
  int value_;
};
}  
#endif  