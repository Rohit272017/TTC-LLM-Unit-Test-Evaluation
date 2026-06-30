#ifndef XLA_HLO_IR_HLO_CASTING_UTILS_H_
#define XLA_HLO_IR_HLO_CASTING_UTILS_H_
#include <type_traits>
#include "xla/hlo/ir/hlo_instruction.h"
#include "tsl/platform/logging.h"
namespace xla {
template <class T>
using EnableIfDerivedFromHlo =
    typename std::enable_if<std::is_base_of<HloInstruction, T>::value>::type;
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* Cast(const HloInstruction* instruction) {
  CHECK(instruction != nullptr);
  CHECK(T::ClassOf(instruction))
      << "Invalid HloInstruction casting. Destination type: "
      << typeid(T).name() << ". Instruction: " << instruction->name();
  const T* casted = static_cast<const T*>(instruction);
#ifndef NDEBUG
  const T* dynamic_casted = dynamic_cast<const T*>(instruction);
  CHECK(dynamic_casted != nullptr)
      << "Invalid HloInstruction casting. Destination type: "
      << typeid(T).name() << ". Instruction: " << instruction->name();
#endif
  return casted;
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* Cast(HloInstruction* instruction) {
  return const_cast<T*>(
      Cast<T>(const_cast<const HloInstruction*>(instruction)));
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* CastOrNull(const HloInstruction* instruction) {
  return instruction != nullptr ? Cast<T>(instruction) : nullptr;
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* CastOrNull(HloInstruction* instruction) {
  return const_cast<T*>(
      CastOrNull<T>(const_cast<const HloInstruction*>(instruction)));
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* DynCast(const HloInstruction* instruction) {
  CHECK(instruction != nullptr);
  const T* casted =
      T::ClassOf(instruction) ? static_cast<const T*>(instruction) : nullptr;
#ifndef NDEBUG
  CHECK_EQ(casted, dynamic_cast<const T*>(instruction));
#endif
  return casted;
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* DynCast(HloInstruction* instruction) {
  return const_cast<T*>(
      DynCast<T>(const_cast<const HloInstruction*>(instruction)));
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
const T* DynCastOrNull(const HloInstruction* instruction) {
  return instruction != nullptr ? DynCast<T>(instruction) : nullptr;
}
template <class T, EnableIfDerivedFromHlo<T>* = nullptr>
T* DynCastOrNull(HloInstruction* instruction) {
  return const_cast<T*>(
      DynCastOrNull<T>(const_cast<const HloInstruction*>(instruction)));
}
}  
#endif  