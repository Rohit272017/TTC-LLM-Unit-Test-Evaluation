#ifndef XLA_SERVICE_FUZZY_MATCHER_H_
#define XLA_SERVICE_FUZZY_MATCHER_H_
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
namespace xla {
namespace fm {
template <typename Pattern>
auto OptConvert(Pattern pattern) {
  auto shared = match::SharedSubpattern(pattern);
  return match::AnyOf<HloInstruction>(match::Convert(shared), shared);
}
#define XLA_FUZZY_UNOP_PATTERN(NAME)                                           \
  template <typename HloInstructionType>                                       \
  inline auto NAME(HloInstructionType** matched_inst) {                        \
    return OptConvert(match::Op(matched_inst).WithOpcode(HloOpcode::k##NAME)); \
  }                                                                            \
                                                                               \
  template <typename Arg>                                                      \
  inline auto NAME(Arg&& arg) {                                                \
    return OptConvert(match::Op()                                              \
                          .WithOpcode(HloOpcode::k##NAME)                      \
                          .WithOperand(0, std::forward<Arg>(arg)));            \
  }                                                                            \
                                                                               \
  template <typename HloInstructionType, typename Arg>                         \
  inline auto NAME(HloInstructionType** matched_inst, Arg&& arg) {             \
    return OptConvert(match::Op(matched_inst)                                  \
                          .WithOpcode(HloOpcode::k##NAME)                      \
                          .WithOperand(0, std::forward<Arg>(arg)));            \
  }
XLA_FUZZY_UNOP_PATTERN(Tanh)
XLA_FUZZY_UNOP_PATTERN(Exp)
XLA_FUZZY_UNOP_PATTERN(Broadcast)
#undef XLA_FUZZY_UNOP_PATTERN
#define XLA_FUZZY_BINOP_PATTERN(NAME)                                         \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return OptConvert(match::Op(matched_inst)                                 \
                          .WithOpcode(HloOpcode::k##NAME)                     \
                          .WithOperand(0, std::forward<Lhs>(lhs))             \
                          .WithOperand(1, std::forward<Rhs>(rhs)));           \
  }                                                                           \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return OptConvert(match::Op()                                             \
                          .WithOpcode(HloOpcode::k##NAME)                     \
                          .WithOperand(0, std::forward<Lhs>(lhs))             \
                          .WithOperand(1, std::forward<Rhs>(rhs)));           \
  }
XLA_FUZZY_BINOP_PATTERN(Dot)
XLA_FUZZY_BINOP_PATTERN(Divide)
XLA_FUZZY_BINOP_PATTERN(Subtract)
XLA_FUZZY_BINOP_PATTERN(Multiply)
XLA_FUZZY_BINOP_PATTERN(Reduce)
#undef XLA_FUZZY_BINOP_PATTERN
#define XLA_FUZZY_TERNOP_PATTERN(NAME)                                 \
  template <typename Arg0, typename Arg1, typename Arg2>               \
  inline auto NAME(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {            \
    return OptConvert(match::Op()                                      \
                          .WithOpcode(HloOpcode::k##NAME)              \
                          .WithOperand(0, std::forward<Arg0>(arg0))    \
                          .WithOperand(1, std::forward<Arg1>(arg1))    \
                          .WithOperand(2, std::forward<Arg2>(arg2)));  \
  }                                                                    \
                                                                       \
  template <typename HloInstructionType, typename Arg0, typename Arg1, \
            typename Arg2>                                             \
  inline auto NAME(HloInstructionType** matched_inst, Arg0&& arg0,     \
                   Arg1&& arg1, Arg2&& arg2) {                         \
    return OptConvert(match::Op(matched_inst)                          \
                          .WithOpcode(HloOpcode::k##NAME)              \
                          .WithOperand(0, std::forward<Arg0>(arg0))    \
                          .WithOperand(1, std::forward<Arg1>(arg1))    \
                          .WithOperand(2, std::forward<Arg2>(arg2)));  \
  }
XLA_FUZZY_TERNOP_PATTERN(Select);
#undef XLA_FUZZY_TERNOP_PATTERN
}  
}  
#endif  