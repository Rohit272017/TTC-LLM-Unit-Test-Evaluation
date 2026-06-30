#ifndef AROLLA_EXPR_OPERATORS_REGISTRATION_H_
#define AROLLA_EXPR_OPERATORS_REGISTRATION_H_
#include <type_traits>  
#include "absl/base/no_destructor.h"      
#include "absl/status/status.h"           
#include "absl/status/statusor.h"         
#include "arolla/expr/expr.h"            
#include "arolla/expr/expr_operator.h"   
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators {
#define AROLLA_DECLARE_EXPR_OPERATOR(op_function_name)                        \
  absl::StatusOr<::arolla::expr::ExprOperatorPtr> Get##op_function_name();    \
  inline absl::Status Register##op_function_name() {                          \
    return Get##op_function_name().status();                                  \
  }                                                                           \
  template <typename... Args>                                                 \
  std::enable_if_t<(std::is_convertible_v<                                    \
                        Args, absl::StatusOr<::arolla::expr::ExprNodePtr>> && \
                    ...),                                                     \
                   absl::StatusOr<::arolla::expr::ExprNodePtr>>               \
  op_function_name(Args... args) {                                            \
    ASSIGN_OR_RETURN(auto op, Get##op_function_name());                       \
    return ::arolla::expr::CallOp(op, {args...});                             \
  }
#define AROLLA_DEFINE_EXPR_OPERATOR(op_function_name, registration_call)    \
  absl::StatusOr<::arolla::expr::ExprOperatorPtr> Get##op_function_name() { \
    static const absl::NoDestructor<                                        \
        absl::StatusOr<::arolla::expr::ExprOperatorPtr>>                    \
        registered(registration_call);                                      \
    return *registered;                                                     \
  }
}  
#endif  