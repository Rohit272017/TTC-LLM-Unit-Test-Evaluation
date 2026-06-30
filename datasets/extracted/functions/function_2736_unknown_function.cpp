#include "arolla/serving/expr_compiler.h"
#include <optional>
#include "absl/base/no_destructor.h"
#include "arolla/expr/optimization/optimizer.h"
namespace arolla::serving_impl {
absl::NoDestructor<std::optional<expr::Optimizer>>
    ExprCompilerDefaultOptimizer::optimizer_;
}  