#include "checker/internal/test_ast_helpers.h"
#include <memory>
#include <utility>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "common/ast.h"
#include "extensions/protobuf/ast_converters.h"
#include "internal/status_macros.h"
#include "parser/parser.h"
namespace cel::checker_internal {
using ::cel::extensions::CreateAstFromParsedExpr;
using ::google::api::expr::parser::Parse;
absl::StatusOr<std::unique_ptr<Ast>> MakeTestParsedAst(
    absl::string_view expression) {
  CEL_ASSIGN_OR_RETURN(auto parsed, Parse(expression));
  return CreateAstFromParsedExpr(std::move(parsed));
}
}  