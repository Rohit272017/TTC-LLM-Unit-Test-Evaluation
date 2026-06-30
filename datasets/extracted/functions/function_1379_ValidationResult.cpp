#ifndef THIRD_PARTY_CEL_CPP_CHECKER_VALIDATION_RESULT_H_
#define THIRD_PARTY_CEL_CPP_CHECKER_VALIDATION_RESULT_H_
#include <memory>
#include <utility>
#include <vector>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "checker/type_check_issue.h"
#include "common/ast.h"
namespace cel {
class ValidationResult {
 public:
  ValidationResult(std::unique_ptr<Ast> ast, std::vector<TypeCheckIssue> issues)
      : ast_(std::move(ast)), issues_(std::move(issues)) {}
  explicit ValidationResult(std::vector<TypeCheckIssue> issues)
      : ast_(nullptr), issues_(std::move(issues)) {}
  bool IsValid() const { return ast_ != nullptr; }
  absl::Nullable<const Ast*> GetAst() const { return ast_.get(); }
  absl::StatusOr<std::unique_ptr<Ast>> ReleaseAst() {
    if (ast_ == nullptr) {
      return absl::FailedPreconditionError(
          "ValidationResult is empty. Check for TypeCheckIssues.");
    }
    return std::move(ast_);
  }
  absl::Span<const TypeCheckIssue> GetIssues() const { return issues_; }
 private:
  absl::Nullable<std::unique_ptr<Ast>> ast_;
  std::vector<TypeCheckIssue> issues_;
};
}  
#endif  