#include "eval/compiler/qualified_reference_resolver.h"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/ast.h"
#include "base/ast_internal/ast_impl.h"
#include "base/ast_internal/expr.h"
#include "base/builtins.h"
#include "base/kind.h"
#include "common/ast_rewrite.h"
#include "eval/compiler/flat_expr_builder_extensions.h"
#include "eval/compiler/resolver.h"
#include "runtime/internal/issue_collector.h"
#include "runtime/runtime_issue.h"
namespace google::api::expr::runtime {
namespace {
using ::cel::RuntimeIssue;
using ::cel::ast_internal::Expr;
using ::cel::ast_internal::Reference;
using ::cel::runtime_internal::IssueCollector;
constexpr absl::string_view kOptionalOr = "or";
constexpr absl::string_view kOptionalOrValue = "orValue";
bool IsSpecialFunction(absl::string_view function_name) {
  return function_name == cel::builtin::kAnd ||
         function_name == cel::builtin::kOr ||
         function_name == cel::builtin::kIndex ||
         function_name == cel::builtin::kTernary ||
         function_name == kOptionalOr || function_name == kOptionalOrValue ||
         function_name == "cel.@block";
}
bool OverloadExists(const Resolver& resolver, absl::string_view name,
                    const std::vector<cel::Kind>& arguments_matcher,
                    bool receiver_style = false) {
  return !resolver.FindOverloads(name, receiver_style, arguments_matcher)
              .empty() ||
         !resolver.FindLazyOverloads(name, receiver_style, arguments_matcher)
              .empty();
}
absl::optional<std::string> BestOverloadMatch(const Resolver& resolver,
                                              absl::string_view base_name,
                                              int argument_count) {
  if (IsSpecialFunction(base_name)) {
    return std::string(base_name);
  }
  auto arguments_matcher = ArgumentsMatcher(argument_count);
  auto names = resolver.FullyQualifiedNames(base_name);
  for (auto name = names.begin(); name != names.end(); ++name) {
    if (OverloadExists(resolver, *name, arguments_matcher)) {
      if (base_name[0] == '.') {
        return std::string(base_name);
      }
      return *name;
    }
  }
  return absl::nullopt;
}
class ReferenceResolver : public cel::AstRewriterBase {
 public:
  ReferenceResolver(
      const absl::flat_hash_map<int64_t, Reference>& reference_map,
      const Resolver& resolver, IssueCollector& issue_collector)
      : reference_map_(reference_map),
        resolver_(resolver),
        issues_(issue_collector),
        progress_status_(absl::OkStatus()) {}
  bool PreVisitRewrite(Expr& expr) override {
    const Reference* reference = GetReferenceForId(expr.id());
    if (reference != nullptr && reference->has_value()) {
      if (reference->value().has_int64_value()) {
        expr.mutable_const_expr().set_int64_value(
            reference->value().int64_value());
        return true;
      } else {
        return false;
      }
    }
    if (reference != nullptr) {
      if (expr.has_ident_expr()) {
        return MaybeUpdateIdentNode(&expr, *reference);
      } else if (expr.has_select_expr()) {
        return MaybeUpdateSelectNode(&expr, *reference);
      } else {
        return false;
      }
    }
    return false;
  }
  bool PostVisitRewrite(Expr& expr) override {
    const Reference* reference = GetReferenceForId(expr.id());
    if (expr.has_call_expr()) {
      return MaybeUpdateCallNode(&expr, reference);
    }
    return false;
  }
  const absl::Status& GetProgressStatus() const { return progress_status_; }
 private:
  bool MaybeUpdateCallNode(Expr* out, const Reference* reference) {
    auto& call_expr = out->mutable_call_expr();
    const std::string& function = call_expr.function();
    if (reference != nullptr && reference->overload_id().empty()) {
      UpdateStatus(issues_.AddIssue(
          RuntimeIssue::CreateWarning(absl::InvalidArgumentError(
              absl::StrCat("Reference map doesn't provide overloads for ",
                           out->call_expr().function())))));
    }
    bool receiver_style = call_expr.has_target();
    int arg_num = call_expr.args().size();
    if (receiver_style) {
      auto maybe_namespace = ToNamespace(call_expr.target());
      if (maybe_namespace.has_value()) {
        std::string resolved_name =
            absl::StrCat(*maybe_namespace, ".", function);
        auto resolved_function =
            BestOverloadMatch(resolver_, resolved_name, arg_num);
        if (resolved_function.has_value()) {
          call_expr.set_function(*resolved_function);
          call_expr.set_target(nullptr);
          return true;
        }
      }
    } else {
      auto maybe_resolved_function =
          BestOverloadMatch(resolver_, function, arg_num);
      if (!maybe_resolved_function.has_value()) {
        UpdateStatus(issues_.AddIssue(RuntimeIssue::CreateWarning(
            absl::InvalidArgumentError(absl::StrCat(
                "No overload found in reference resolve step for ", function)),
            RuntimeIssue::ErrorCode::kNoMatchingOverload)));
      } else if (maybe_resolved_function.value() != function) {
        call_expr.set_function(maybe_resolved_function.value());
        return true;
      }
    }
    if (call_expr.has_target() && !IsSpecialFunction(function) &&
        !OverloadExists(resolver_, function, ArgumentsMatcher(arg_num + 1),
                         true)) {
      UpdateStatus(issues_.AddIssue(RuntimeIssue::CreateWarning(
          absl::InvalidArgumentError(absl::StrCat(
              "No overload found in reference resolve step for ", function)),
          RuntimeIssue::ErrorCode::kNoMatchingOverload)));
    }
    return false;
  }
  bool MaybeUpdateSelectNode(Expr* out, const Reference& reference) {
    if (out->select_expr().test_only()) {
      UpdateStatus(issues_.AddIssue(RuntimeIssue::CreateWarning(
          absl::InvalidArgumentError("Reference map points to a presence "
                                     "test -- has(container.attr)"))));
    } else if (!reference.name().empty()) {
      out->mutable_ident_expr().set_name(reference.name());
      rewritten_reference_.insert(out->id());
      return true;
    }
    return false;
  }
  bool MaybeUpdateIdentNode(Expr* out, const Reference& reference) {
    if (!reference.name().empty() &&
        reference.name() != out->ident_expr().name()) {
      out->mutable_ident_expr().set_name(reference.name());
      rewritten_reference_.insert(out->id());
      return true;
    }
    return false;
  }
  absl::optional<std::string> ToNamespace(const Expr& expr) {
    absl::optional<std::string> maybe_parent_namespace;
    if (rewritten_reference_.find(expr.id()) != rewritten_reference_.end()) {
      return absl::nullopt;
    }
    if (expr.has_ident_expr()) {
      return expr.ident_expr().name();
    } else if (expr.has_select_expr()) {
      if (expr.select_expr().test_only()) {
        return absl::nullopt;
      }
      maybe_parent_namespace = ToNamespace(expr.select_expr().operand());
      if (!maybe_parent_namespace.has_value()) {
        return absl::nullopt;
      }
      return absl::StrCat(*maybe_parent_namespace, ".",
                          expr.select_expr().field());
    } else {
      return absl::nullopt;
    }
  }
  const Reference* GetReferenceForId(int64_t expr_id) {
    auto iter = reference_map_.find(expr_id);
    if (iter == reference_map_.end()) {
      return nullptr;
    }
    if (expr_id == 0) {
      UpdateStatus(issues_.AddIssue(
          RuntimeIssue::CreateWarning(absl::InvalidArgumentError(
              "reference map entries for expression id 0 are not supported"))));
      return nullptr;
    }
    return &iter->second;
  }
  void UpdateStatus(absl::Status status) {
    if (progress_status_.ok() && !status.ok()) {
      progress_status_ = std::move(status);
      return;
    }
    status.IgnoreError();
  }
  const absl::flat_hash_map<int64_t, Reference>& reference_map_;
  const Resolver& resolver_;
  IssueCollector& issues_;
  absl::Status progress_status_;
  absl::flat_hash_set<int64_t> rewritten_reference_;
};
class ReferenceResolverExtension : public AstTransform {
 public:
  explicit ReferenceResolverExtension(ReferenceResolverOption opt)
      : opt_(opt) {}
  absl::Status UpdateAst(PlannerContext& context,
                         cel::ast_internal::AstImpl& ast) const override {
    if (opt_ == ReferenceResolverOption::kCheckedOnly &&
        ast.reference_map().empty()) {
      return absl::OkStatus();
    }
    return ResolveReferences(context.resolver(), context.issue_collector(), ast)
        .status();
  }
 private:
  ReferenceResolverOption opt_;
};
}  
absl::StatusOr<bool> ResolveReferences(const Resolver& resolver,
                                       IssueCollector& issues,
                                       cel::ast_internal::AstImpl& ast) {
  ReferenceResolver ref_resolver(ast.reference_map(), resolver, issues);
  bool was_rewritten = cel::AstRewrite(ast.root_expr(), ref_resolver);
  if (!ref_resolver.GetProgressStatus().ok()) {
    return ref_resolver.GetProgressStatus();
  }
  return was_rewritten;
}
std::unique_ptr<AstTransform> NewReferenceResolverExtension(
    ReferenceResolverOption option) {
  return std::make_unique<ReferenceResolverExtension>(option);
}
}  