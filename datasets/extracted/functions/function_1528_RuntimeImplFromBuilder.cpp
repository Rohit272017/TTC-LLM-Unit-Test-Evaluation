#include "runtime/regex_precompilation.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "common/native_type.h"
#include "eval/compiler/regex_precompilation_optimization.h"
#include "internal/casts.h"
#include "internal/status_macros.h"
#include "runtime/internal/runtime_friend_access.h"
#include "runtime/internal/runtime_impl.h"
#include "runtime/runtime.h"
#include "runtime/runtime_builder.h"
namespace cel::extensions {
namespace {
using ::cel::internal::down_cast;
using ::cel::runtime_internal::RuntimeFriendAccess;
using ::cel::runtime_internal::RuntimeImpl;
using ::google::api::expr::runtime::CreateRegexPrecompilationExtension;
absl::StatusOr<RuntimeImpl*> RuntimeImplFromBuilder(RuntimeBuilder& builder) {
  Runtime& runtime = RuntimeFriendAccess::GetMutableRuntime(builder);
  if (RuntimeFriendAccess::RuntimeTypeId(runtime) !=
      NativeTypeId::For<RuntimeImpl>()) {
    return absl::UnimplementedError(
        "regex precompilation only supported on the default cel::Runtime "
        "implementation.");
  }
  RuntimeImpl& runtime_impl = down_cast<RuntimeImpl&>(runtime);
  return &runtime_impl;
}
}  
absl::Status EnableRegexPrecompilation(RuntimeBuilder& builder) {
  CEL_ASSIGN_OR_RETURN(RuntimeImpl * runtime_impl,
                       RuntimeImplFromBuilder(builder));
  ABSL_ASSERT(runtime_impl != nullptr);
  runtime_impl->expr_builder().AddProgramOptimizer(
      CreateRegexPrecompilationExtension(
          runtime_impl->expr_builder().options().regex_max_program_size));
  return absl::OkStatus();
}
}  