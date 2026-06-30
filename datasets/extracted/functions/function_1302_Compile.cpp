#include "xla/service/llvm_compiler.h"
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/executable.h"
#include "xla/service/stream_pool.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#ifdef __FAST_MATH__
#error "Don't build XLA with -ffast-math"
#endif
namespace xla {
absl::StatusOr<std::vector<std::unique_ptr<Executable>>> LLVMCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    const CompileOptions& options) {
  tsl::port::ScopedDontFlushDenormal dont_flush_denormals;
  std::vector<std::unique_ptr<Executable>> result;
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  for (size_t i = 0; i < modules.size(); i++) {
    tsl::profiler::ScopedAnnotation annotation{[&] {
      return absl::StrFormat("XlaCompile:#module=%s,program_id=%d#",
                             modules[i]->name(), modules[i]->unique_id());
    }};
    TF_ASSIGN_OR_RETURN(modules[i], RunHloPasses(std::move(modules[i]),
                                                 stream_execs[i][0], options));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        RunBackend(std::move(modules[i]), stream_execs[i][0], options));
    result.push_back(std::move(executable));
  }
  return std::move(result);
}
}  