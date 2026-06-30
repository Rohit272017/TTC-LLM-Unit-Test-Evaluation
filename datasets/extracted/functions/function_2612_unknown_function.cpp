#ifndef XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_H_
#define XLA_STREAM_EXECUTOR_CUDA_PTX_COMPILER_H_
#include <cstdint>
#include <vector>
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/semantic_version.h"
namespace stream_executor {
absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmUsingLibNvPtxCompiler(
    int cc_major, int cc_minor, const char* ptx_contents, GpuAsmOpts options,
    bool cancel_if_reg_spill);
absl::StatusOr<SemanticVersion> GetLibNvPtxCompilerVersion();
}  
#endif  