#ifndef XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_H_
#define XLA_STREAM_EXECUTOR_CUDA_NVJITLINK_H_
#include <cstdint>
#include <tuple>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
namespace stream_executor {
using NvJitLinkVersion = std::tuple<unsigned, unsigned>;
absl::StatusOr<NvJitLinkVersion> GetNvJitLinkVersion();
struct NvJitLinkInput {
  enum class Type { kPtx, kCubin };
  Type type;
  absl::Span<const uint8_t> bytes;
};
absl::StatusOr<std::vector<uint8_t>> CompileAndLinkUsingLibNvJitLink(
    int cc_major, int cc_minor, absl::Span<const NvJitLinkInput> inputs,
    GpuAsmOpts options, bool cancel_if_reg_spill);
}  
#endif  