#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/util/determinism.h"
namespace tensorflow {
xla::ExecutableBuildOptions GetExecutableBuildOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result, int default_device_ordinal) {
  xla::ExecutableBuildOptions build_options;
  if (result.collective_info) {
    build_options.set_num_replicas(result.collective_info->group_size);
  }
  if (options.device_ordinal != -1) {
    build_options.set_device_ordinal(options.device_ordinal);
  } else if (default_device_ordinal != -1) {
    build_options.set_device_ordinal(default_device_ordinal);
  }
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator.get());
  build_options.set_alias_passthrough_params(options.alias_passthrough_params);
  build_options.mutable_debug_options()->set_xla_detailed_logging(
      options.detailed_logging);
  if (tensorflow::OpDeterminismRequired()) {
    build_options.mutable_debug_options()->set_xla_gpu_deterministic_ops(true);
  }
  return build_options;
}
}  