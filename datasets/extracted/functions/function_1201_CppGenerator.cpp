#include "tensorflow/c/experimental/ops/gen/cpp/cpp_generator.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_file_renderer.h"
#include "tensorflow/core/lib/io/path.h"
namespace tensorflow {
namespace generator {
CppGenerator::CppGenerator(cpp::CppConfig cpp_config, PathConfig path_config)
    : controller_(path_config),
      cpp_config_(cpp_config),
      path_config_(path_config) {}
SourceCode CppGenerator::GenerateOneFile(
    cpp::RendererContext::Mode mode) const {
  SourceCode generated_code;
  const std::vector<OpSpec> ops(controller_.GetModelOps());
  std::vector<cpp::OpView> views(ops.begin(), ops.end());
  cpp::RendererContext context{mode, generated_code, cpp_config_, path_config_};
  cpp::CppFileRenderer(context, views).Render();
  return generated_code;
}
SourceCode CppGenerator::HeaderFileContents() const {
  return GenerateOneFile(cpp::RendererContext::kHeader);
}
SourceCode CppGenerator::SourceFileContents() const {
  return GenerateOneFile(cpp::RendererContext::kSource);
}
string CppGenerator::HeaderFileName() const {
  return io::JoinPath(path_config_.output_path, cpp_config_.unit + "_ops.h");
}
string CppGenerator::SourceFileName() const {
  return io::JoinPath(path_config_.output_path, cpp_config_.unit + "_ops.cc");
}
void CppGenerator::WriteHeaderFile() const {
  controller_.WriteFile(HeaderFileName(), HeaderFileContents());
}
void CppGenerator::WriteSourceFile() const {
  controller_.WriteFile(SourceFileName(), SourceFileContents());
}
}  
}  