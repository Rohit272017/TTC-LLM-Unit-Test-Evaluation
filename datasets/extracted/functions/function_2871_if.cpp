#include "tensorflow/lite/delegates/gpu/gl/kernels/relu.h"
#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace {
class ReLU : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const ReLUAttributes&>(ctx.op_attr);
    std::vector<Variable> params;
    std::string min;
    if (attr.alpha == 0) {
      min = "vec4($activation_min$)";
      params.push_back({"activation_min", attr.activation_min});
    } else {
      min = "min($alpha$ * value_0, 0.0)";
      params.push_back({"alpha", attr.alpha});
    }
    std::string code;
    if (attr.activation_max == 0) {
      code = "value_0 = max(value_0, " + min + ");";
    } else {
      code = "value_0 = clamp(value_0, " + min + ", vec4($activation_max$));";
      params.push_back({"activation_max", attr.activation_max});
    }
    *generated_code = {
        std::move(params),
        {},
        {},
        uint3(),
        uint3(),
        std::move(code),
        IOStructure::AUTO,
        IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
}  
std::unique_ptr<NodeShader> NewReLUNodeShader() {
  return std::make_unique<ReLU>();
}
}  
}  
}  