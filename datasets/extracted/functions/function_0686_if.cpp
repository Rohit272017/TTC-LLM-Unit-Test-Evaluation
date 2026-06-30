#include "tensorflow/lite/delegates/gpu/gl/kernels/prelu.h"
#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace {
class PReLULinearAlpha : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const PReLUAttributes&>(ctx.op_attr);
    auto alpha = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
    if (!alpha) {
      return absl::InvalidArgumentError("Alpha is missing");
    }
    if (alpha->shape.v != ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError(
          "Alpha shape does not match the number of channels.");
    }
    *generated_code = GeneratedCode{
        {},
        {{"alpha", MakeReadonlyObject(alpha->data)}},
        {},
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.output_shapes[0][3]), 4)),
        uint3(),
        "value_0 = max(value_0, 0.0) + $alpha[gid.z]$ * min(value_0, "
        "0.0);",
        IOStructure::AUTO,
        IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
class PReLUFull : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const PReLUAttributes&>(ctx.op_attr);
    auto alpha = std::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
    if (!alpha) {
      return absl::InvalidArgumentError("Alpha is missing");
    }
    if (alpha->shape.h != ctx.output_shapes[0][1] ||
        alpha->shape.w != ctx.output_shapes[0][2] ||
        alpha->shape.c != ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError(
          "Alpha shape does not match input shape.");
    }
    ObjectSize obj_size =
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.output_shapes[0][3]), 4));
    *generated_code = GeneratedCode{
        {},
        {{"alpha", MakeReadonlyObject(obj_size, ConvertToPHWC4(*alpha))}},
        {},
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.output_shapes[0][3]), 4)),
        uint3(),
        "value_0 = max(value_0, 0.0) + $alpha[gid.x, gid.y, gid.z]$ "
        "* min(value_0, 0.0);",
        IOStructure::AUTO,
        IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
class PReLU : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const PReLUAttributes&>(ctx.op_attr);
    auto* alpha = std::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
    return alpha ? full_.GenerateCode(ctx, generated_code)
                 : linear_.GenerateCode(ctx, generated_code);
  }
 private:
  PReLULinearAlpha linear_;
  PReLUFull full_;
};
}  
std::unique_ptr<NodeShader> NewPReLUNodeShader() {
  return std::make_unique<PReLU>();
}
}  
}  
}  