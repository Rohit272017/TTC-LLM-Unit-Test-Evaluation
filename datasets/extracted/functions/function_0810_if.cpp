#include "tensorflow/lite/delegates/gpu/gl/kernels/max_unpooling.h"
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
class MaxUnpooling : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr =
        std::any_cast<const MaxUnpooling2DAttributes&>(ctx.op_attr);
    std::vector<Variable> parameters = {
        {"stride", int2(attr.strides.w, attr.strides.h)},
        {"offset", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
        {"window_h", attr.kernel.h},
        {"window_w", attr.kernel.w},
    };
    std::string source = R"(
      ivec2 coord = (gid.xy + $offset$) / $stride$;
      ivec4 indices = $input_data_1[coord.x, coord.y, gid.z]$;
      vec4 input_ = $input_data_0[coord.x, coord.y, gid.z]$;
      coord = coord * $stride$ - $offset$;
      for (int i = 0; i < 4; ++i) {
        ivec2 t = coord + ivec2(indices[i] % $window_w$, indices[i] / $window_w$);
        if (t.x == gid.x && t.y == gid.y) {
          value_0[i] = input_[i];
        }
      }
    )";
    *generated_code = {
        std::move(parameters),
        {},
        {},
        uint3(),
        uint3(),
        std::move(source),
        IOStructure::ONLY_DEFINITIONS,
        IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
}  
std::unique_ptr<NodeShader> NewMaxUnpoolingNodeShader() {
  return std::make_unique<MaxUnpooling>();
}
}  
}  
}  