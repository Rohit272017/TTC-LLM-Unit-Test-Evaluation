#include "tensorflow/lite/delegates/gpu/gl/kernels/lstm.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace {
class LstmNodeShader : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code = R"(
      vec4 prev_state  = $input_data_1[gid.x, gid.y, gid.z]$;
      int c0 = 0 * $workload_z$;
      int c1 = 1 * $workload_z$;
      int c2 = 2 * $workload_z$;
      int c3 = 3 * $workload_z$;
      vec4 gate_0 = $input_data_0[gid.x, gid.y, gid.z + c0]$;
      vec4 gate_1 = $input_data_0[gid.x, gid.y, gid.z + c1]$;
      vec4 gate_2 = $input_data_0[gid.x, gid.y, gid.z + c2]$;
      vec4 gate_3 = $input_data_0[gid.x, gid.y, gid.z + c3]$;
      vec4 input_gate  = 1.0f / (1.0f + exp(-1.0 * gate_0));  
      vec4 new_input   = tanh(gate_1);                        
      vec4 forget_gate = 1.0f / (1.0f + exp(-1.0 * gate_2));  
      vec4 output_gate = 1.0f / (1.0f + exp(-1.0 * gate_3));  
      vec4 new_state = input_gate * new_input + forget_gate * prev_state;
      vec4 activation = output_gate * tanh(new_state);
      value_0 = new_state;
      value_1 = activation;
    )";
    *generated_code = {
        {},
        {},
        {},
        uint3(),
        uint3(),
        std::move(code),
        IOStructure::ONLY_DEFINITIONS,
        IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
}  
std::unique_ptr<NodeShader> NewLstmNodeShader() {
  return std::make_unique<LstmNodeShader>();
}
}  
}  
}  