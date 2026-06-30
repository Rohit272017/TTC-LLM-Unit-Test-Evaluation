#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"
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
class AlignedConcatByChannels : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);
    if (attr.axis != Axis::CHANNELS) return false;
    if (ctx.input_shapes.size() != 2) return false;
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }
    for (const auto& shape : ctx.input_shapes) {
      if (shape[3] % 4 != 0) return false;
    }
    return true;
  }
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::InvalidArgumentError(
          "This case is not supported by aligned concat");
    }
    std::string source = R"(
      if (gid.z < $border$) {
        value_0 = $input_data_0[gid.x, gid.y, gid.z]$;
      } else {
        int z = gid.z - $border$;
        value_0 = $input_data_1[gid.x, gid.y, z]$;
      }
)";
    *generated_code = {
        {
            {"border", static_cast<int>(ctx.input_shapes[0][3]) / 4}},
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
class ConcatByAnyChannel : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);
    if (attr.axis != Axis::CHANNELS) return false;
    if (ctx.input_shapes.size() <= 1) return false;
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }
    return true;
  }
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::UnimplementedError("This case is not supported by concat");
    }
    std::string code = DeclareVariables();
    int already_written = 0;
    int t = 0;
    for (int current_input_id = 0; current_input_id < ctx.input_shapes.size();
         current_input_id++) {
      int in_ch = ctx.input_shapes[current_input_id][3];
      code += PrintStartMessage(current_input_id, in_ch, already_written);
      std::string input = "input_data_" + std::to_string(current_input_id);
      int reminder = already_written % 4;
      if (reminder == 0) {
        code += AlignedCase(in_ch, input);
      } else {
        code += UnalignedCase(reminder, in_ch, input, &t);
      }
      already_written += in_ch;
    }
    *generated_code = {
        {},
        {},
        {},
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]), 1),
        uint3(),
        std::move(code),
        IOStructure::ONLY_DEFINITIONS,
        IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }
 private:
  std::string temp(int t) const { return "temp" + std::to_string(t); }
  std::string DeclareVariables() const {
    return R"(
int z = gid.z;
vec4 val = vec4(0.0f);
)";
  }
  std::string PrintStartMessage(int current_input_id, int in_ch,
                                int already_written) const {
    return "
           " tensor with " + std::to_string(in_ch) +
           " channels\n
           std::to_string(already_written) + " elements\n\n";
  }
  std::string AlignedCase(int in_ch, const std::string& input) const {
    std::string code;
    int blocks_amount = DivideRoundUp<int>(in_ch, 4);
    code += "
    code += "
            " write(s)\n\n";
    for (int block = 0; block < blocks_amount; block++) {
      code += "val = $" + input + "[gid.x, gid.y, " + std::to_string(block) +
              "]$;\n" +
              "$output_data_0[gid.x, gid.y, z] = val$;\n"
              + "z++; \n\n";
    }
    return code;
  }
  std::string UnalignedCase(int reminder, int in_ch, const std::string& input,
                            int* t) const {
    std::string code = "
    int shift = 4 - reminder;
    if (shift > in_ch) {
      shift = in_ch;
    }
    code += "\n
    code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, 0]$;\n";
    for (int i = 0; i < shift; i++) {
      code += "val[" + std::to_string(reminder + i) + "] = " + temp(*t) + "[" +
              std::to_string(i) + "];\n";
    }
    code += "$output_data_0[gid.x, gid.y, z - 1] = val$;\n";
    (*t)++;
    int left_blocks = (in_ch - shift) / 4;
    if ((in_ch - shift) % 4 != 0) {
      left_blocks++;
    }
    if (left_blocks) {
      code += "\n
      for (int block = 0; block < left_blocks; block++) {
        for (int elem = 0; elem < 4; elem++) {
          if (shift % 4 == 0) {
            code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, " +
                    std::to_string(block + 1) + "]$;\n";
            (*t)++;
          }
          code += "val[" + std::to_string(elem) + "] = " + temp(*t - 1) + "[" +
                  std::to_string(shift % 4) + "];\n";
          if (shift == in_ch) {
            break;
          }
          shift++;
        }
        code += "$output_data_0[gid.x, gid.y, z] = val$;\n";
        code += "z++;\n";
      }
    } else {
      code += "
    }
    return code;
  }
};
class FlatConcatByHeight : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);
    if (attr.axis != Axis::HEIGHT) return false;
    if (ctx.input_shapes.size() <= 1) return false;
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }
    return true;
  }
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][1], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_h$ <= gid.y && ";
      }
      code +=
          "gid.y < " + std::to_string(shift + ctx.input_shapes[i][1]) + ") {\n";
      code += "if (gid.y - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_h$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) +
              "[gid.x, gid.y - " + std::to_string(shift) + ", gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_h",
                        static_cast<int>(ctx.input_shapes[i][1])});
    }
    *generated_code = {
        std::move(params),
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
class FlatConcatByWidth : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);
    if (attr.axis != Axis::WIDTH) return false;
    if (ctx.input_shapes.size() <= 1) return false;
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][1] != ctx.input_shapes[i][1]) {
        return false;
      }
    }
    return true;
  }
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][2], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_w$ <= gid.x && ";
      }
      code +=
          "gid.x < " + std::to_string(shift + ctx.input_shapes[i][2]) + ") {\n";
      code += "if (gid.x - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_w$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) + "[gid.x - " +
              std::to_string(shift) + ", gid.y, gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_w",
                        static_cast<int>(ctx.input_shapes[i][2])});
    }
    *generated_code = {
        std::move(params),
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
class FlatConcat : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (FlatConcatByHeight::IsSupported(ctx)) {
      return flat_concat_by_height_.GenerateCode(ctx, generated_code);
    }
    if (FlatConcatByWidth::IsSupported(ctx)) {
      return flat_concat_by_width_.GenerateCode(ctx, generated_code);
    }
    return absl::InvalidArgumentError(
        "This case is not supported by flat concat");
  }
 private:
  FlatConcatByHeight flat_concat_by_height_;
  FlatConcatByWidth flat_concat_by_width_;
};
}  
std::unique_ptr<NodeShader> NewAlignedConcatNodeShader() {
  return std::make_unique<AlignedConcatByChannels>();
}
std::unique_ptr<NodeShader> NewConcatNodeShader() {
  return std::make_unique<ConcatByAnyChannel>();
}
std::unique_ptr<NodeShader> NewFlatConcatNodeShader() {
  return std::make_unique<FlatConcat>();
}
}  
}  
}  