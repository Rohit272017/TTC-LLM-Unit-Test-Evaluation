#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"
#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace {
bool UseSubgroupBasedImpl(const GpuInfo& gpu_info) {
  return gpu_info.IsApiVulkan() &&
         (gpu_info.vulkan_info.api_version_major > 1 ||
          gpu_info.vulkan_info.api_version_minor >= 1) &&
         gpu_info.vulkan_info.subgroup_size >= 32 &&
         gpu_info.vulkan_info.supports_subgroup_arithmetic;
}
void GenerateSubgroupBasedMean(const NodeShader::GenerationContext& ctx,
                               GeneratedCode* generated_code) {
  int height = ctx.input_shapes[0][1];
  int width = ctx.input_shapes[0][2];
  int depth = ctx.input_shapes[0][3];
  std::vector<Variable> parameters = {
      {"input_data_0_h", height},
      {"input_data_0_w", width},
      {"output_data_0_h", 1},
      {"output_data_0_w", 1},
  };
  std::string source = R"(
  const uint columns_per_invocation =
      ($input_data_0_w$ + (gl_WorkGroupSize.x - 1))/gl_WorkGroupSize.x;
  const uint rows_per_invocation =
      ($input_data_0_h$ + (gl_WorkGroupSize.y - 1))/gl_WorkGroupSize.y;
  const uint first_row = gl_GlobalInvocationID.y*rows_per_invocation;
  const uint first_col = gl_GlobalInvocationID.x*columns_per_invocation;
  const uint last_row_exclusive =
      min(first_row+rows_per_invocation, $input_data_0_h$);
  const uint last_column_exclusive =
      min(first_col+columns_per_invocation, $input_data_0_w$);
  vec4 value = vec4(0);
  for (uint h = first_row; h < last_row_exclusive; ++h) {
    for (uint w = first_col; w < last_column_exclusive; ++w) {
      value += $input_data_0[w, h, gid.z]$;
    }
  }
  highp vec4 subgroup_sum = subgroupAdd(value);
  if(subgroupElect()) {
    subgroup_sums[gl_SubgroupID] = subgroup_sum;
  }
  memoryBarrierShared();
  barrier();
  if(gl_SubgroupID == 0) {
    highp vec4 subtotal = vec4(0);
    if (gl_SubgroupInvocationID < gl_NumSubgroups) {
      subtotal = subgroup_sums[gl_SubgroupInvocationID];
    }
    highp vec4 grand_total = subgroupAdd(subtotal);
    if(subgroupElect()) {
      highp vec4 result = grand_total / $input_data_0_w$ / $input_data_0_h$;
      $output_data_0[0, 0, gid.z] = result$;
    }
  }
  )";
  const uint32_t subgroup_size = ctx.gpu_info->vulkan_info.subgroup_size;
  const uint32_t max_wg_size_x = ctx.gpu_info->GetMaxWorkGroupSizeForX();
  const uint32_t max_wg_size_y = ctx.gpu_info->GetMaxWorkGroupSizeForY();
  const uint32_t max_wg_size =
      std::min(static_cast<uint32_t>(ctx.gpu_info->GetMaxWorkGroupTotalSize()),
               subgroup_size * subgroup_size);
  const uint32_t max_number_of_subgroups = max_wg_size / subgroup_size;
  uint32_t wg_size_x = 0;
  uint32_t wg_size_y = 0;
  if (width * height <= max_wg_size && width <= max_wg_size_x &&
      height <= max_wg_size_y) {
    wg_size_x = width;
    wg_size_y = height;
  } else {
    wg_size_x = std::min({static_cast<uint32_t>(std::sqrt(max_wg_size)),
                          max_wg_size_x, static_cast<uint32_t>(width)});
    wg_size_y = std::min({max_wg_size / wg_size_x, max_wg_size_y,
                          static_cast<uint32_t>(height)});
  }
  std::vector<Variable> shared_variables = {
      {"subgroup_sums", std::vector<float4>(max_number_of_subgroups)},
  };
  *generated_code = {
      std::move(parameters),
      {},
      {std::move(shared_variables)},
      uint3(wg_size_x, wg_size_y, uint32_t(DivideRoundUp(depth, 4))),
      uint3(wg_size_x, wg_size_y, 1u),
      std::move(source),
      IOStructure::ONLY_DEFINITIONS,
      IOStructure::ONLY_DEFINITIONS,
  };
}
void GenerateTrivialMean(const NodeShader::GenerationContext& ctx,
                         GeneratedCode* generated_code) {
  std::vector<Variable> parameters = {
      {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
      {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])}};
  std::string source = R"(
    highp vec4 sum = vec4(0.0);
    highp float size = float($input_data_0_w$ * $input_data_0_h$);
    for (int w = 0; w < $input_data_0_w$; w++) {
      for (int h = 0; h < $input_data_0_h$; h++) {
        sum += $input_data_0[w, h, gid.z]$;
      }
    }
    value_0 = sum / size;
  )";
  *generated_code = {
      std::move(parameters),
      {},
      {},
      uint3(),
      uint3(1, 1, 4),
      std::move(source),
      IOStructure::ONLY_DEFINITIONS,
      IOStructure::AUTO,
  };
}
constexpr uint3 kTileSize = {8, 8, 1};
inline bool UseTiledImpl(const NodeShader::GenerationContext& ctx) {
  const int h = ctx.input_shapes[0][1];
  const int w = ctx.input_shapes[0][2];
  const int c = ctx.input_shapes[0][3];
  return h % kTileSize.y == 0 && w % kTileSize.x == 0 && c % 4 == 0 &&
         (h / kTileSize.y) * (w / kTileSize.x) * c * sizeof(float) <=
             32768;  
}
void GenerateTiledMean(const NodeShader::GenerationContext& ctx,
                       GeneratedCode* generated_code) {
  const int h = ctx.input_shapes[0][1];
  const int w = ctx.input_shapes[0][2];
  const int s = DivideRoundUp(ctx.input_shapes[0][3], 4);
  std::vector<Variable> parameters = {
      {"input_data_0_h", h},
      {"input_data_0_w", w},
      {"tile_size_h", kTileSize.y},
      {"tile_size_w", kTileSize.x},
  };
  std::vector<Variable> shared_variables = {
      {"tile_sum",
       std::vector<float4>((w / kTileSize.x) * (h / kTileSize.y) * s)}};
  std::string source = R"(
  ivec2 tile_size = ivec2($tile_size_w$, $tile_size_h$);
  ivec2 num_tiles = ivec2($input_data_0_w$, $input_data_0_h$) / tile_size;
  highp vec4 partial_sum = vec4(0.0);
  for (int x = gid.x * tile_size.x; x < (gid.x + 1) * tile_size.x; ++x) {
    for (int y = gid.y * tile_size.y; y < (gid.y + 1) * tile_size.y; ++y) {
      partial_sum += $input_data_0[x, y, gid.z]$;
    }
  }
  $tile_sum$[num_tiles.x * num_tiles.y * gid.z + num_tiles.x * gid.y + gid.x] = partial_sum;
  memoryBarrierShared(); barrier();
  if (gid.x == 0 && gid.y == 0) {
    highp vec4 sum = vec4(0.0);
    for (int i = 0; i < num_tiles.x * num_tiles.y; ++i) {
      sum += $tile_sum$[num_tiles.x * num_tiles.y * gid.z + i];
    }
    highp vec4 mean = sum / float($input_data_0_w$ * $input_data_0_h$);
    $output_data_0[0, 0, gid.z] = mean$;
  }
)";
  *generated_code = {
      std::move(parameters),
      {},
      std::move(shared_variables),
      uint3(kTileSize.x, kTileSize.y, static_cast<uint32_t>(s)),
      kTileSize,
      std::move(source),
      IOStructure::ONLY_DEFINITIONS,
      IOStructure::ONLY_DEFINITIONS,
  };
}
class Mean : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const MeanAttributes&>(ctx.op_attr);
    if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
      return absl::InvalidArgumentError(
          "Mean calculation is supported only for height and width.");
    }
    if (!(ctx.input_shapes.size() == 1 && ctx.output_shapes.size() == 1 &&
          ctx.output_shapes[0][1] == 1 && ctx.output_shapes[0][2] == 1 &&
          ctx.output_shapes[0][3] == ctx.input_shapes[0][3])) {
      return absl::InvalidArgumentError(
          "Mean calculation is supported for one input and one 1x1 output with "
          "the same channel count.");
    }
    if (UseSubgroupBasedImpl(*ctx.gpu_info)) {
      GenerateSubgroupBasedMean(ctx, generated_code);
    } else if (UseTiledImpl(ctx)) {
      GenerateTiledMean(ctx, generated_code);
    } else {
      GenerateTrivialMean(ctx, generated_code);
    }
    return absl::OkStatus();
  }
};
}  
std::unique_ptr<NodeShader> NewMeanNodeShader() {
  return std::make_unique<Mean>();
}
}  
}  
}  