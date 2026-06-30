#include "xla/service/gpu/reduction_utils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <ostream>
#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#ifdef GOOGLE_CUDA
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#endif  
namespace xla {
namespace gpu {
namespace {
Vector3 PartitionShapeByMiddleDimensions(
    const Shape& shape, absl::Span<const int64_t> dims_middle) {
  CHECK(LayoutUtil::AreDimensionsConsecutive(shape.layout(), dims_middle));
  Vector3 values = {1, 1, 1};
  enum Segment { kMajor = 0, kMiddle = 1, kMinor = 2 };
  Segment cur_segment = kMinor;
  for (int64_t cur_dim : LayoutUtil::MinorToMajor(shape)) {
    if (cur_segment != kMajor) {
      bool cur_dim_in_middle = absl::c_linear_search(dims_middle, cur_dim);
      if (cur_segment == kMinor) {
        if (cur_dim_in_middle) {
          cur_segment = kMiddle;
        }
      } else if (cur_segment == kMiddle) {
        if (!cur_dim_in_middle) {
          cur_segment = kMajor;
        }
      }
    }
    values[cur_segment] *= shape.dimensions(cur_dim);
  }
  return values;
}
}  
int64_t MinThreadsXRowReduction(const HloModuleConfig& hlo_module_config) {
#ifdef GOOGLE_CUDA
  static absl::Mutex mutex(absl::kConstInit);
  static std::atomic<bool*> use_reduced_thread_count_atomic = nullptr;
  bool* use_reduced_thread_count =
      use_reduced_thread_count_atomic.load(std::memory_order_acquire);
  if (use_reduced_thread_count == nullptr) {
    absl::MutexLock lock(&mutex);
    use_reduced_thread_count =
        use_reduced_thread_count_atomic.load(std::memory_order_relaxed);
    if (use_reduced_thread_count == nullptr) {
      auto ptxas_config =
          PtxOptsFromDebugOptions(hlo_module_config.debug_options());
      auto ptxas_version_tuple =
          se::GetAsmCompilerVersion(ptxas_config.preferred_cuda_dir);
      use_reduced_thread_count = new bool(false);
      if (!ptxas_version_tuple.ok() ||
          ptxas_version_tuple.value() <
              stream_executor::SemanticVersion{12, 2, 0}) {
        *use_reduced_thread_count = true;
      }
      use_reduced_thread_count_atomic.store(use_reduced_thread_count,
                                            std::memory_order_release);
    }
  }
  if (*use_reduced_thread_count) {
    return 512;
  }
#endif  
  return 1024;
}
Vector3 GetReductionTiling(const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    int64_t tile_z = std::min(reduction_dimensions.dimensions[0],
                              BatchedReductionRaceFreeBound());
    return {tile_z, 1, 16};
  }
  return {1, 128, 1};
}
int64_t ReductionDimensionRaceFreeBound(
    const HloModuleConfig& hlo_module_config,
    const ReductionDimensions& reduction_dimensions) {
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);
  if (reduction_dimensions.is_row_reduction) {
    return MinThreadsXRowReduction(hlo_module_config) * reduction_tiling[2];
  }
  return WarpSize() * reduction_tiling[1];
}
bool IsUnnestedReductionFasterThanElemental(
    const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    return (reduction_dimensions.dimensions[2] >= WarpSize()) ||
           ((WarpSize() % reduction_dimensions.dimensions[2]) == 0);
  }
  int64_t major_size = reduction_dimensions.dimensions[1];
  int64_t minor_size = reduction_dimensions.dimensions[2];
  bool prefer_elemental_emitter =
      (major_size < WarpSize()) ||
      (major_size < 2 * WarpSize() && minor_size < WarpSize()) ||
      (major_size < 4 * WarpSize() && minor_size < 8) ||
      (major_size < 8 * WarpSize() && minor_size < 3);
  return !prefer_elemental_emitter;
}
bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce) {
  if (reduce.opcode() != HloOpcode::kReduce) {
    return false;
  }
  const Shape& operand_shape = reduce.operand(0)->shape();
  absl::Span<const int64_t> dims_to_reduce = reduce.dimensions();
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < operand_shape.dimensions().size(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }
  return (LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_keep) ||
          LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_reduce)) &&
         IsUnnestedReductionFasterThanElemental(
             GetReductionKindAndContiguousComponents(reduce));
}
bool ReductionIsRaceFree(const HloModuleConfig& hlo_module_config,
                         const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    return reduction_dimensions.dimensions[2] <=
               ReductionDimensionRaceFreeBound(hlo_module_config,
                                               reduction_dimensions) &&
           reduction_dimensions.dimensions[0] <=
               BatchedReductionRaceFreeBound();
  }
  return reduction_dimensions.dimensions[1] <=
         ReductionDimensionRaceFreeBound(hlo_module_config,
                                         reduction_dimensions);
}
std::ostream& operator<<(std::ostream& os,
                         const ReductionDimensions& reduction_dimensions) {
  bool is_row_reduction = reduction_dimensions.is_row_reduction;
  os << (is_row_reduction ? "row " : "column ") << "reduction ["
     << absl::StrJoin(reduction_dimensions.dimensions, ",") << "] -> ["
     << reduction_dimensions.dimensions[0] << ", "
     << reduction_dimensions
            .dimensions[is_row_reduction
                            ? ReductionDimensions::kRowKeptDimension
                            : ReductionDimensions::kColMinorKeptDimension]
     << "]";
  return os;
}
ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce) {
  Shape input_shape = reduce.operand(0)->shape();
  absl::Span<const int64_t> dims_to_reduce = reduce.dimensions();
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < input_shape.rank(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }
  if (dims_to_keep.empty()) {
    return {true,
            {1, 1, ShapeUtil::ElementsIn(input_shape)}};
  }
  if (LayoutUtil::AreDimensionsConsecutive(input_shape.layout(),
                                           dims_to_keep)) {
    Vector3 shape_partition =
        PartitionShapeByMiddleDimensions(input_shape, dims_to_keep);
    if (shape_partition[1] == 1) {
      return {true,
              {1, 1, shape_partition[0] * shape_partition[2]}};
    }
    if (shape_partition[2] == 1) {
      return {false,
              {1, shape_partition[0], shape_partition[1]}};
    }
    return {true, shape_partition};
  }
  Vector3 shape_partition =
      PartitionShapeByMiddleDimensions(input_shape, dims_to_reduce);
  if (shape_partition[2] == 1) {
    return {true,
            {1, shape_partition[0], shape_partition[1]}};
  }
  return {false, shape_partition};
}
bool IsRealReductionHero(const HloInstruction& root,
                         const HloInstruction& hero) {
  if (!IsReductionFromOrToContiguousDimensions(hero)) {
    return false;
  }
  return &root == &hero ||
         ReductionIsRaceFree(hero.GetModule()->config(),
                             GetReductionKindAndContiguousComponents(hero));
}
bool AreReductionsMultiOutputFusionCompatible(
    const HloInstruction* reduce_hero, const HloInstruction* first_reduce) {
  return GetReductionKindAndContiguousComponents(*reduce_hero) ==
         GetReductionKindAndContiguousComponents(*first_reduce);
}
}  
}  