#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
absl::InlinedVector<int64_t, 2> GetSortedReducedDims(
    HloReduceInstruction *reduce) {
  absl::InlinedVector<int64_t, 2> reduced_dims{reduce->dimensions().begin(),
                                               reduce->dimensions().end()};
  absl::c_sort(reduced_dims);
  return reduced_dims;
}
bool IsMinMaxReduction(HloReduceInstruction *reduce) {
  HloComputation *called = &reduce->to_apply()[0];
  if (auto reduction_kind = MatchReductionComputation(called)) {
    return reduction_kind == ReductionKind::MAX ||
           reduction_kind == ReductionKind::MIN;
  }
  return false;
}
}  
class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}
  absl::Status HandleReduce(HloInstruction *hlo) override {
    auto *reduce = Cast<HloReduceInstruction>(hlo);
    VLOG(3) << "Reduction instruction: " << reduce->ToString();
    const HloModuleConfig &config = reduce->GetModule()->config();
    if (!MatchReductionForSplit(reduce, config)) {
      return absl::OkStatus();
    }
    ReductionDimensions reduction_dims =
        GetReductionKindAndContiguousComponents(*hlo);
    if (ReductionIsRaceFree(config, reduction_dims)) {
      VLOG(3) << "Base case: dimensions fit";
      return absl::OkStatus();
    }
    auto sorted_dims_to_reduce = GetSortedReducedDims(reduce);
    CHECK_LE(sorted_dims_to_reduce.size(), 2);
    if (reduction_dims.is_row_reduction &&
        reduction_dims
                .dimensions[ReductionDimensions::kRowMajorReducedDimension] >
            BatchedReductionRaceFreeBound()) {
      VLOG(2) << "Splitting batched dimension reduce into a separate reduction";
      return RewriteBatchDimensionLargerThanTile(reduce, reduction_dims,
                                                 sorted_dims_to_reduce);
    }
    SplitParams split_params =
        ComputeSplitParams(reduce, reduction_dims, sorted_dims_to_reduce);
    return SplitReductionDimension(reduce, split_params, sorted_dims_to_reduce);
  }
 private:
  bool MatchReductionForSplit(HloReduceInstruction *reduce,
                              const HloModuleConfig &config) {
    bool reductions_via_mlir_disabled =
        config.debug_options().xla_gpu_mlir_emitter_level() < 4;
    if (reductions_via_mlir_disabled && IsMinMaxReduction(reduce)) {
      VLOG(1) << "Not performing tree expansion on min/max-reduction: "
              << reduce->ToString()
              << " since min/max operations are associative";
      return false;
    }
    if (!IsReductionFromOrToContiguousDimensions(*reduce)) {
      VLOG(3) << "Is not a reduction from or to contiguous dimensions";
      return false;
    }
    VLOG(3) << "Perform rewrite";
    return true;
  }
  bool ShouldSwapInnerAndOuterReducedMinorDimension(uint64_t k1, uint64_t k2,
                                                    uint64_t n,
                                                    int64_t race_free_bound,
                                                    bool is_row_reduction) {
    CHECK(k1 >= k2);
    if (k1 > race_free_bound) {
      return false;
    }
    if (is_row_reduction) {
      bool maybe_vectorized = k2 % 2 == 0 && n % 2 == 0;
      if (maybe_vectorized) {
        return k2 * 2 < k1 || k1 % 2 == 0;
      }
      return n % 2 == 0 || k1 % 2 != 0;
    }
    return true;
  }
  struct SplitParams {
    int64_t k1;
    int64_t k2;
    int64_t dim;
  };
  SplitParams ComputeSplitParams(
      HloReduceInstruction *reduce, const ReductionDimensions &reduction_dims,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    absl::Span<int64_t const> input_shape_dims =
        reduce->inputs()[0]->shape().dimensions();
    int64_t reduced_dim = sorted_dims_to_reduce.back();
    int64_t reduced_dim_size = input_shape_dims[reduced_dim];
    VLOG(3) << "reduced dim size = " << reduced_dim_size;
    uint64_t k2 =
        static_cast<uint64_t>(std::floor(std::sqrt(reduced_dim_size)));
    int64_t race_free_bound = ReductionDimensionRaceFreeBound(
        reduce->GetModule()->config(), reduction_dims);
    if (k2 > race_free_bound) {
      k2 = race_free_bound;
    }
    uint64_t minimum_padding = (k2 - reduced_dim_size % k2) % k2;
    uint64_t best_k1 = (reduced_dim_size + minimum_padding) / k2;
    for (uint64_t i = k2 - 1; i > k2 / 2; --i) {
      uint64_t padding = (i - reduced_dim_size % i) % i;
      if (padding < minimum_padding ||
          (padding == minimum_padding && absl::has_single_bit(i))) {
        minimum_padding = padding;
        best_k1 = (reduced_dim_size + padding) / i;
      }
    }
    uint64_t padded_k = reduced_dim_size + minimum_padding;
    uint64_t best_k2 = padded_k / best_k1;
    if (ShouldSwapInnerAndOuterReducedMinorDimension(
            best_k1, best_k2, reduced_dim_size, race_free_bound,
            reduction_dims.is_row_reduction)) {
      std::swap(best_k1, best_k2);
    }
    return SplitParams{static_cast<int64_t>(best_k1),
                       static_cast<int64_t>(best_k2), reduced_dim};
  }
  absl::Status SplitReductionDimension(
      HloReduceInstruction *reduce, const SplitParams &split_params,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    absl::Span<int64_t const> reduce_input_dims =
        reduce->inputs()[0]->shape().dimensions();
    int64_t split_dim_size = reduce_input_dims[split_params.dim];
    VLOG(2) << "dimension to split = " << split_params.dim << " with "
            << split_dim_size << " elements into " << split_params.k1 << " by "
            << split_params.k2;
    HloInstruction::InstructionVector padded_inputs(reduce->inputs().begin(),
                                                    reduce->inputs().end());
    auto padded_size = split_params.k1 * split_params.k2;
    absl::InlinedVector<int64_t, 3> padded_dimensions(reduce_input_dims.begin(),
                                                      reduce_input_dims.end());
    if (split_dim_size != padded_size) {
      padded_dimensions[split_params.dim] = padded_size;
      PaddingConfig padding_config =
          MakeNoPaddingConfig(reduce_input_dims.size());
      padding_config.mutable_dimensions(split_params.dim)
          ->set_edge_padding_high(padded_size - split_dim_size);
      for (int input_idx = 0; input_idx < padded_inputs.size(); ++input_idx) {
        auto &reduction_input = padded_inputs[input_idx];
        Shape padded_shape = ShapeUtil::MakeShape(
            reduction_input->shape().element_type(), padded_dimensions);
        VLOG(2) << "Generated padded shape: " << padded_shape.ToString();
        reduction_input = reduce->parent()->AddInstruction(
            HloInstruction::CreatePad(padded_shape, reduction_input,
                                      reduce->init_values()[input_idx],
                                      padding_config),
            &reduction_input->metadata());
      }
    }
    absl::InlinedVector<int64_t, 3> reshaped_dimensions;
    int64_t input_rank = reduce_input_dims.size();
    for (int64_t dim_idx = 0; dim_idx < input_rank; dim_idx++) {
      if (dim_idx == split_params.dim) {
        reshaped_dimensions.push_back(split_params.k1);
        reshaped_dimensions.push_back(split_params.k2);
      } else {
        reshaped_dimensions.push_back(padded_dimensions[dim_idx]);
      }
    }
    absl::InlinedVector<int64_t, 2> inner_reduce_dims(
        sorted_dims_to_reduce.begin(), sorted_dims_to_reduce.end());
    auto split_dim_it = std::find(inner_reduce_dims.begin(),
                                  inner_reduce_dims.end(), split_params.dim);
    *split_dim_it += 1;
    absl::InlinedVector<int64_t, 1> outer_reduce_dims{
        split_params.dim -
        std::distance(inner_reduce_dims.begin(), split_dim_it)};
    absl::InlinedVector<int64_t, 3> inner_reduce_shape =
        RemoveElements(inner_reduce_dims, reshaped_dimensions);
    HloInstruction::InstructionVector reshaped_padded_inputs;
    absl::InlinedVector<Shape, 2> inner_reduce_shapes;
    for (HloInstruction *padded_input : padded_inputs) {
      Shape reshaped_shape = ShapeUtil::MakeShape(
          padded_input->shape().element_type(), reshaped_dimensions);
      HloInstruction *reshaped_padded_input = reduce->parent()->AddInstruction(
          HloInstruction::CreateBitcast(reshaped_shape, padded_input),
          &padded_input->metadata());
      VLOG(2) << "Generated reshape: " << reshaped_padded_input->ToString();
      reshaped_padded_inputs.push_back(reshaped_padded_input);
      inner_reduce_shapes.push_back(ShapeUtil::MakeShape(
          padded_input->shape().element_type(), inner_reduce_shape));
    }
    HloInstruction *inner_reduce = reduce->parent()->AddInstruction(
        HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(inner_reduce_shapes),
            reshaped_padded_inputs, reduce->init_values(), inner_reduce_dims,
            reduce->to_apply()),
        &reduce->metadata());
    VLOG(1) << "Generated inner reduction: " << inner_reduce->ToString();
    std::unique_ptr<HloInstruction> outer_reduce = HloInstruction::CreateReduce(
        reduce->shape(), inner_reduce, reduce->init_values(), outer_reduce_dims,
        reduce->to_apply());
    VLOG(1) << "Generated outer reduction: " << outer_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(outer_reduce));
  }
  absl::Status RewriteBatchDimensionLargerThanTile(
      HloReduceInstruction *hlo,
      const ReductionDimensions &reduction_dimensions,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    CHECK(reduction_dimensions.is_row_reduction);
    absl::InlinedVector<Shape, 2> tuple_shapes;
    int64_t minor_reduction_dim = sorted_dims_to_reduce.back();
    for (HloInstruction *input : hlo->inputs()) {
      tuple_shapes.push_back(
          ShapeUtil::DeleteDimension(minor_reduction_dim, input->shape()));
    }
    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(tuple_shapes), hlo->inputs(),
            hlo->init_values(), {minor_reduction_dim}, hlo->to_apply()));
    VLOG(1) << "Inner reduction: " << inner_reduce->ToString();
    std::unique_ptr<HloInstruction> out = HloInstruction::CreateReduce(
        hlo->shape(), inner_reduce, hlo->init_values(), {0}, hlo->to_apply());
    VLOG(1) << "Generated: " << out->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(out));
  }
  se::GpuComputeCapability gpu_version_;
};
absl::StatusOr<bool> TreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  VLOG(5) << "Rewriter input: " << module->ToString();
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionRewriterVisitor(gpu_version_)
                          .RunOnModule(module, execution_threads));
  VLOG(5) << "Rewriter output: " << module->ToString();
  return changed;
}
}  
}  