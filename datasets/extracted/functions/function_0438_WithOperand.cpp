#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include <cstdint>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace m = match;
class GemmBroadcastFoldingVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction *instr) override {
    HloInstruction *existing_gemm;
    HloInstruction *bcast;
    if (Match(instr, m::CustomCall(&existing_gemm,
                                   {kGemmCallTarget, kCublasLtMatmulCallTarget})
                         .WithOperand(0, m::Broadcast(&bcast, m::Op()))) ||
        (Match(instr, m::CustomCall(&existing_gemm, {kGemmCallTarget,
                                                     kCublasLtMatmulCallTarget})
                          .WithOperand(1, m::Broadcast(&bcast, m::Op()))))) {
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          existing_gemm->backend_config<GpuBackendConfig>());
      GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
      DotDimensionNumbers *dim_nums = config.mutable_dot_dimension_numbers();
      int bcast_operand_index = instr->operand_index(bcast);
      int num_bcast_dims = (bcast->shape().dimensions_size() -
                            bcast->operand(0)->shape().dimensions_size());
      int num_batch_dims = dim_nums->lhs_batch_dimensions_size();
      const tsl::protobuf::RepeatedField<int64_t> &batch_dimensions =
          (bcast_operand_index == 1) ? dim_nums->rhs_batch_dimensions()
                                     : dim_nums->lhs_batch_dimensions();
      for (int64_t bcast_dim : bcast->dimensions()) {
        if (bcast_dim < num_bcast_dims) {
          return absl::OkStatus();
        }
        if (absl::c_linear_search(batch_dimensions, bcast_dim)) {
          return absl::OkStatus();
        }
      }
      CHECK_GT(num_bcast_dims, 0);
      if (num_bcast_dims != num_batch_dims) {
        return absl::OkStatus();
      }
      if (bcast_operand_index == 1) {
        CHECK_EQ(dim_nums->rhs_contracting_dimensions_size(), 1);
        dim_nums->set_rhs_contracting_dimensions(
            0, dim_nums->rhs_contracting_dimensions(0) - num_batch_dims);
        dim_nums->clear_rhs_batch_dimensions();
      } else {
        CHECK_EQ(dim_nums->lhs_contracting_dimensions_size(), 1);
        dim_nums->set_lhs_contracting_dimensions(
            0, dim_nums->lhs_contracting_dimensions(0) - num_batch_dims);
        dim_nums->clear_lhs_batch_dimensions();
      }
      TF_RETURN_IF_ERROR(existing_gemm->ReplaceOperandWithDifferentShape(
          bcast_operand_index, bcast->mutable_operand(0)));
      TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(gpu_config));
      MarkAsChanged();
    }
    return absl::OkStatus();
  }
};
static absl::StatusOr<bool> RunOnComputation(HloComputation *computation) {
  GemmBroadcastFoldingVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}
absl::StatusOr<bool> GemmBroadcastFoldingRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}
}  
}  