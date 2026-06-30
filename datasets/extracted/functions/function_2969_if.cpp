#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace gpu {
namespace {
class CustomCallVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction *hlo) override {
    if (hlo->custom_call_target() != kCuDnnFusionKind) {
      return absl::OkStatus();
    }
    HloComputation *computation = hlo->GetModule()->AddEmbeddedComputation(
        hlo->called_computations()[0]->Clone());
    HloInstruction *fusion =
        hlo->parent()->AddInstruction(HloInstruction::CreateFusion(
            hlo->shape(), HloInstruction::FusionKind::kCustom, hlo->operands(),
            computation));
    GpuBackendConfig gpu_config;
    FusionBackendConfig &backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.set_kind(hlo->custom_call_target());
    TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, fusion));
    return absl::OkStatus();
  }
};
}  
absl::StatusOr<bool> CuDnnCustomCallConverter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  return CustomCallVisitor().RunOnModule(module, execution_threads);
}
}  
}  