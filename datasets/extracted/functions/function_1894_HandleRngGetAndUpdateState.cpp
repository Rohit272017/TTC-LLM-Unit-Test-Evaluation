#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include <memory>
#include <utility>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace spmd {
absl::Status StatefulRngSpmdPartitioningVisitor::HandleRngGetAndUpdateState(
    HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }
  TF_RET_CHECK(hlo->sharding().IsReplicated());
  auto clone =
      builder()->AddInstruction(hlo->CloneWithNewOperands(hlo->shape(), {}));
  clone->set_sharding(hlo->sharding());
  SetPartitionedHlo(
      hlo, spmd::PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
               .Reshard(hlo->sharding()));
  return absl::OkStatus();
}
std::unique_ptr<spmd::SpmdPartitioningVisitor>
StatefulRngSpmdPartitioner::CreateVisitor(
    HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
    const spmd::SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, spmd::SpmdLogger* logger,
    spmd::SpmdPartitionerOptions options, const CallGraph& call_graph) {
  return std::make_unique<StatefulRngSpmdPartitioningVisitor>(
      computation, num_partitions, num_replicas, collective_ops_creator,
      next_channel_id, logger, std::move(options), this, call_graph);
}
absl::Status StatefulRngSpmdPartitioner::PreprocessSharding(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState &&
          !hlo->has_sharding()) {
        hlo->set_sharding(HloSharding::Replicate());
      }
    }
  }
  return spmd::SpmdPartitioner::PreprocessSharding(module, execution_threads);
}
bool StatefulRngSpmdPartitioner::CanSideEffectingHaveReplicatedSharding(
    const HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState) return true;
  return spmd::SpmdPartitioner::CanSideEffectingHaveReplicatedSharding(hlo);
}
absl::Status StatefulRngSpmdPartitioner::HandleRotateRightWhilePreprocessing(
    HloComputation* computation) {
  if (!computation->IsWhileBodyComputation()) {
    return absl::OkStatus();
  }
  HloInstruction* while_loop = computation->WhileCallInstruction();
  TF_RET_CHECK(while_loop);
  if (computation->parent()
          ->config()
          .debug_options()
          .xla_gpu_unsafe_pipelined_loop_annotator()) {
    xla::FrontendAttributes attributes;
    (*attributes.mutable_map())["is_pipelined_while_loop"] = "true";
    while_loop->add_frontend_attributes(attributes);
  }
  return absl::OkStatus();
}
}  
}  