#include "xla/service/cpu/parallel_task_assignment.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/shape_partition.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"  
#include "tsl/platform/status.h"
namespace xla {
namespace cpu {
class SimpleCostModel : public ParallelCostModel {
 public:
  SimpleCostModel(const int64_t max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~SimpleCostModel() override {}
  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    const int64_t instruction_cost = shape_size_(instruction->shape());
    const int64_t min_cost_per_thread = 256LL << 10;  
    return std::min(
        max_parallelism_,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }
 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};
class DefaultCostModel : public ParallelCostModel {
 public:
  DefaultCostModel(const int64_t max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        shape_size_(shape_size),
        cost_analysis_(std::move(cost_analysis)) {}
  ~DefaultCostModel() override {}
  int64_t GetParallelTaskCount(HloInstruction* instruction) override {
    int64_t instruction_cost;
    int64_t min_cost_per_thread;
    int64_t max_parallelism;
    const int64_t bytes_accessed =
        std::max(int64_t{1}, cost_analysis_->bytes_accessed(*instruction));
    const float flops_to_bytes_ratio =
        cost_analysis_->flop_count(*instruction) /
        static_cast<float>(bytes_accessed);
    if (flops_to_bytes_ratio <= 1.0) {
      max_parallelism = std::min<int64_t>(
          max_parallelism_, std::ceil(std::sqrt(tsl::port::MaxParallelism())));
      instruction_cost = bytes_accessed;
      min_cost_per_thread = 256LL << 10;  
    } else {
      max_parallelism = max_parallelism_;
      instruction_cost =
          1 * cost_analysis_->flop_count(*instruction) +
          2 * cost_analysis_->transcendental_count(*instruction) +
          10 * cost_analysis_->bytes_accessed(*instruction);
      min_cost_per_thread = 100000;
    }
    return std::min(
        max_parallelism,
        std::max(int64_t{1}, instruction_cost / min_cost_per_thread));
  }
 private:
  const int64_t max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};
ParallelTaskAssignment::ParallelTaskAssignment(
    const int64_t max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module,
    const TargetMachineFeatures* target_machine_features)
    : target_machine_features_(*target_machine_features) {
  VLOG(1) << "ParallelTaskAssignment max_parallelism: " << max_parallelism;
  auto cost_analysis = std::make_unique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  absl::Status status =
      computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    cost_model_ = std::make_unique<DefaultCostModel>(
        max_parallelism, shape_size, std::move(cost_analysis));
  } else {
    cost_model_ =
        std::make_unique<SimpleCostModel>(max_parallelism, shape_size);
  }
}
int64_t ParallelTaskAssignment::GetTargetParallelTaskCount(
    HloInstruction* instruction) {
  auto opcode = instruction->opcode();
  if (llvm_ir::MayBeImplementedAsInPlaceDynamicUpdateSlice(instruction) ||
      instruction->shape().IsTuple() || opcode == HloOpcode::kRng ||
      opcode == HloOpcode::kConstant) {
    return 1;
  }
  if (instruction->IsElementwise() || instruction->IsLoopFusion() ||
      opcode == HloOpcode::kBroadcast || opcode == HloOpcode::kConcatenate ||
      opcode == HloOpcode::kDynamicSlice ||
      opcode == HloOpcode::kDynamicUpdateSlice ||
      opcode == HloOpcode::kGather || opcode == HloOpcode::kIota ||
      opcode == HloOpcode::kPad || opcode == HloOpcode::kReduce ||
      opcode == HloOpcode::kReduceWindow || opcode == HloOpcode::kReshape ||
      opcode == HloOpcode::kReverse || opcode == HloOpcode::kSlice ||
      opcode == HloOpcode::kTranspose ||
      (opcode == HloOpcode::kConvolution &&
       !PotentiallyImplementedAsEigenConvolution(*instruction,
                                                 target_machine_features_))) {
    return cost_model_->GetParallelTaskCount(instruction);
  }
  return 1;
}
absl::StatusOr<bool> ParallelTaskAssigner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "ParallelTaskAssigner ENTRY");
  XLA_VLOG_LINES(3, module->ToString());
  HloToParallelTasks hlo_to_parallel_tasks;
  ComputeTargetParallelTasks(module, &hlo_to_parallel_tasks);
  bool changed = AssignParallelTasks(module, hlo_to_parallel_tasks);
  XLA_VLOG_LINES(2, "ParallelTaskAssigner EXIT");
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}
bool ParallelTaskAssigner::AssignParallelTasks(
    HloModule* module, const HloToParallelTasks& hlo_to_parallel_tasks) {
  return AssignParallelTasksHelper(module, module->entry_computation(),
                                   hlo_to_parallel_tasks);
}
bool ParallelTaskAssigner::AssignParallelTasksHelper(
    HloModule* module, HloComputation* computation,
    const HloToParallelTasks& hlo_to_parallel_tasks) {
  bool changed = false;
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  for (auto* instruction : instructions) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      changed |= AssignParallelTasksHelper(module, instruction->while_body(),
                                           hlo_to_parallel_tasks);
      continue;
    } else if (instruction->opcode() == HloOpcode::kCall) {
      changed |= AssignParallelTasksHelper(module, instruction->to_apply(),
                                           hlo_to_parallel_tasks);
      continue;
    }
    auto it = hlo_to_parallel_tasks.find(instruction);
    if (it == hlo_to_parallel_tasks.end()) {
      continue;
    }
    const int64_t target_parallel_task_count = (*it).second;
    auto dim_partition_counts = ShapePartitionAssigner(instruction->shape())
                                    .Run(target_parallel_task_count);
    const int64_t total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    if (total_partition_count <= 1) {
      continue;
    }
    auto* call = module->OutlineExpressionFromComputation(
        {instruction}, absl::StrCat("parallel_", instruction->name()),
        computation);
    auto* new_root = call->to_apply()->root_instruction();
    BackendConfig backend_config;
    absl::c_copy(dim_partition_counts,
                 tsl::protobuf::RepeatedFieldBackInserter(
                     backend_config.mutable_outer_dimension_partitions()));
    TF_CHECK_OK(new_root->set_backend_config(backend_config));
    VLOG(2) << "Assigned parallel task count: " << total_partition_count
            << " to instruction: " << new_root->name()
            << " parent: " << new_root->parent()->name();
    changed = true;
  }
  return changed;
}
void ParallelTaskAssigner::ComputeTargetParallelTasks(
    HloModule* module, HloToParallelTasks* hlo_to_parallel_tasks) {
  ParallelTaskAssignment parallel_task_assignment(max_parallelism_,
                                                  shape_size_function_, module,
                                                  &target_machine_features_);
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      const int64_t target_parallel_task_count =
          parallel_task_assignment.GetTargetParallelTaskCount(instruction);
      if (target_parallel_task_count > 1) {
        hlo_to_parallel_tasks->insert(
            {instruction, target_parallel_task_count});
      }
    }
  }
}
}  
}  