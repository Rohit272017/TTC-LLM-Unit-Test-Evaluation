#include "xla/service/hlo_memory_scheduler.h"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"
namespace xla {
namespace {
using ::tsl::strings::HumanReadableNumBytes;
class ListScheduler {
 public:
  static absl::StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const BufferValue::SizeFunction& size_function) {
    ListScheduler scheduler(computation, points_to_analysis, size_function);
    return scheduler.CreateSchedule();
  }
  static bool IgnoreInstruction(const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kParameter ||
           instruction.opcode() == HloOpcode::kConstant;
  }
 private:
  using Priority = std::pair<int64_t, int64_t>;
  ListScheduler(HloComputation* computation,
                const TuplePointsToAnalysis& points_to_analysis,
                const BufferValue::SizeFunction& size_function)
      : computation_(computation),
        points_to_analysis_(points_to_analysis),
        size_function_(size_function) {
    for (auto* instruction : computation->instructions()) {
      absl::flat_hash_set<const LogicalBuffer*> instr_uses;
      for (auto* operand : instruction->operands()) {
        points_to_analysis.GetPointsToSet(operand).ForEachElement(
            [&](const ShapeIndex& ,
                const PointsToSet::BufferList& buffers) {
              instr_uses.insert(buffers.begin(), buffers.end());
            });
      }
      buffer_uses_[instruction] = std::vector<const LogicalBuffer*>(
          instr_uses.begin(), instr_uses.end());
    }
    unscheduled_use_count_.reserve(points_to_analysis.num_logical_buffers());
    for (auto* instruction : computation->instructions()) {
      for (auto* buffer :
           points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
        unscheduled_use_count_[buffer] = 0;
      }
    }
    for (auto* instruction : computation->instructions()) {
      for (const LogicalBuffer* buffer : buffer_uses_.at(instruction)) {
        ++unscheduled_use_count_[buffer];
      }
    }
    for (const LogicalBuffer* live_out_buffer :
         points_to_analysis.GetPointsToSet(computation->root_instruction())
             .CreateFlattenedSet()) {
      ++unscheduled_use_count_[live_out_buffer];
    }
  }
  static bool IgnoreBuffer(const LogicalBuffer& buffer) {
    return IgnoreInstruction(*buffer.instruction());
  }
  struct ReadyListEntry {
    HloInstruction* instruction;
    int64_t bytes_defined;
    std::vector<const std::pair<const LogicalBuffer* const, int64_t>*>
        used_buffer_unscheduled_use_counts;
  };
  ReadyListEntry MakeReadyListEntry(HloInstruction* instruction) {
    ReadyListEntry entry;
    entry.instruction = instruction;
    entry.bytes_defined = 0;
    for (auto* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (!IgnoreBuffer(*buffer)) {
        entry.bytes_defined += size_function_(*buffer);
      }
    }
    for (auto* buffer : buffer_uses_.at(instruction)) {
      if (IgnoreBuffer(*buffer)) {
        continue;
      }
      auto unscheduled_use_count_it = unscheduled_use_count_.find(buffer);
      CHECK(unscheduled_use_count_it != unscheduled_use_count_.end());
      entry.used_buffer_unscheduled_use_counts.push_back(
          &*unscheduled_use_count_it);
    }
    return entry;
  }
  int64_t BytesFreedIfScheduled(const ReadyListEntry& entry) {
    auto instruction = entry.instruction;
    auto opcode = instruction->opcode();
    if (opcode == HloOpcode::kOutfeed &&
        !instruction->outfeed_config().empty()) {
      return INT_MAX;
    }
    if (opcode == HloOpcode::kInfeed && !instruction->infeed_config().empty()) {
      return INT_MIN;
    }
    int64_t freed_bytes = 0;
    for (const auto& kv : entry.used_buffer_unscheduled_use_counts) {
      auto buffer = kv->first;
      auto use_count = kv->second;
      if (use_count == 1) {
        freed_bytes += size_function_(*buffer);
      }
    }
    return freed_bytes - entry.bytes_defined;
  }
  Priority GetPriority(const ReadyListEntry& entry) {
    if (ShapeUtil::IsEffectiveScalar(entry.instruction->shape())) {
      return {std::numeric_limits<int64_t>::max(),
              std::numeric_limits<int64_t>::max()};
    }
    return {BytesFreedIfScheduled(entry), entry.instruction->user_count()};
  }
  HloInstructionSequence CreateSchedule() {
    HloInstructionSequence schedule;
    absl::flat_hash_map<const HloInstruction*, int64_t> unscheduled_pred_count;
    for (auto* instruction : computation_->instructions()) {
      for (HloInstruction* user : instruction->users()) {
        unscheduled_pred_count[user]++;
      }
      for (HloInstruction* succ : instruction->control_successors()) {
        unscheduled_pred_count[succ]++;
      }
    }
    std::multimap<Priority, ReadyListEntry> ready_queue;
    absl::flat_hash_map<const HloInstruction*,
                        std::multimap<Priority, ReadyListEntry>::iterator>
        ready_instructions;
    auto add_to_ready_queue = [&](HloInstruction* inst) {
      auto entry = MakeReadyListEntry(inst);
      auto it = ready_queue.emplace(GetPriority(entry), std::move(entry));
      ready_instructions[inst] = it;
    };
    for (auto* instruction : computation_->instructions()) {
      if (instruction->operands().empty() &&
          instruction->control_predecessors().empty()) {
        add_to_ready_queue(instruction);
      }
    }
    while (!ready_queue.empty()) {
      auto best_it = ready_queue.end();
      --best_it;
      HloInstruction* best = best_it->second.instruction;
      VLOG(2) << "Schedule instruction: " << best->ToShortString()
              << " Bytes freed: " << best_it->first.first;
      ready_queue.erase(best_it);
      ready_instructions.erase(best);
      schedule.push_back(best);
      scheduled_instructions_.insert(best);
      bool adjust_ready_queue = false;
      for (const LogicalBuffer* buffer : buffer_uses_.at(best)) {
        int64_t& count = unscheduled_use_count_[buffer];
        CHECK_GT(count, 0);
        --count;
        if (count == 1) {
          adjust_ready_queue = true;
        }
      }
      auto update_pred_count = [&](HloInstruction* inst) {
        int64_t pred_count = --unscheduled_pred_count.at(inst);
        CHECK_GE(pred_count, 0);
        if (pred_count == 0) {
          add_to_ready_queue(inst);
        }
      };
      for (HloInstruction* user : best->users()) {
        update_pred_count(user);
      }
      for (HloInstruction* succ : best->control_successors()) {
        update_pred_count(succ);
      }
      if (adjust_ready_queue) {
        for (HloInstruction* operand : best->operands()) {
          for (HloInstruction* operand_user : operand->users()) {
            auto ready_instructions_it = ready_instructions.find(operand_user);
            if (ready_instructions_it == ready_instructions.end()) {
              continue;
            }
            auto ready_queue_it = ready_instructions_it->second;
            auto& entry = ready_queue_it->second;
            Priority new_priority = GetPriority(entry);
            if (new_priority == ready_queue_it->first) {
              continue;
            }
            ready_instructions_it->second =
                ready_queue.emplace(new_priority, std::move(entry));
            ready_queue.erase(ready_queue_it);
          }
        }
      }
    }
    CHECK_EQ(schedule.size(), computation_->instruction_count());
    CHECK_EQ(scheduled_instructions_.size(), computation_->instruction_count());
    return schedule;
  }
  HloComputation* computation_;
  const TuplePointsToAnalysis& points_to_analysis_;
  const BufferValue::SizeFunction& size_function_;
  absl::flat_hash_map<const HloInstruction*, std::vector<const LogicalBuffer*>>
      buffer_uses_;
  absl::flat_hash_map<const LogicalBuffer*, int64_t> unscheduled_use_count_;
  absl::flat_hash_set<const HloInstruction*> scheduled_instructions_;
};
int64_t SumLogicalBufferSizes(
    const TuplePointsToAnalysis::BufferDefinitionVector& buffers,
    const BufferValue::SizeFunction& size_function) {
  int64_t size = 0;
  for (const LogicalBuffer* buffer : buffers) {
    size += size_function(*buffer);
  }
  return size;
}
absl::StatusOr<HloInstructionSequence> ScheduleComputationHelper(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerAlgorithm& algorithm,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  VLOG(2) << "Computation: " << computation->name();
  if (algorithm) {
    return algorithm(computation, points_to_analysis, alias_analysis,
                     size_function, postprocessor, peak_memory);
  }
  return DefaultMemoryScheduler(computation, points_to_analysis, alias_analysis,
                                size_function, postprocessor, peak_memory);
}
}  
absl::StatusOr<HloInstructionSequence> DFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  int64_t cumulative_total_size = 0;
  int64_t total_hlos = computation->instruction_count();
  struct Stats {
    int64_t extra_users = 0;
    int64_t total_sizes = 0;
  };
  absl::flat_hash_map<const HloInstruction*, Stats> stats_map;
  stats_map.reserve(computation->instruction_count());
  for (const HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    auto& stats = stats_map[hlo];
    if (ListScheduler::IgnoreInstruction(*hlo)) {
      continue;
    }
    stats.extra_users = hlo->users().empty() ? 0 : hlo->users().size() - 1;
    int64_t logical_buffer_size = SumLogicalBufferSizes(
        points_to_analysis.GetBuffersDefinedByInstruction(hlo), size_function);
    stats.total_sizes = logical_buffer_size;
    cumulative_total_size += logical_buffer_size;
    absl::flat_hash_set<const HloInstruction*> unique_operands(
        hlo->operands().begin(), hlo->operands().end());
    for (const HloInstruction* operand : unique_operands) {
      auto& operand_stats = stats_map.at(operand);
      stats.extra_users += operand_stats.extra_users;
      stats.total_sizes += operand_stats.total_sizes;
    }
    stats.total_sizes = std::min(stats.total_sizes, cumulative_total_size);
    stats.extra_users = std::min(stats.extra_users, total_hlos);
  }
  CHECK_EQ(stats_map.size(), computation->instruction_count());
  HloInstructionSequence sequence;
  FunctionVisitor visitor([&sequence](HloInstruction* hlo) {
    sequence.push_back(hlo);
    return absl::OkStatus();
  });
  visitor.ReserveVisitStates(computation->instruction_count());
  TF_RETURN_IF_ERROR(computation->AcceptWithOperandOrder(
      &visitor, [&stats_map](const HloInstruction* a, const HloInstruction* b) {
        auto& stats_a = stats_map.at(a);
        auto& stats_b = stats_map.at(b);
        if (stats_a.extra_users != stats_b.extra_users) {
          return stats_a.extra_users > stats_b.extra_users;
        }
        if (stats_a.total_sizes != stats_b.total_sizes) {
          return stats_a.total_sizes > stats_b.total_sizes;
        }
        return a->name() < b->name();
      }));
  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  CHECK_EQ(sequence.size(), computation->instruction_count());
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory,
        HeapSimulator::MinimumMemoryForComputation(
            *computation, sequence, alias_analysis, size_function));
  }
  return sequence;
}
absl::StatusOr<HloInstructionSequence> BFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  absl::flat_hash_map<const HloInstruction*, int64_t> inst_index;
  std::vector<int64_t> inst_deps(computation->instruction_count(), 0);
  std::queue<HloInstruction*> ready_queue;
  auto update_queue = [&](HloInstruction* inst) {
    int64_t index = inst_index.at(inst);
    CHECK_GE(--inst_deps[index], 0);
    if (inst_deps[index] == 0) {
      ready_queue.push(inst);
    }
  };
  for (HloInstruction* inst : computation->instructions()) {
    size_t index = inst_index.size();
    inst_index[inst] = index;
    inst_deps[index] =
        inst->unique_operands().size() + inst->control_predecessors().size();
    if (inst_deps[index] == 0) {
      ready_queue.push(inst);
    }
  }
  HloInstructionSequence sequence;
  while (!ready_queue.empty()) {
    HloInstruction* inst = ready_queue.front();
    ready_queue.pop();
    for (HloInstruction* user : inst->users()) update_queue(user);
    for (HloInstruction* succ : inst->control_successors()) update_queue(succ);
    sequence.push_back(inst);
  }
  CHECK_EQ(sequence.size(), computation->instruction_count());
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory,
        HeapSimulator::MinimumMemoryForComputation(
            *computation, sequence, alias_analysis, size_function));
  }
  return sequence;
}
ModuleSchedulerAlgorithm ComputationSchedulerToModuleScheduler(
    const MemorySchedulerAlgorithm& computation_scheduler,
    const MemorySchedulerPostprocessor& postprocessor) {
  return [computation_scheduler, postprocessor](
             const HloModule* module,
             const TuplePointsToAnalysis& points_to_analysis,
             const HloAliasAnalysis& alias_analysis,
             const LogicalBuffer::SizeFunction& size_func,
             const absl::flat_hash_set<absl::string_view>& execution_threads,
             int64_t* peak_memory) -> absl::StatusOr<HloSchedule> {
    HloSchedule schedule(module);
    for (auto* computation :
         module->MakeComputationPostOrder(execution_threads)) {
      if (!computation->IsFusionComputation()) {
        TF_ASSIGN_OR_RETURN(HloInstructionSequence computation_sequence,
                            ScheduleComputationHelper(
                                computation, points_to_analysis, alias_analysis,
                                size_func, computation_scheduler, postprocessor,
                                nullptr));
        schedule.set_sequence(computation, std::move(computation_sequence));
      }
    }
    if (peak_memory) {
      TF_ASSIGN_OR_RETURN(*peak_memory, HeapSimulator::MinimumMemoryForModule(
                                            schedule, size_func));
    }
    return std::move(schedule);
  };
}
absl::StatusOr<HloInstructionSequence> ListMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence sequence,
      ListScheduler::Run(computation, points_to_analysis, size_function));
  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory,
        HeapSimulator::MinimumMemoryForComputation(
            *computation, sequence, alias_analysis, size_function));
  }
  return sequence;
}
absl::StatusOr<HloInstructionSequence> PostOrderMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  HloInstructionSequence sequence(computation->MakeInstructionPostOrder());
  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory,
        HeapSimulator::MinimumMemoryForComputation(
            *computation, sequence, alias_analysis, size_function));
  }
  return sequence;
}
absl::StatusOr<HloInstructionSequence> DefaultMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  int64_t list_memory;
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence list_sequence,
      ListMemoryScheduler(computation, points_to_analysis, alias_analysis,
                          size_function, postprocessor, &list_memory));
  VLOG(2) << "Min-memory list sequence: " << HumanReadableNumBytes(list_memory);
  int64_t dfs_memory;
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence dfs_sequence,
      DFSMemoryScheduler(computation, points_to_analysis, alias_analysis,
                         size_function, postprocessor, &dfs_memory));
  VLOG(2) << "Min-memory dfs sequence: " << HumanReadableNumBytes(dfs_memory);
  int64_t post_order_memory;
  TF_ASSIGN_OR_RETURN(HloInstructionSequence post_order_sequence,
                      PostOrderMemoryScheduler(
                          computation, points_to_analysis, alias_analysis,
                          size_function, postprocessor, &post_order_memory));
  VLOG(2) << "Min-memory post order sequence: "
          << HumanReadableNumBytes(post_order_memory);
  auto min_memory = std::min({dfs_memory, post_order_memory, list_memory});
  if (peak_memory) {
    *peak_memory = min_memory;
  }
  if (min_memory == list_memory) {
    VLOG(2) << "Chose min-memory list sequence: "
            << HumanReadableNumBytes(list_memory);
    return list_sequence;
  } else if (min_memory == dfs_memory) {
    VLOG(2) << "Chose min-memory dfs sequence: "
            << HumanReadableNumBytes(dfs_memory);
    return dfs_sequence;
  } else {
    VLOG(2) << "Chose min-memory post_order sequence: "
            << HumanReadableNumBytes(post_order_memory);
    return post_order_sequence;
  }
}
absl::StatusOr<HloSchedule> DefaultModuleScheduler(
    const HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    int64_t* peak_memory) {
  int64_t list_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule list_sequence,
      ComputationSchedulerToModuleScheduler(ListMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &list_memory));
  VLOG(2) << "Min-memory list sequence: " << HumanReadableNumBytes(list_memory);
  int64_t dfs_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule dfs_sequence,
      ComputationSchedulerToModuleScheduler(DFSMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &dfs_memory));
  VLOG(2) << "Min-memory dfs sequence: " << HumanReadableNumBytes(dfs_memory);
  int64_t post_order_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule post_order_sequence,
      ComputationSchedulerToModuleScheduler(PostOrderMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &post_order_memory));
  VLOG(2) << "Min-memory post order sequence: "
          << HumanReadableNumBytes(post_order_memory);
  auto min_memory = std::min({dfs_memory, post_order_memory, list_memory});
  if (peak_memory) {
    *peak_memory = min_memory;
  }
  if (min_memory == list_memory) {
    VLOG(2) << "Chose min-memory list sequence: "
            << HumanReadableNumBytes(list_memory);
    return list_sequence;
  } else if (min_memory == dfs_memory) {
    VLOG(2) << "Chose min-memory dfs sequence: "
            << HumanReadableNumBytes(dfs_memory);
    return dfs_sequence;
  } else {
    VLOG(2) << "Chose min-memory post_order sequence: "
            << HumanReadableNumBytes(post_order_memory);
    return post_order_sequence;
  }
}
absl::StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const BufferValue::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    int64_t* peak_memory) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaMemoryScheduler:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  });
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      (algorithm ? algorithm : DefaultModuleScheduler)(
                          module, *points_to_analysis, *alias_analysis,
                          size_function, execution_threads, peak_memory));
  TF_RETURN_IF_ERROR(schedule.Verify());
  return std::move(schedule);
}
HloMemoryScheduler::HloMemoryScheduler(
    const BufferValue::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm)
    : size_function_(size_function), algorithm_(algorithm) {}
absl::StatusOr<bool> HloMemoryScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(module, size_function_, algorithm_, execution_threads));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  return true;
}
absl::StatusOr<bool> HloTrivialScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloSchedule schedule(module);
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      HloInstructionSequence& computation_sequence =
          schedule.GetOrCreateSequence(computation);
      FunctionVisitor visitor(
          [&computation_sequence](HloInstruction* instruction) {
            computation_sequence.push_back(instruction);
            return absl::OkStatus();
          });
      visitor.ReserveVisitStates(computation->instruction_count());
      TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    }
  }
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  return true;
}
absl::StatusOr<bool> HloDescheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = module->has_schedule();
  module->clear_schedule();
  return changed;
}
}  