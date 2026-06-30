#include "xla/service/memory_space_assignment/memory_bound_loop_optimizer.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace memory_space_assignment {
namespace {
std::optional<int64_t> GetInstructionIndex(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        instructions_to_index) {
  auto it = instructions_to_index.find(instruction);
  return it == instructions_to_index.end() ? std::nullopt
                                           : std::optional<int64_t>(it->second);
}
}  
void LoopOptimizerBestFitHeap::CreateBufferInterval(
    const AllocationBlock& allocation_block,
    const AllocationBlock* colocated_with) {
  buffer_intervals_[&allocation_block] =
      BufferInterval({&allocation_block,
                      allocation_block.size,
                      allocation_block.inclusive_start_time,
                      allocation_block.end_time,
                      {},
                      colocated_with == nullptr});
  if (colocated_with) {
    buffer_intervals_[colocated_with].colocations.push_back(&allocation_block);
  }
}
std::optional<HeapSimulator::Chunk>
LoopOptimizerBestFitHeap::MaybeFindChunkCandidate(
    const AllocationBlock& allocation_block, int64_t preferred_offset) {
  Chunk chunk_candidate = FindChunkCandidate(
      buffer_intervals_[&allocation_block], preferred_offset);
  if (chunk_candidate.chunk_end() <= size_limit_per_heap_) {
    return chunk_candidate;
  }
  return std::nullopt;
}
std::optional<HeapSimulator::Chunk>
LoopOptimizerBestFitHeap::FindAndCommitChunkCandidate(
    const AllocationBlock& allocation_block, int64_t preferred_offset) {
  std::optional<Chunk> chunk =
      MaybeFindChunkCandidate(allocation_block, preferred_offset);
  if (chunk.has_value()) {
    CommitChunk(buffer_intervals_[&allocation_block], chunk.value());
  }
  return chunk;
}
void LoopOptimizerBestFitHeap::RemoveChunk(int64_t start_time, int64_t end_time,
                                           Chunk chunk) {
  CHECK(interval_tree_.Remove(start_time, end_time, chunk));
}
void LoopOptimizerBestFitHeap::RemoveEvenChunks(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop,
    std::optional<HeapSimulator::Chunk>& chunk) {
  RemoveChunk(begin_idx_in_loop, end_idx_in_loop, chunk.value());
  RemoveChunk(begin_idx_in_loop + 2 * loop_size_,
              end_idx_in_loop + 2 * loop_size_, chunk.value());
}
void LoopOptimizerBestFitHeap::RemoveOddChunks(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop,
    std::optional<HeapSimulator::Chunk>& chunk) {
  RemoveChunk(begin_idx_in_loop + loop_size_, end_idx_in_loop + loop_size_,
              chunk.value());
  RemoveChunk(begin_idx_in_loop + 3 * loop_size_,
              end_idx_in_loop + 3 * loop_size_, chunk.value());
}
void LoopOptimizerBestFitHeap::RemoveEvenOddChunkPair(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop,
    EvenOddChunkPair& chunks) {
  CheckAllocationIntervalValid(begin_idx_in_loop, end_idx_in_loop);
  ShiftAllocationIntervalIfRequired(begin_idx_in_loop, end_idx_in_loop);
  auto [even_chunk, odd_chunk] = chunks;
  RemoveEvenChunks(begin_idx_in_loop, end_idx_in_loop, even_chunk);
  RemoveOddChunks(begin_idx_in_loop, end_idx_in_loop, odd_chunk);
}
const AllocationBlock& LoopOptimizerBestFitHeap::GetAllocationBlock(
    int64_t start_time, int64_t end_time, int64_t size) {
  allocation_blocks_.push_back(
      {start_time, end_time, size, static_cast<int64_t>(-1),
       static_cast<int64_t>(-1),
       static_cast<int64_t>(allocation_blocks_.size())});
  return allocation_blocks_.back();
}
const AllocationBlock& LoopOptimizerBestFitHeap::CreateEvenAllocationBlock(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size) {
  const AllocationBlock& first_allocation_block =
      GetAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  CreateBufferInterval(first_allocation_block);
  const AllocationBlock& second_allocation_block =
      GetAllocationBlock(begin_idx_in_loop + 2 * loop_size_,
                         end_idx_in_loop + 2 * loop_size_, size);
  CreateBufferInterval(second_allocation_block, &first_allocation_block);
  return first_allocation_block;
}
const AllocationBlock& LoopOptimizerBestFitHeap::CreateOddAllocationBlock(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size) {
  const AllocationBlock& first_allocation_block = GetAllocationBlock(
      begin_idx_in_loop + loop_size_, end_idx_in_loop + loop_size_, size);
  CreateBufferInterval(first_allocation_block);
  const AllocationBlock& second_allocation_block =
      GetAllocationBlock(begin_idx_in_loop + 3 * loop_size_,
                         end_idx_in_loop + 3 * loop_size_, size);
  CreateBufferInterval(second_allocation_block, &first_allocation_block);
  return first_allocation_block;
}
void LoopOptimizerBestFitHeap::CheckAllocationIntervalValid(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop) const {
  CHECK_LE(begin_idx_in_loop, end_idx_in_loop);
  CHECK_LE(-1 * loop_size_, begin_idx_in_loop);
  CHECK_LT(begin_idx_in_loop, loop_size_);
  CHECK_LE(0, end_idx_in_loop);
  CHECK_LT(end_idx_in_loop, 2 * loop_size_);
  CHECK_LE(end_idx_in_loop - begin_idx_in_loop + 1, 2 * loop_size_);
}
void LoopOptimizerBestFitHeap::ShiftAllocationIntervalIfRequired(
    int64_t& begin_idx_in_loop, int64_t& end_idx_in_loop) const {
  if (begin_idx_in_loop < 0) {
    begin_idx_in_loop += loop_size_;
    end_idx_in_loop += loop_size_;
  }
}
EvenOddChunkPair LoopOptimizerBestFitHeap::FindEvenAndOddAllocationBetween(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
    std::pair<int64_t, int64_t> preferred_offsets) {
  CheckAllocationIntervalValid(begin_idx_in_loop, end_idx_in_loop);
  ShiftAllocationIntervalIfRequired(begin_idx_in_loop, end_idx_in_loop);
  auto [even_offset, odd_offset] = preferred_offsets;
  const AllocationBlock& even_allocation =
      CreateEvenAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  const AllocationBlock& odd_allocation =
      CreateOddAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  std::optional<HeapSimulator::Chunk> even_chunk =
      FindAndCommitChunkCandidate(even_allocation, even_offset);
  if (!even_chunk.has_value()) {
    return {std::nullopt, std::nullopt};
  }
  std::optional<HeapSimulator::Chunk> odd_chunk =
      MaybeFindChunkCandidate(odd_allocation, odd_offset);
  RemoveEvenChunks(begin_idx_in_loop, end_idx_in_loop, even_chunk);
  if (odd_chunk.has_value()) {
    return {even_chunk, odd_chunk};
  }
  return {std::nullopt, std::nullopt};
}
EvenOddChunkPair LoopOptimizerBestFitHeap::AllocateEvenAndOddBetween(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
    std::pair<int64_t, int64_t> preferred_offsets) {
  CheckAllocationIntervalValid(begin_idx_in_loop, end_idx_in_loop);
  ShiftAllocationIntervalIfRequired(begin_idx_in_loop, end_idx_in_loop);
  auto [even_offset, odd_offset] = preferred_offsets;
  const AllocationBlock& even_allocation =
      CreateEvenAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  const AllocationBlock& odd_allocation =
      CreateOddAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  std::optional<HeapSimulator::Chunk> even_chunk =
      FindAndCommitChunkCandidate(even_allocation, even_offset);
  if (!even_chunk.has_value()) {
    return {std::nullopt, std::nullopt};
  }
  std::optional<HeapSimulator::Chunk> odd_chunk =
      FindAndCommitChunkCandidate(odd_allocation, odd_offset);
  if (odd_chunk.has_value()) {
    return {even_chunk, odd_chunk};
  }
  RemoveEvenChunks(begin_idx_in_loop, end_idx_in_loop, even_chunk);
  return {std::nullopt, std::nullopt};
}
const AllocationBlock&
LoopOptimizerBestFitHeap::CreateSameEvenAndOddAllocationBlock(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size) {
  const AllocationBlock& first_allocation_block =
      GetAllocationBlock(begin_idx_in_loop, end_idx_in_loop, size);
  CreateBufferInterval(first_allocation_block);
  const AllocationBlock& second_allocation_block =
      GetAllocationBlock(begin_idx_in_loop + 1 * loop_size_,
                         end_idx_in_loop + 1 * loop_size_, size);
  CreateBufferInterval(second_allocation_block, &first_allocation_block);
  const AllocationBlock& third_allocation_block =
      GetAllocationBlock(begin_idx_in_loop + 2 * loop_size_,
                         end_idx_in_loop + 2 * loop_size_, size);
  CreateBufferInterval(third_allocation_block, &first_allocation_block);
  const AllocationBlock& fourth_allocation_block =
      GetAllocationBlock(begin_idx_in_loop + 3 * loop_size_,
                         end_idx_in_loop + 3 * loop_size_, size);
  CreateBufferInterval(fourth_allocation_block, &first_allocation_block);
  return first_allocation_block;
}
EvenOddChunkPair LoopOptimizerBestFitHeap::FindSameEvenAndOddAllocationBetween(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
    int64_t preferred_offset) {
  CheckAllocationIntervalValid(begin_idx_in_loop, end_idx_in_loop);
  ShiftAllocationIntervalIfRequired(begin_idx_in_loop, end_idx_in_loop);
  CHECK_LE(end_idx_in_loop - begin_idx_in_loop + 1, loop_size_);
  const AllocationBlock& allocation = CreateSameEvenAndOddAllocationBlock(
      begin_idx_in_loop, end_idx_in_loop, size);
  std::optional<HeapSimulator::Chunk> chunk =
      MaybeFindChunkCandidate(allocation, preferred_offset);
  return {chunk, chunk};
}
EvenOddChunkPair LoopOptimizerBestFitHeap::AllocateSameEvenAndOddBetween(
    int64_t begin_idx_in_loop, int64_t end_idx_in_loop, int64_t size,
    int64_t preferred_offset) {
  CheckAllocationIntervalValid(begin_idx_in_loop, end_idx_in_loop);
  ShiftAllocationIntervalIfRequired(begin_idx_in_loop, end_idx_in_loop);
  CHECK_LE(end_idx_in_loop - begin_idx_in_loop + 1, loop_size_);
  const AllocationBlock& allocation = CreateSameEvenAndOddAllocationBlock(
      begin_idx_in_loop, end_idx_in_loop, size);
  std::optional<HeapSimulator::Chunk> chunk =
      FindAndCommitChunkCandidate(allocation, preferred_offset);
  return {chunk, chunk};
}
std::string LoopOptimizerBestFitHeap::MemoryUsageToAsciiArt(
    int64_t begin_iteration, int64_t end_iteration) const {
  CHECK_LE(0, begin_iteration);
  CHECK_LE(begin_iteration, end_iteration);
  return interval_tree_.NodesOverlappingInTimeToAsciiArt(
      loop_size_ * begin_iteration, loop_size_ * (end_iteration + 1) - 1,
      loop_size_);
}
std::vector<int64_t> LoopOptimizerBestFitHeap::RemainingMemoryByTime() const {
  std::vector<int64_t> memory_used_by_time =
      interval_tree_.MemoryUsedInInterval(loop_size_ * 2, loop_size_ * 3 - 1);
  std::vector<int64_t> remaining_memory_by_time(loop_size_);
  for (int i = 0; i < loop_size_; ++i) {
    remaining_memory_by_time[i] = size_limit_per_heap_ - memory_used_by_time[i];
  }
  return remaining_memory_by_time;
}
int64_t LoopOptimizerBestFitHeap::LastMemoryOffsetOccupied() const {
  return interval_tree_.HeapSizeInInterval(loop_size_ * 2, loop_size_ * 4 - 1);
}
 absl::StatusOr<std::unique_ptr<MemoryBoundLoopOptimizer>>
MemoryBoundLoopOptimizer::Create(
    int loop_start, int loop_end, uint64_t alternate_memory_size,
    const MemoryBoundLoopOptimizerOptions& options,
    const HloLiveRange& hlo_live_range, const HloAliasAnalysis& alias_analysis,
    const CostAnalysis& cost_analysis,
    const BufferValue::SizeFunction& size_function,
    const ReservedScopedMemoryFunction& reserved_scoped_memory_fn) {
  std::unique_ptr<MemoryBoundLoopOptimizer> optimizer =
      absl::WrapUnique(new MemoryBoundLoopOptimizer(
          loop_start, loop_end, alternate_memory_size, options, hlo_live_range,
          alias_analysis, cost_analysis, size_function,
          reserved_scoped_memory_fn));
  TF_RETURN_IF_ERROR(optimizer->Initialize());
  return std::move(optimizer);
}
MemoryBoundLoopOptimizer::MemoryBoundLoopOptimizer(
    int loop_start, int loop_end, uint64_t alternate_memory_size,
    const MemoryBoundLoopOptimizerOptions& options,
    const HloLiveRange& hlo_live_range, const HloAliasAnalysis& alias_analysis,
    const CostAnalysis& cost_analysis,
    const BufferValue::SizeFunction& size_function,
    const ReservedScopedMemoryFunction& reserved_scoped_memory_fn)
    : loop_start_(loop_start),
      loop_end_(loop_end),
      loop_size_(loop_end - loop_start),
      alternate_memory_size_(alternate_memory_size),
      options_(options),
      hlo_live_range_(hlo_live_range),
      alias_analysis_(alias_analysis),
      cost_analysis_(cost_analysis),
      size_function_(size_function),
      reserved_scoped_memory_fn_(reserved_scoped_memory_fn) {}
absl::Status MemoryBoundLoopOptimizer::Initialize() {
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  VLOG(3) << "MemoryBoundLoopOptimizer::Initialize, loop start: " << loop_start_
          << ", loop end: " << loop_end_ << ", loop size: " << loop_size_;
  const HloComputation* loop_computation = nullptr;
  int prev_iteration_start = loop_start_ - loop_size_;
  int next_iteration_start = loop_start_ + loop_size_;
  for (int i = 0; i < loop_size_; ++i) {
    const HloInstruction* loop_inst = instruction_sequence[loop_start_ + i];
    instructions_in_loop_[loop_inst] = i;
    const HloInstruction* prev_iteration_inst =
        instruction_sequence[prev_iteration_start + i];
    instructions_in_prev_iteration_[prev_iteration_inst] = i;
    const HloInstruction* next_iteration_inst =
        instruction_sequence[next_iteration_start + i];
    instructions_in_next_iteration_[next_iteration_inst] = i;
    VLOG(3) << "  inst in loop [" << (i) << "]: " << loop_inst->name();
    if (!loop_computation) {
      loop_computation = loop_inst->parent();
    } else {
      TF_RET_CHECK(loop_computation == loop_inst->parent());
    }
    remaining_memory_.push_back(
        alternate_memory_size_ -
        reserved_scoped_memory_fn_(loop_inst,
                                   {},
                                   {}));
  }
  std::set<const HloBuffer*> buffers_to_process;
  for (const auto& [instruction, idx] : instructions_in_loop_) {
    auto maybe_add_buffer = [&](const HloInstruction* instruction) {
      return [this, &buffers_to_process, instruction](const Shape& subshape,
                                                      const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloBuffer& buffer =
            alias_analysis_.GetUniqueBufferAt(instruction, index);
        if (buffers_to_process.find(&buffer) == buffers_to_process.end()) {
          buffers_to_process.insert(&buffer);
        }
      };
    };
    ShapeUtil::ForEachSubshape(instruction->shape(),
                               maybe_add_buffer(instruction));
    for (const HloInstruction* operand : instruction->operands()) {
      ShapeUtil::ForEachSubshape(operand->shape(), maybe_add_buffer(operand));
    }
  }
  for (const HloBuffer* buffer : buffers_to_process) {
    MaybeCreateLoopValue(*buffer, loop_computation);
  }
  return absl::OkStatus();
}
void MemoryBoundLoopOptimizer::MaybeCreateLoopValue(
    const HloBuffer& buffer, const HloComputation* loop_computation) {
  loop_values_.push_back({});
  LoopValue& loop_value = loop_values_.back();
  float pos_bytes = 0;
  float use_bytes = 0;
  bool has_footer_consumer = false;
  for (const HloValue* value : buffer.values()) {
    for (const HloPosition& position : value->positions()) {
      if (position.instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      std::optional<int64_t> loop_index =
          GetInstructionIndex(position.instruction, instructions_in_loop_);
      std::optional<int64_t> prev_iteration_index;
      if (loop_index) {
        loop_value.loop_positions.push_back({*loop_index, position});
        VLOG(3) << "Pos match: " << position.instruction->name() << " at "
                << *loop_index;
      } else if ((prev_iteration_index = GetInstructionIndex(
                      position.instruction, instructions_in_prev_iteration_))) {
        loop_value.prev_iteration_positions.push_back(
            {*prev_iteration_index, position});
        VLOG(3) << "Pos match (prev iteration): "
                << position.instruction->name() << " at "
                << *prev_iteration_index;
      } else if (loop_value.prev_iteration_positions.empty() &&
                 loop_value.loop_positions.empty() &&
                 position.instruction->parent() == loop_computation &&
                 !loop_value.header_position) {
        loop_value.header_position = position;
      }
      if (loop_index || prev_iteration_index) {
        float bytes_accessed = cost_analysis_.base_costs().OutputBytesAccessed(
            *position.instruction, position.index);
        pos_bytes += bytes_accessed;
        VLOG(3) << " accessed: " << bytes_accessed;
      }
    }
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      std::optional<int64_t> loop_index =
          GetInstructionIndex(use.instruction, instructions_in_loop_);
      std::optional<int64_t> next_iteration_index;
      if (loop_index) {
        loop_value.loop_uses.push_back({*loop_index, use});
        VLOG(3) << "Use match: " << use.instruction->name() << " at "
                << *loop_index;
      } else if ((next_iteration_index = GetInstructionIndex(
                      use.instruction, instructions_in_next_iteration_))) {
        loop_value.next_iteration_uses.push_back({*next_iteration_index, use});
        VLOG(3) << "Use match (next iteration): " << use.instruction->name()
                << " at " << *next_iteration_index;
      } else if (!loop_value.loop_positions.empty() ||
                 !loop_value.loop_uses.empty()) {
        has_footer_consumer = true;
      }
      if (loop_index || next_iteration_index) {
        float bytes_accessed = cost_analysis_.base_costs().OperandBytesAccessed(
            *use.instruction, use.operand_number, use.operand_index);
        use_bytes += bytes_accessed;
        VLOG(3) << " accessed: " << bytes_accessed;
      }
    }
  }
  if ((!loop_value.loop_positions.empty() || !loop_value.loop_uses.empty()) &&
      loop_value.prev_iteration_positions.empty()) {
    loop_value.size = size_function_(**buffer.values().begin());
    VLOG(3) << "Size: " << loop_value.size;
    loop_value.allocation_type = LoopValue::AllocationType::kUnsupported;
    auto position_compare = [](const std::pair<int64_t, HloPosition>& a,
                               const std::pair<int64_t, HloPosition>& b) {
      return a.first < b.first;
    };
    auto use_compare = [](const std::pair<int64_t, HloUse>& a,
                          const std::pair<int64_t, HloUse>& b) {
      return a.first < b.first;
    };
    absl::c_sort(loop_value.loop_positions, position_compare);
    absl::c_sort(loop_value.prev_iteration_positions, position_compare);
    absl::c_sort(loop_value.loop_uses, use_compare);
    absl::c_sort(loop_value.next_iteration_uses, use_compare);
    if (!loop_value.loop_positions.empty()) {
      if (loop_value.next_iteration_uses.empty() &&
          !loop_value.loop_uses.empty()) {
        loop_value.allocation_type = LoopValue::AllocationType::kTemporary;
      } else if (!loop_value.next_iteration_uses.empty()) {
        if (loop_value.next_iteration_uses.back().first >=
            loop_value.loop_positions.front().first) {
          loop_value.allocation_type =
              LoopValue::AllocationType::kLoopCarriedDependence;
        } else {
          loop_value.allocation_type = LoopValue::AllocationType::kTemporary;
        }
      }
    } else if (loop_value.header_position && !loop_value.loop_uses.empty()) {
      if (loop_value.loop_uses.size() ==
              loop_value.next_iteration_uses.size() &&
          loop_value.loop_uses.front().first ==
              loop_value.next_iteration_uses.front().first) {
        loop_value.allocation_type = LoopValue::AllocationType::kPinned;
      } else if (loop_value.next_iteration_uses.empty() ||
                 loop_value.next_iteration_uses.back().first <
                     loop_value.loop_uses.front().first) {
        loop_value.allocation_type = LoopValue::AllocationType::kPrefetch;
      }
    }
    VLOG(3) << "Allocation type "
            << LoopValue::AllocationTypeToString(loop_value.allocation_type);
    VLOG(3) << "Pos bytes: " << pos_bytes << " use bytes: " << use_bytes;
    float savings = pos_bytes + use_bytes;
    if (loop_value.header_position) {
      savings -= loop_value.size;
    }
    if (!loop_value.loop_positions.empty() && has_footer_consumer) {
      savings -= loop_value.size;
    }
    loop_value.savings = savings;
    loop_value.savings_per_byte = savings / loop_value.size;
    VLOG(3) << "Savings: " << loop_value.savings;
    VLOG(3) << "Savings per byte: " << loop_value.savings_per_byte;
    for (const HloValue* value : buffer.values()) {
      VLOG(3) << value->ToString();
    }
    loop_value.hlo_values = buffer.values();
  } else {
    loop_values_.pop_back();
  }
}
void MemoryBoundLoopOptimizer::Optimize() {
  SortLoopValues();
  AllocateLoopValues();
  PostProcess();
}
float MemoryBoundLoopOptimizer::CalculateExecutionTime() const {
  std::vector<std::pair<const CopyAllocation*, float>> prefetches;
  for (const LoopValue& value : loop_values_) {
    if (!value.allocations.empty() &&
        value.allocations.back()->is_copy_allocation()) {
      prefetches.push_back(
          {static_cast<const CopyAllocation*>(value.allocations.back().get()),
           cost_analysis_.GetAsyncCopyElapsed(
               value.hlo_values.front()->shape())});
    }
  }
  auto get_effective_done_time =
      [&](int64_t copy_start_schedule_after,
          int64_t copy_done_schedule_before) -> int64_t {
    if (copy_start_schedule_after == loop_size_ - 1 &&
        copy_done_schedule_before == 0) {
      return 2 * loop_size_;
    }
    if (copy_start_schedule_after + 1 >= copy_done_schedule_before) {
      return copy_done_schedule_before + loop_size_;
    }
    return copy_done_schedule_before;
  };
  absl::c_sort(
      prefetches, [&](const std::pair<const CopyAllocation*, float>& a,
                      const std::pair<const CopyAllocation*, float>& b) {
        return std::forward_as_tuple(
                   a.first->copy_start_schedule_after(),
                   get_effective_done_time(
                       a.first->copy_start_schedule_after(),
                       a.first->copy_done_schedule_before())) <
               std::forward_as_tuple(b.first->copy_start_schedule_after(),
                                     get_effective_done_time(
                                         b.first->copy_start_schedule_after(),
                                         b.first->copy_done_schedule_before()));
      });
  std::vector<std::optional<int>> required_prefetch_completions(loop_size_);
  for (int i = 0; i < prefetches.size(); ++i) {
    const auto& [prefetch, elapsed] = prefetches[i];
    int required_prefetch_completion = i;
    if (prefetch->copy_start_schedule_after() == loop_size_ - 1 &&
        prefetch->copy_done_schedule_before() == 0) {
      required_prefetch_completion -= 2 * prefetches.size();
    } else if (prefetch->copy_start_schedule_after() + 1 >=
               prefetch->copy_done_schedule_before()) {
      required_prefetch_completion -= prefetches.size();
    }
    VLOG(3) << "Prefetch #" << i << " (elapsed " << elapsed
            << "): " << prefetch->ToString();
    if (required_prefetch_completions[prefetch->copy_done_schedule_before()]) {
      required_prefetch_completions[prefetch->copy_done_schedule_before()] =
          std::max(
              *required_prefetch_completions[prefetch
                                                 ->copy_done_schedule_before()],
              required_prefetch_completion);
    } else {
      required_prefetch_completions[prefetch->copy_done_schedule_before()] =
          required_prefetch_completion;
    }
    VLOG(4)
        << "Required completion at " << prefetch->copy_done_schedule_before()
        << " = "
        << *required_prefetch_completions[prefetch
                                              ->copy_done_schedule_before()];
  }
  float result;
  std::vector<float> bandwidth_idle_times;
  std::vector<float> instructions_elapsed;
  bandwidth_idle_times.reserve(loop_size_);
  instructions_elapsed.reserve(loop_size_);
  for (int i = 0; i < loop_size_; ++i) {
    bandwidth_idle_times.push_back(GetBandwidthIdleTime(i));
    instructions_elapsed.push_back(GetInstructionElapsed(i));
  }
  const int kNumIterations = 3;
  std::vector<float> prefetch_remaining_elapsed_times(prefetches.size() *
                                                      kNumIterations);
  int prefetch_start_index = 0;
  int prefetch_done_index = 0;
  int prefetch_completed_index = 0;
  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    float total_elapsed = 0;
    float total_bandwidth_idle_time = 0;
    float total_critical_prefetch = 0;
    for (int i = 0; i < loop_size_; ++i) {
      std::optional<int> required_prefetch_completion =
          required_prefetch_completions[i];
      if (required_prefetch_completion) {
        int required_prefetch_done_index =
            iteration * static_cast<int>(prefetches.size()) +
            *required_prefetch_completion;
        VLOG(4) << "Prefetch #"
                << ((*required_prefetch_completion + prefetches.size()) %
                    prefetches.size())
                << " (" << required_prefetch_done_index
                << ") is required to be completed at " << i;
        for (; prefetch_done_index <= required_prefetch_done_index;
             ++prefetch_done_index) {
          CHECK_LE(prefetch_done_index, prefetch_start_index);
          if (prefetch_done_index == prefetch_completed_index) {
            float& prefetch_remaining =
                prefetch_remaining_elapsed_times[prefetch_done_index];
            VLOG(4) << "Prefetch #" << (prefetch_done_index % prefetches.size())
                    << " (" << prefetch_done_index
                    << ") did not complete, remaining elapsed = "
                    << prefetch_remaining;
            total_critical_prefetch += prefetch_remaining;
            prefetch_remaining = 0;
            ++prefetch_completed_index;
          }
        }
      }
      float elapsed = instructions_elapsed[i];
      total_elapsed += elapsed;
      float bandwidth_idle_time = bandwidth_idle_times[i];
      for (; prefetch_completed_index < prefetch_start_index;
           ++prefetch_completed_index) {
        float& prefetch_remaining =
            prefetch_remaining_elapsed_times[prefetch_completed_index];
        if (bandwidth_idle_time < prefetch_remaining) {
          prefetch_remaining -= bandwidth_idle_time;
          bandwidth_idle_time = 0;
          VLOG(4) << "Prefetch #"
                  << (prefetch_completed_index % prefetches.size()) << " ("
                  << prefetch_completed_index << ") still ongoing at " << i
                  << ", remaining elapsed = " << prefetch_remaining;
          break;
        }
        bandwidth_idle_time -= prefetch_remaining;
        prefetch_remaining = 0;
        VLOG(4) << "Prefetch #"
                << (prefetch_completed_index % prefetches.size()) << " ("
                << prefetch_completed_index << ") completed at " << i
                << ", bandwidth idle time = " << bandwidth_idle_time;
      }
      if (bandwidth_idle_time > 0) {
        VLOG(4) << "Bandwidth idle time at " << i << " = "
                << bandwidth_idle_time;
        total_bandwidth_idle_time += bandwidth_idle_time;
      }
      for (; prefetch_start_index < (iteration + 1) * prefetches.size() &&
             prefetches[prefetch_start_index % prefetches.size()]
                     .first->copy_start_schedule_after() == i;
           ++prefetch_start_index) {
        float& prefetch_remaining =
            prefetch_remaining_elapsed_times[prefetch_start_index];
        prefetch_remaining =
            prefetches[prefetch_start_index % prefetches.size()].second;
        VLOG(4) << "Prefetch #" << (prefetch_start_index % prefetches.size())
                << " (" << prefetch_start_index << ") started at " << i
                << ", remaining elapsed = " << prefetch_remaining;
      }
    }
    VLOG(3) << "Iteration " << iteration;
    VLOG(3) << "Total elapsed: " << total_elapsed
            << ", total critical prefetch: " << total_critical_prefetch
            << ", total bandwidth idle time: " << total_bandwidth_idle_time;
    result = total_elapsed + total_critical_prefetch;
  }
  return result;
}
 std::string
MemoryBoundLoopOptimizer::LoopValue::AllocationTypeToString(
    LoopValue::AllocationType allocation_type) {
  switch (allocation_type) {
    case AllocationType::kTemporary:
      return "temporary";
    case AllocationType::kLoopCarriedDependence:
      return "loop-carried dependence";
    case AllocationType::kPinned:
      return "pinned";
    case AllocationType::kPrefetch:
      return "prefetch";
    default:
      CHECK(allocation_type == AllocationType::kUnsupported);
      return "unsupported";
  }
}
std::string MemoryBoundLoopOptimizer::LoopValue::ToString() const {
  std::string values_str;
  absl::StrAppend(&values_str, "Values:");
  for (const HloValue* hlo_value : hlo_values) {
    absl::StrAppend(&values_str, "\n  - ", hlo_value->ToShortString());
  }
  std::string allocations_str;
  if (!allocations.empty()) {
    absl::StrAppend(&allocations_str, "Allocations:");
  }
  for (const auto& allocation : allocations) {
    absl::StrAppend(&allocations_str, "\n  - ", allocation->ToString());
  }
  return absl::StrCat(
      "Size: ", size, " savings: ", savings,
      " savings per byte: ", savings_per_byte,
      " allocation type: ", AllocationTypeToString(allocation_type), "\n",
      values_str, "\n", allocations_str);
}
bool MemoryBoundLoopOptimizer::LoopValue::IsAllocationTypeSupported() const {
  return allocation_type == AllocationType::kTemporary ||
         allocation_type == AllocationType::kPinned ||
         allocation_type == AllocationType::kPrefetch;
}
void MemoryBoundLoopOptimizer::SortLoopValues() {
  absl::c_stable_sort(loop_values_, [](const LoopValue& a, const LoopValue& b) {
    return a.savings_per_byte > b.savings_per_byte;
  });
}
void MemoryBoundLoopOptimizer::AllocateLoopValues() {
  std::vector<LoopValue*> prefetch_values;
  VLOG(3) << "Pre optimization execution time: " << CalculateExecutionTime();
  for (LoopValue& value : loop_values_) {
    switch (value.allocation_type) {
      case LoopValue::AllocationType::kTemporary:
        AllocateTemporary(value);
        break;
      case LoopValue::AllocationType::kPinned:
        if (value.savings > 0) {
          AllocatePinned(value);
        }
        break;
      case LoopValue::AllocationType::kPrefetch:
        prefetch_values.push_back(&value);
        break;
      case LoopValue::AllocationType::kLoopCarriedDependence:
      case LoopValue::AllocationType::kUnsupported:
        VLOG(1) << "Unsupported allocation: " << value.ToString();
    }
  }
  VLOG(3) << "Execution time after allocating temporaries: "
          << CalculateExecutionTime();
  AllocatePrefetches(absl::MakeSpan(prefetch_values));
  VLOG(3) << "Execution time after allocating prefetches:  "
          << CalculateExecutionTime();
}
void MemoryBoundLoopOptimizer::PostProcess() {
  for (LoopValue& value : loop_values_) {
    absl::flat_hash_set<HloUse> allocated_uses;
    for (const auto& allocation : value.allocations) {
      for (const HloUse& use : allocation->uses()) {
        allocated_uses.insert(use);
      }
    }
    std::vector<HloUse> unallocated_uses;
    absl::flat_hash_set<int> use_indices;
    for (const auto& [idx, use] : value.loop_uses) {
      use_indices.insert(idx);
      if (!allocated_uses.contains(use)) {
        unallocated_uses.push_back(use);
      }
    }
    for (const auto& [next_iteration_idx, use] : value.next_iteration_uses) {
      if (use_indices.contains(next_iteration_idx)) {
        continue;
      }
      HloInstruction* loop_instruction =
          hlo_live_range_.flattened_instruction_sequence().instructions().at(
              loop_start_ + next_iteration_idx);
      HloUse loop_use{loop_instruction, use.operand_number, use.operand_index};
      if (!allocated_uses.contains(loop_use)) {
        unallocated_uses.push_back(loop_use);
      }
    }
    if (!unallocated_uses.empty()) {
      value.allocations.push_back(std::make_unique<PinnedAllocation>(
          value.hlo_values.front()->defining_position(), MemorySpace::kDefault,
          std::nullopt, 0, loop_size_, false));
      for (const HloUse& use : unallocated_uses) {
        value.allocations.back()->AddUse(use);
      }
    }
  }
}
bool MemoryBoundLoopOptimizer::AllocateBetween(int64_t begin_idx,
                                               int64_t end_idx, int64_t size) {
  int64_t end_idx_sentinel = end_idx;
  if (end_idx < begin_idx) {
    end_idx_sentinel += loop_size_;
  }
  for (int64_t i = begin_idx; i <= end_idx_sentinel; ++i) {
    if (remaining_memory_[i % loop_size_] < size) {
      return false;
    }
  }
  for (int64_t i = begin_idx; i <= end_idx_sentinel; ++i) {
    remaining_memory_[i % loop_size_] -= size;
  }
  return true;
}
bool MemoryBoundLoopOptimizer::AllocateTemporary(LoopValue& value) {
  VLOG(3) << "AllocateTemporary: " << value.ToString();
  if (value.hlo_values.size() > 1) {
    VLOG(3) << "LoopValue has more than one hlo value associated.";
    return false;
  }
  int64_t definition_idx = value.loop_positions.front().first;
  int64_t max_use_idx;
  if (!value.next_iteration_uses.empty()) {
    max_use_idx = value.next_iteration_uses.back().first;
    CHECK_LT(max_use_idx, definition_idx);
  } else {
    max_use_idx = value.loop_uses.back().first;
  }
  bool success = AllocateBetween(definition_idx, max_use_idx, value.size);
  if (success) {
    VLOG(3) << "Pos: " << value.loop_positions[0].second;
    value.allocations.push_back(std::make_unique<PinnedAllocation>(
        value.loop_positions[0].second, MemorySpace::kAlternate, std::nullopt,
        definition_idx, max_use_idx,
        false));
    AddAllLoopPositionsAndUses(value, true);
  }
  return success;
}
bool MemoryBoundLoopOptimizer::AllocatePinned(LoopValue& value) {
  bool success = AllocateBetween(0, loop_size_ - 1, value.size);
  if (success) {
    CHECK(value.header_position);
    value.allocations.push_back(std::make_unique<PinnedAllocation>(
        *value.header_position, MemorySpace::kAlternate, std::nullopt, 0,
        loop_size_,
        false));
    AddAllLoopPositionsAndUses(value, false);
  }
  return success;
}
bool MemoryBoundLoopOptimizer::AllocatePrefetches(
    absl::Span<LoopValue*> values) {
  VLOG(3) << "Allocating prefetches num values: " << values.size();
  AllocatePrefetchesContext context;
  context.values = values;
  context.value_indices.resize(values.size());
  absl::c_iota(context.value_indices, 0);
  absl::c_stable_sort(context.value_indices, [&](int a, int b) {
    return std::forward_as_tuple(
               values[a]->loop_uses.begin()->first,
               values[a]->loop_uses.begin()->second.operand_number) >
           std::forward_as_tuple(
               values[b]->loop_uses.begin()->first,
               values[b]->loop_uses.begin()->second.operand_number);
  });
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      additional_uses_in_alternate_mem;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      additional_positions_in_alternate_mem;
  for (const LoopValue* value : values) {
    VLOG(3) << "  prefetch value: " << value->ToString();
    for (const auto& [idx, use] : value->loop_uses) {
      additional_uses_in_alternate_mem[use.instruction].push_back(
          {use.operand_number, use.operand_index});
    }
    for (const auto& [idx, position] : value->loop_positions) {
      additional_positions_in_alternate_mem[position.instruction].push_back(
          position.index);
    }
  }
  for (int i = 0; i < loop_size_; ++i) {
    context.bandwidth_idle_times.push_back(
        GetBandwidthIdleTime(i, additional_uses_in_alternate_mem,
                             additional_positions_in_alternate_mem));
    VLOG(3) << "Remaining bandwidth at " << i << " = "
            << *context.bandwidth_idle_times.rbegin();
  }
  context.additional_memory_used.resize(loop_size_, 0);
  for (int value_index : context.value_indices) {
    AllocatePrefetch(value_index, context);
  }
  for (int i = 0; i < loop_size_; ++i) {
    remaining_memory_[i] -= context.additional_memory_used[i];
    VLOG(3) << "Additional memory [" << i
            << "]: " << context.additional_memory_used[i];
    VLOG(3) << "Remaining memory [" << i << "]: " << remaining_memory_[i];
    VLOG(3) << "Remaining bandwidth [" << i
            << "] : " << context.bandwidth_idle_times[i];
  }
  return true;
}
bool MemoryBoundLoopOptimizer::AllocatePrefetch(
    int value_index, AllocatePrefetchesContext& context) {
  LoopValue* value = context.values.at(value_index);
  VLOG(3) << "Allocating value: " << value->ToString();
  int first_use_idx = value->loop_uses.front().first;
  int last_use_idx = value->loop_uses.back().first;
  int last_use_idx_sentinel = last_use_idx;
  if (!value->next_iteration_uses.empty()) {
    last_use_idx = value->next_iteration_uses.back().first;
    last_use_idx_sentinel = last_use_idx + loop_size_;
    CHECK_LT(last_use_idx, first_use_idx);
  }
  bool out_of_memory = false;
  for (int i = first_use_idx; i <= last_use_idx_sentinel; ++i) {
    int loop_idx = i % loop_size_;
    if (context.additional_memory_used[loop_idx] + value->size >
        remaining_memory_[loop_idx]) {
      VLOG(3) << "Ran out of memory allocating for uses.";
      out_of_memory = true;
    }
  }
  if (out_of_memory) {
    return false;
  }
  float copy_resource =
      cost_analysis_.GetAsyncCopyElapsed(value->hlo_values.front()->shape());
  VLOG(3) << "First use: " << value->loop_uses.begin()->second
          << " use idx: " << first_use_idx
          << " copy resource: " << copy_resource;
  std::optional<int> copy_start_time;
  float accumulated_copy_resource = 0;
  std::vector<int> early_forced_prefetch_value_indices;
  int early_forced_prefetch_value_search_index = 0;
  float early_forced_prefetch_additional_memory = 0;
  for (int i = first_use_idx - 1; i >= last_use_idx_sentinel - loop_size_;
       --i) {
    int loop_idx = (i + loop_size_) % loop_size_;
    if (i < 0) {
      for (; context.value_indices[early_forced_prefetch_value_search_index] !=
             value_index;
           ++early_forced_prefetch_value_search_index) {
        VLOG(3) << "Searching for early forced: "
                << early_forced_prefetch_value_search_index;
        LoopValue* early_forced_value = context.values.at(
            context.value_indices[early_forced_prefetch_value_search_index]);
        if (early_forced_value->allocations.empty()) {
          continue;
        }
        const CopyAllocation* early_forced_prefetch =
            static_cast<const CopyAllocation*>(
                early_forced_value->allocations.back().get());
        VLOG(3) << "Prefetch: " << early_forced_prefetch->ToString();
        if (early_forced_prefetch->copy_done_schedule_before() <=
                early_forced_prefetch->copy_start_schedule_after() + 1 ||
            (early_forced_prefetch->copy_start_schedule_after() ==
                 loop_size_ - 1 &&
             early_forced_prefetch->copy_done_schedule_before() == 0)) {
          break;
        }
        if (early_forced_prefetch->copy_start_schedule_after() != loop_idx) {
          break;
        }
        early_forced_prefetch_value_indices.push_back(
            early_forced_prefetch_value_search_index);
        early_forced_prefetch_additional_memory += early_forced_value->size;
        VLOG(3) << "Found early-forced prefetch value: "
                << early_forced_value->ToString();
        VLOG(3) << "Early forced prefetch additional memory: "
                << early_forced_prefetch_additional_memory;
      }
    }
    int64_t overlap_memory_overhead = 0;
    if (loop_idx == last_use_idx) {
      overlap_memory_overhead = value->size;
      VLOG(3) << "Loop idx == last use idx (" << loop_idx
              << "), overlap memory overhead = " << overlap_memory_overhead;
    }
    if (context.additional_memory_used[loop_idx] + value->size +
            overlap_memory_overhead + early_forced_prefetch_additional_memory >
        remaining_memory_[loop_idx]) {
      VLOG(3) << "Ran out of memory. Accumulated copy resource "
              << accumulated_copy_resource << " out of " << copy_resource
              << " at " << loop_idx;
      break;
    }
    float bandwidth_idle_time = context.bandwidth_idle_times[loop_idx];
    VLOG(3) << "Idx " << loop_idx
            << " bandwidth_idle_time: " << bandwidth_idle_time
            << " copy resource remaining: "
            << (copy_resource - accumulated_copy_resource) << " diff: "
            << (bandwidth_idle_time -
                (copy_resource - accumulated_copy_resource));
    if (bandwidth_idle_time >= copy_resource - accumulated_copy_resource) {
      accumulated_copy_resource = copy_resource;
      copy_start_time = loop_idx;
      VLOG(3) << "Found the complete copy ratio and updated accumulated copy "
                 "resource: "
              << accumulated_copy_resource;
      break;
    } else if (!copy_start_time &&
               accumulated_copy_resource + bandwidth_idle_time >=
                   copy_resource * options_.desired_copy_ratio()) {
      accumulated_copy_resource += bandwidth_idle_time;
      copy_start_time = loop_idx;
      VLOG(3) << "Found the desired copy ratio and updated accumulated copy "
                 "resource: "
              << accumulated_copy_resource;
    } else if (options_.allow_unsatisfied_fully_pipelined_prefetch() &&
               loop_idx == last_use_idx) {
      accumulated_copy_resource += bandwidth_idle_time;
      copy_start_time = loop_idx;
      VLOG(3) << "Could not reach the desired copy ratio but scheduling "
                 "fully pipelined prefetch anyway: "
              << accumulated_copy_resource;
      break;
    } else {
      accumulated_copy_resource += bandwidth_idle_time;
      VLOG(3) << "Updated accumulated copy resource: "
              << accumulated_copy_resource;
    }
  }
  if (!copy_start_time) {
    return false;
  }
  VLOG(3) << "Success: copy_start_time: " << *copy_start_time
          << " leftover copy resource: "
          << (copy_resource - accumulated_copy_resource);
  auto update_additional_memory_used = [&](int loop_idx, int64_t addition) {
    VLOG(4) << "Updating additional memory used at " << loop_idx << ". "
            << context.additional_memory_used[loop_idx] << " + " << addition
            << " => " << (context.additional_memory_used[loop_idx] + addition)
            << " (remaining: " << remaining_memory_[loop_idx] << ")";
    context.additional_memory_used[loop_idx] += addition;
    CHECK_LE(context.additional_memory_used[loop_idx],
             remaining_memory_[loop_idx]);
  };
  for (int i = first_use_idx; i <= last_use_idx_sentinel; ++i) {
    int loop_idx = i % loop_size_;
    update_additional_memory_used(loop_idx, value->size);
  }
  accumulated_copy_resource = 0.0;
  for (int i = first_use_idx - 1; i >= last_use_idx_sentinel - loop_size_;
       --i) {
    int loop_idx = (i + loop_size_) % loop_size_;
    float& bandwidth_idle_time = context.bandwidth_idle_times[loop_idx];
    int64_t overlap_memory_overhead = 0;
    update_additional_memory_used(loop_idx,
                                  value->size + overlap_memory_overhead);
    if (bandwidth_idle_time < copy_resource - accumulated_copy_resource) {
      accumulated_copy_resource += bandwidth_idle_time;
      bandwidth_idle_time = 0;
      if (loop_idx == *copy_start_time) {
        VLOG(3) << "Remaining copy resource: "
                << (copy_resource - accumulated_copy_resource);
        break;
      }
    } else {
      bandwidth_idle_time -= copy_resource - accumulated_copy_resource;
      CHECK_EQ(loop_idx, *copy_start_time);
      break;
    }
  }
  CHECK(value->header_position);
  value->allocations.push_back(std::make_unique<PinnedAllocation>(
      *value->header_position, MemorySpace::kDefault, std::nullopt, 0,
      loop_size_, false));
  value->allocations.push_back(std::make_unique<CopyAllocation>(
      *value->allocations.back(), MemorySpace::kAlternate, std::nullopt,
      ((*copy_start_time - 1) + loop_size_) % loop_size_, first_use_idx,
      last_use_idx_sentinel));
  AddAllLoopPositionsAndUses(*value, true);
  for (int early_forced_prefetch_value_index :
       early_forced_prefetch_value_indices) {
    LoopValue* early_forced_value = context.values.at(
        context.value_indices[early_forced_prefetch_value_index]);
    CHECK(!early_forced_value->allocations.empty());
    CopyAllocation* early_forced_prefetch = static_cast<CopyAllocation*>(
        early_forced_value->allocations.back().get());
    for (int index = early_forced_prefetch->copy_start_schedule_after();
         index >= *copy_start_time; --index) {
      update_additional_memory_used(index, early_forced_value->size);
      VLOG(3) << "Additional memory used: " << index << " "
              << context.additional_memory_used[index];
    }
    early_forced_prefetch->set_copy_start_schedule_after(
        ((*copy_start_time - 1) + loop_size_) % loop_size_);
    VLOG(3) << "Updated prefetch: " << early_forced_prefetch->ToString();
  }
  return true;
}
void MemoryBoundLoopOptimizer::AddAllLoopPositionsAndUses(
    LoopValue& value, bool allocate_next_iteration_uses) {
  CHECK_GE(value.allocations.size(), 1);
  Allocation& allocation = *value.allocations.back();
  for (const auto& [idx, position] : value.loop_positions) {
    positions_in_alternate_mem_[position.instruction].push_back(position.index);
  }
  for (const auto& [idx, use] : value.loop_uses) {
    uses_in_alternate_mem_[use.instruction].push_back(
        {use.operand_number, use.operand_index});
    allocation.AddUse(use);
  }
  if (allocate_next_iteration_uses) {
    for (const auto& [next_iteration_idx, use] : value.next_iteration_uses) {
      HloInstruction* loop_instruction =
          hlo_live_range_.flattened_instruction_sequence().instructions().at(
              loop_start_ + next_iteration_idx);
      uses_in_alternate_mem_[loop_instruction].push_back(
          {use.operand_number, use.operand_index});
      allocation.AddUse(
          {loop_instruction, use.operand_number, use.operand_index});
    }
  }
}
float MemoryBoundLoopOptimizer::GetBandwidthIdleTime(int idx) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> empty_operands;
  std::vector<ShapeIndex> empty_outputs;
  const std::vector<std::pair<int64_t, ShapeIndex>>* operands_in_alternate_mem =
      &empty_operands;
  const std::vector<ShapeIndex>* outputs_in_alternate_mem = &empty_outputs;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = &uses_it->second;
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = &positions_it->second;
  }
  return cost_analysis_.GetDefaultMemoryBandwidthIdleTime(
      *inst, *operands_in_alternate_mem, *outputs_in_alternate_mem);
}
float MemoryBoundLoopOptimizer::GetBandwidthIdleTime(
    int idx,
    const absl::flat_hash_map<const HloInstruction*,
                              std::vector<std::pair<int64_t, ShapeIndex>>>&
        additional_uses_in_alternate_mem,
    const absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>&
        additional_positions_in_alternate_mem) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> operands_in_alternate_mem;
  std::vector<ShapeIndex> outputs_in_alternate_mem;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = uses_it->second;
  }
  auto additional_uses_it = additional_uses_in_alternate_mem.find(inst);
  if (additional_uses_it != additional_uses_in_alternate_mem.end()) {
    absl::c_copy(additional_uses_it->second,
                 std::back_inserter(operands_in_alternate_mem));
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = positions_it->second;
  }
  auto additional_positions_it =
      additional_positions_in_alternate_mem.find(inst);
  if (additional_positions_it != additional_positions_in_alternate_mem.end()) {
    absl::c_copy(additional_positions_it->second,
                 std::back_inserter(outputs_in_alternate_mem));
  }
  return cost_analysis_.GetDefaultMemoryBandwidthIdleTime(
      *inst, operands_in_alternate_mem, outputs_in_alternate_mem);
}
float MemoryBoundLoopOptimizer::GetInstructionElapsed(int idx) const {
  const HloInstruction* inst =
      hlo_live_range_.flattened_instruction_sequence().instructions().at(
          loop_start_ + idx);
  std::vector<std::pair<int64_t, ShapeIndex>> empty_operands;
  std::vector<ShapeIndex> empty_outputs;
  const std::vector<std::pair<int64_t, ShapeIndex>>* operands_in_alternate_mem =
      &empty_operands;
  const std::vector<ShapeIndex>* outputs_in_alternate_mem = &empty_outputs;
  auto uses_it = uses_in_alternate_mem_.find(inst);
  if (uses_it != uses_in_alternate_mem_.end()) {
    operands_in_alternate_mem = &uses_it->second;
  }
  auto positions_it = positions_in_alternate_mem_.find(inst);
  if (positions_it != positions_in_alternate_mem_.end()) {
    outputs_in_alternate_mem = &positions_it->second;
  }
  return cost_analysis_.GetInstructionElapsedInAlternateMemory(
      *inst, *operands_in_alternate_mem, *outputs_in_alternate_mem);
}
}  
}  