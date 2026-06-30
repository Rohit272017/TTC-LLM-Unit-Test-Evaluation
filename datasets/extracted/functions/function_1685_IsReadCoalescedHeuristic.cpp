#include "xla/service/gpu/model/coalescing_analysis.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stack>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
namespace xla {
namespace gpu {
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const HloInstruction* producer,
                              const HloInstruction* consumer) {
  if (fusion_kind != HloFusionAnalysis::EmitterFusionKind::kTranspose) {
    auto is_broadcast = [&](const HloInstruction* instr) {
      while (true) {
        if (instr->opcode() == HloOpcode::kBroadcast ||
            instr->opcode() == HloOpcode::kIota) {
          return true;
        }
        if (instr->operand_count() != 1) return false;
        if (instr->opcode() != HloOpcode::kBitcast && !instr->IsElementwise()) {
          return false;
        }
        instr = instr->operand(0);
      }
    };
    auto is_bad_transpose = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kFusion) {
        for (auto* instr : instr->fused_instructions()) {
          if (TransposesMinorDimension(instr) &&
              !is_broadcast(instr->operand(0))) {
            return true;
          }
        }
        return false;
      }
      return TransposesMinorDimension(instr) &&
             !is_broadcast(instr->operand(0));
    };
    if (is_bad_transpose(producer)) return false;
    if (consumer && is_bad_transpose(consumer)) return false;
  }
  if (fusion_kind == HloFusionAnalysis::EmitterFusionKind::kReduction &&
      IsInputFusibleReduction(*producer) && consumer &&
      IsInputFusibleReduction(*consumer)) {
    return false;
  }
  return true;
}
bool IsTiledReadCoalescedHeuristic(const TiledHloInstruction& operand,
                                   const se::DeviceDescription& device_info) {
  const Shape& shape = operand.hlo()->shape();
  int64_t contiguous_read_elements = 1;
  for (const auto dim_idx : shape.layout().minor_to_major()) {
    if (operand.tile_stride(dim_idx) != 1) {
      break;
    }
    int64_t tile_size = operand.tile_size(dim_idx);
    int64_t dim_size = shape.dimensions(dim_idx);
    contiguous_read_elements *= std::min(tile_size, dim_size);
    if (tile_size < dim_size) {
      break;
    }
  }
  int64_t contiguous_bytes_accessed =
      contiguous_read_elements *
      ShapeUtil::ByteSizeOfPrimitiveType(operand.hlo()->shape().element_type());
  return contiguous_bytes_accessed >=
         device_info.dram_to_l2_transaction_size_bytes();
}
namespace {
using ::mlir::AffineBinaryOpExpr;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::getAffineConstantExpr;
using ::mlir::MLIRContext;
bool EstimateCoalescingViaMemoryTransactionsCount(
    absl::Span<const Interval> intervals, PrimitiveType element_type) {
  constexpr int64_t kBytesPerMemoryTransaction = 128;
  int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  int memory_transactions = 0;
  int total_num_elements = 0;
  for (const auto& range : intervals) {
    int64_t num_elements = range.upper - range.lower + 1;
    memory_transactions += llvm::divideCeilSigned(num_elements * type_size,
                                                  kBytesPerMemoryTransaction);
    total_num_elements += num_elements;
  }
  if (memory_transactions == 0) {
    return true;
  }
  int memory_transactions_lower_bound = llvm::divideCeilSigned(
      total_num_elements * type_size, kBytesPerMemoryTransaction);
  constexpr float kIsCoalescedThreshold = 0.9;
  return memory_transactions_lower_bound >
         memory_transactions * kIsCoalescedThreshold;
}
Shape GetLinearizedShape(const Shape& shape) {
  if (shape.rank() == 0) {
    return shape;
  }
  std::vector<int64_t> dims{ShapeUtil::ElementsIn(shape)};
  auto result = Shape(shape.element_type(), dims,
                      absl::InlinedVector<bool, 4>(dims.size(), false), {});
  *result.mutable_layout() = xla::Layout({0});
  return result;
}
std::optional<GroupedByOpIndexingMap> GetThreadIdToInputMemoryLayoutsMaps(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context) {
  GroupedByOpIndexingMap result;
  for (const auto& [root_index, hero] :
       llvm::enumerate(fusion_analysis.fusion_heroes())) {
    for (const auto& [hero_operand_index, hero_operand] :
         llvm::enumerate(hero.GetOperands())) {
      if (hero_operand.shape().rank() == 0) {
        continue;
      }
      std::optional<IndexingMap> thread_id_to_hero_operand_map =
          fusion_interface->ComputeThreadIdToInputIndexing(
              root_index, hero_operand_index, mlir_context);
      if (!thread_id_to_hero_operand_map.has_value()) {
        return std::nullopt;
      }
      GroupedByOpIndexingMap instr_indexing_keyed_by_operands =
          ComputeGroupedOutputToInputIndexing(fusion_adaptor, hero_operand,
                                              mlir_context);
      for (const HloInstruction* operand : operands) {
        auto operand_indexing_maps_it =
            instr_indexing_keyed_by_operands.find(operand);
        if (operand_indexing_maps_it ==
            instr_indexing_keyed_by_operands.end()) {
          continue;
        }
        const Shape& operand_shape = operand->shape();
        IndexingMap operand_logical_to_physical_map =
            GetIndexingMapFromLogicalToPhysicalLayout(operand_shape,
                                                      mlir_context);
        IndexingMap operand_physical_to_linearized_shape = GetBitcastMap(
            ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
                operand_shape),
            GetLinearizedShape(operand_shape), mlir_context);
        IndexingMap operand_logical_to_linearized_physical_shape =
            operand_logical_to_physical_map *
            operand_physical_to_linearized_shape;
        operand_logical_to_linearized_physical_shape.Simplify();
        for (const IndexingMap& operand_indexing_map :
             operand_indexing_maps_it->second) {
          if (operand_indexing_map.IsUndefined()) {
            result[operand] = {operand_indexing_map};
            break;
          }
          IndexingMap logical_output_to_linearized_physical_input_map =
              operand_indexing_map *
              operand_logical_to_linearized_physical_shape;
          IndexingMap thread_id_to_linearized_physical_input_map =
              *thread_id_to_hero_operand_map *
              logical_output_to_linearized_physical_input_map;
          thread_id_to_linearized_physical_input_map.Simplify();
          result[operand].insert(thread_id_to_linearized_physical_input_map);
        }
      }
    }
  }
  return result;
}
void AssignValuesToRTVars(IndexingMap* indexing_map) {
  if (indexing_map->GetRTVarsCount() == 0) {
    return;
  }
  MLIRContext* mlir_context = indexing_map->GetMLIRContext();
  llvm::SmallVector<AffineExpr, 2> symbol_replacements;
  for (int64_t symbol_id = 0; symbol_id < indexing_map->GetRangeVarsCount();
       ++symbol_id) {
    symbol_replacements.push_back(
        mlir::getAffineSymbolExpr(symbol_id, mlir_context));
  }
  for (const IndexingMap::Variable& rt_var : indexing_map->GetRTVars()) {
    symbol_replacements.push_back(getAffineConstantExpr(
        (rt_var.bounds.lower + rt_var.bounds.upper) / 2, mlir_context));
  }
  AffineMap thread_x_to_input_no_dim_symbols =
      indexing_map->GetAffineMap().replaceDimsAndSymbols(
          {}, symbol_replacements, indexing_map->GetDimVarsCount(),
          indexing_map->GetRangeVarsCount());
  *indexing_map = IndexingMap{thread_x_to_input_no_dim_symbols,
                              indexing_map->GetDimVars(),
                              indexing_map->GetRangeVars(),
                              {}};
  indexing_map->Simplify();
  indexing_map->RemoveUnusedSymbols();
}
void AssignValuesToOuterLoopIVs(IndexingMap* indexing_map) {
  if (indexing_map->GetRangeVarsCount() <= 1) {
    return;
  }
  MLIRContext* mlir_context = indexing_map->GetMLIRContext();
  llvm::SmallVector<AffineExpr, 2> symbol_replacements;
  for (int64_t symbol_id = 0; symbol_id < indexing_map->GetRangeVarsCount() - 1;
       ++symbol_id) {
    symbol_replacements.push_back(getAffineConstantExpr(
        indexing_map->GetRangeVar(symbol_id).bounds.lower, mlir_context));
  }
  symbol_replacements.push_back(mlir::getAffineSymbolExpr(0, mlir_context));
  AffineMap thread_x_to_input_no_dim_symbols =
      indexing_map->GetAffineMap().replaceDimsAndSymbols(
          {}, symbol_replacements, indexing_map->GetDimVarsCount(), 1);
  *indexing_map = IndexingMap{thread_x_to_input_no_dim_symbols,
                              indexing_map->GetDimVars(),
                              {indexing_map->GetRangeVars().back()},
                              {}};
  indexing_map->Simplify();
  indexing_map->RemoveUnusedSymbols();
}
struct PartitionedExpr {
  explicit PartitionedExpr(MLIRContext* mlir_context) {
    AffineExpr zero = getAffineConstantExpr(0, mlir_context);
    func_of_d0 = zero;
    func_of_s0 = zero;
  }
  AffineExpr func_of_d0;
  AffineExpr func_of_s0;
};
std::optional<PartitionedExpr> Partition(AffineExpr expr) {
  PartitionedExpr result(expr.getContext());
  std::vector<AffineExpr> summands;
  std::stack<AffineExpr> dfs;
  dfs.push(expr);
  while (!dfs.empty()) {
    auto top = dfs.top();
    dfs.pop();
    auto sum = mlir::dyn_cast<AffineBinaryOpExpr>(top);
    if (sum && sum.getKind() == AffineExprKind::Add) {
      dfs.push(sum.getLHS());
      dfs.push(sum.getRHS());
      continue;
    }
    bool depends_on_thread_x = top.isFunctionOfDim(0);
    bool depends_on_range = top.isFunctionOfSymbol(0);
    if (depends_on_thread_x && depends_on_range) {
      return std::nullopt;
    }
    if (depends_on_thread_x) {
      result.func_of_d0 = top + result.func_of_d0;
    }
    if (depends_on_range) {
      result.func_of_s0 = top + result.func_of_s0;
    }
  }
  return result;
}
void FindAllIndices(AffineExpr expr, int dim_id, int symbol_id,
                    const std::vector<Interval>& dimension_ranges,
                    const std::vector<Interval>& symbol_ranges,
                    std::vector<int64_t>* dimensions,
                    std::vector<int64_t>* symbols,
                    std::vector<int64_t>* indices) {
  if (dim_id < dimension_ranges.size()) {
    Interval dim_range = dimension_ranges[dim_id];
    for (int64_t dim_value = dim_range.lower; dim_value <= dim_range.upper;
         ++dim_value) {
      dimensions->push_back(dim_value);
      FindAllIndices(expr, dim_id + 1, symbol_id, dimension_ranges,
                     symbol_ranges, dimensions, symbols, indices);
      dimensions->pop_back();
    }
    return;
  }
  if (symbol_id < symbol_ranges.size()) {
    Interval symbol_range = symbol_ranges[symbol_id];
    for (int64_t symbol_value = symbol_range.lower;
         symbol_value <= symbol_range.upper; ++symbol_value) {
      symbols->push_back(symbol_value);
      FindAllIndices(expr, dim_id, symbol_id + 1, dimension_ranges,
                     symbol_ranges, dimensions, symbols, indices);
      symbols->pop_back();
    }
    return;
  }
  indices->push_back(EvaluateAffineExpr(expr, *dimensions, *symbols));
}
std::vector<Interval> FindIntervals(
    AffineExpr expr, const std::vector<Interval>& dimension_ranges,
    const std::vector<Interval>& symbol_ranges = {}) {
  std::vector<int64_t> dimensions, symbols;
  std::vector<int64_t> linear_indices;
  FindAllIndices(expr, 0, 0, dimension_ranges, symbol_ranges, &dimensions,
                 &symbols, &linear_indices);
  std::sort(linear_indices.begin(), linear_indices.end());
  linear_indices.erase(
      std::unique(linear_indices.begin(), linear_indices.end()),
      linear_indices.end());
  std::vector<Interval> intervals;
  for (int i = 0, start, end; i < linear_indices.size();) {
    start = linear_indices[i++];
    end = start;
    while (i < linear_indices.size() && linear_indices[i] == end + 1) {
      ++end;
      ++i;
    }
    intervals.push_back(Interval{start, end});
  }
  return intervals;
}
std::vector<Interval> ExtendIntervals(const std::vector<Interval>& intervals,
                                      int64_t length) {
  std::vector<Interval> overlapped_intervals;
  for (int i = 0; i < intervals.size();) {
    int64_t lower = intervals[i].lower;
    int64_t upper = intervals[i].upper + length;
    ++i;
    while (i < intervals.size() && upper >= intervals[i].lower - 1) {
      upper = std::max(upper, intervals[i].upper + length);
      ++i;
    }
    overlapped_intervals.push_back(Interval{lower, upper});
  }
  return overlapped_intervals;
}
std::vector<Interval> FindContiguousIntervals(
    const PartitionedExpr& partitioned_expr, const IndexingMap& indexing_map) {
  constexpr int64_t kNumThreadsPerWarp = 32;
  MLIRContext* mlir_context = indexing_map.GetMLIRContext();
  AffineExpr thread_x = mlir::getAffineDimExpr(0, mlir_context);
  AffineExpr range = mlir::getAffineSymbolExpr(0, mlir_context);
  if (partitioned_expr.func_of_d0 == thread_x) {
    return {Interval{0, kNumThreadsPerWarp - 1}};
  }
  if (auto mul =
          mlir::dyn_cast<AffineBinaryOpExpr>(partitioned_expr.func_of_d0);
      mul && mul.getKind() == AffineExprKind::Mul) {
    if (auto multiplier = mlir::dyn_cast<AffineConstantExpr>(mul.getRHS());
        multiplier) {
      if (multiplier.getValue() == -1) {
        return {Interval{0, kNumThreadsPerWarp - 1}};
      }
      if (partitioned_expr.func_of_s0 == range) {
        Interval range_interval = indexing_map.GetSymbolBound(0);
        int64_t num_elems = range_interval.GetLoopTripCount();
        if (num_elems >= std::abs(multiplier.getValue())) {
          return {Interval{0, multiplier.getValue() * (kNumThreadsPerWarp - 1) +
                                  num_elems - 1}};
        }
        std::vector<Interval> intervals;
        for (int i = 0, dm = 0; i < kNumThreadsPerWarp;
             ++i, dm += multiplier.getValue()) {
          intervals.push_back(
              {range_interval.lower + dm, range_interval.upper + dm});
        }
        return intervals;
      }
      std::vector<Interval> intervals;
      for (int i = 0, dm = 0; i < kNumThreadsPerWarp;
           ++i, dm += multiplier.getValue()) {
        intervals.push_back({dm, dm});
      }
      return intervals;
    }
  }
  auto intervals = FindIntervals(partitioned_expr.func_of_d0,
                                 {indexing_map.GetDimVars(0).bounds});
  if (partitioned_expr.func_of_s0 != range) {
    return intervals;
  }
  Interval range_interval = indexing_map.GetSymbolBound(0);
  return ExtendIntervals(intervals, range_interval.GetLoopTripCount() - 1);
}
bool IsIndexingCoalesced(IndexingMap& thread_x_to_linearized_input,
                         PrimitiveType element_type) {
  if (thread_x_to_linearized_input.IsUndefined()) {
    return false;
  }
  if (thread_x_to_linearized_input.GetAffineMap().getNumResults() == 0) {
    return true;
  }
  AssignValuesToRTVars(&thread_x_to_linearized_input);
  MLIRContext* mlir_context = thread_x_to_linearized_input.GetMLIRContext();
  AffineExpr thread_x_dim = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  AffineExpr c0 = getAffineConstantExpr(0, mlir_context);
  IndexingMap thread_x_first_32_elements{
      AffineMap::get(1, 0, {thread_x_dim, c0, c0, c0, c0, c0}, mlir_context),
      {IndexingMap::Variable{{0, 31}}},
      {},
      {}};
  IndexingMap thread_x_to_input_sample =
      thread_x_first_32_elements * thread_x_to_linearized_input;
  thread_x_to_input_sample.Simplify();
  thread_x_to_input_sample.RescaleSymbols();
  thread_x_to_input_sample.RemoveUnusedSymbols();
  if (thread_x_to_input_sample.IsKnownEmpty()) {
    return true;
  }
  AssignValuesToOuterLoopIVs(&thread_x_to_input_sample);
  auto partitioned_expr =
      Partition(thread_x_to_input_sample.GetAffineMap().getResult(0));
  if (!partitioned_expr.has_value()) {
    return false;
  }
  if (thread_x_to_input_sample.GetConstraintsCount() > 1 ||
      (thread_x_to_input_sample.GetConstraintsCount() == 1 &&
       thread_x_to_input_sample.GetConstraints().begin()->first !=
           partitioned_expr->func_of_d0 + partitioned_expr->func_of_s0)) {
    return false;
  }
  return EstimateCoalescingViaMemoryTransactionsCount(
      FindContiguousIntervals(*partitioned_expr, thread_x_to_input_sample),
      element_type);
}
}  
CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* instr,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context,
    bool use_heuristic) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(instr);
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            *fusion_adaptor, operands, fusion_analysis,
                            fusion_interface, mlir_context)) {
    return;
  }
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_analysis.GetEmitterFusionKind(), instr);
}
CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* producer, const HloInstruction* consumer,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context,
    bool use_heuristic) {
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(producer, consumer);
  if (!use_heuristic && ComputeCoalescingForAllOperands(
                            *fusion_adaptor, operands, fusion_analysis,
                            fusion_interface, mlir_context)) {
    return;
  }
  is_coalesced_computed_by_heuristic_ = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer, consumer);
}
bool CoalescingAnalysis::ComputeCoalescingForAllOperands(
    const HloFusionAdaptor& fusion_adaptor,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis,
    KernelFusionInterface* fusion_interface, MLIRContext* mlir_context) {
  std::optional<GroupedByOpIndexingMap> thread_id_to_input_memory_layouts =
      GetThreadIdToInputMemoryLayoutsMaps(fusion_adaptor, operands,
                                          fusion_analysis, fusion_interface,
                                          mlir_context);
  if (!thread_id_to_input_memory_layouts.has_value()) {
    return false;
  }
  for (const HloInstruction* operand : operands) {
    if (operand->shape().rank() == 0) {
      coalescing_per_operand_.insert({operand, true});
      continue;
    }
    auto operand_indexing_maps =
        thread_id_to_input_memory_layouts->find(operand);
    if (operand_indexing_maps == thread_id_to_input_memory_layouts->end()) {
      coalescing_per_operand_.insert({operand, true});
      continue;
    }
    for (IndexingMap operand_indexing_map : operand_indexing_maps->second) {
      bool is_coalesced = IsIndexingCoalesced(operand_indexing_map,
                                              operand->shape().element_type());
      auto [it, inserted] =
          coalescing_per_operand_.insert({operand, is_coalesced});
      if (!inserted) {
        it->second &= is_coalesced;
      }
      if (!is_coalesced) break;
    }
  }
  return true;
}
bool CoalescingAnalysis::IsReadCoalesced(const HloInstruction* operand) const {
  auto it = coalescing_per_operand_.find(operand);
  if (it == coalescing_per_operand_.end()) {
    return is_coalesced_computed_by_heuristic_;
  }
  return it->second;
}
}  
}  