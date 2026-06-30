#include "xla/service/gpu/gpu_fusible.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stack>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
namespace xla {
namespace gpu {
namespace {
bool HasAnyTiledTransposeRoot(const HloComputation& computation) {
  return absl::c_any_of(GetFusionRoots(computation),
                        [&](const HloInstruction* instr) {
                          return GetDescriptionForTiledTransposeEmitter(
                                     FindNonTrivialHero(*instr))
                              .has_value();
                        });
}
const Shape& GetElementShape(const HloFusionAnalysis& analysis) {
  const Shape* shape = &analysis.fusion_root(0).shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}
int ComputeMaxUnrollFactor(int64_t num_elements) {
  constexpr int kMaxUnrollFactor = 4;
  for (int i = kMaxUnrollFactor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }
  return 1;
}
}  
bool IfFusedReadsElementsMultipleTimes(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  if (instr.opcode() == HloOpcode::kGather ||
      instr.opcode() == HloOpcode::kBroadcast) {
    return ShapeUtil::ElementsIn(instr.shape()) >
           ShapeUtil::ElementsIn(instr.operand(0)->shape());
  }
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    for (const auto& dim : instr.window().dimensions()) {
      if (dim.size() > dim.stride()) {
        return true;
      }
    }
  }
  return false;
}
bool IsExpensiveToRepeat(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  constexpr int kMaxInputsPerOutput = 10;
  if (instr.opcode() == HloOpcode::kReduce &&
      !IsReductionFromOrToContiguousDimensions(instr)) {
    int64_t reduction_ratio = ShapeUtil::ElementsIn(instr.operand(0)->shape()) /
                              ShapeUtil::ElementsIn(instr.shape());
    if (reduction_ratio > kMaxInputsPerOutput) return true;
  }
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    int64_t reduction_ratio = 1;
    for (const auto& dim : instr.window().dimensions())
      reduction_ratio *= dim.size();
    if (reduction_ratio > kMaxInputsPerOutput) return true;
  }
  return false;
}
bool IsPhysicallyTransposing(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    for (const HloInstruction* fused_instr : instr.fused_instructions()) {
      if (IsPhysicallyTransposing(*fused_instr)) {
        return true;
      }
    }
  }
  return instr.opcode() == HloOpcode::kCopy ||
         (instr.opcode() == HloOpcode::kTranspose &&
          !ShapeUtil::TransposeIsBitcast(instr.operand(0)->shape(),
                                         instr.shape(), instr.dimensions()));
}
namespace {
std::pair<int64_t, int64_t> MostMinorNonTrivialDimension(const Shape& shape) {
  int64_t position_of_first_non_trivial_dim = 0;
  for (int64_t dim : shape.layout().minor_to_major()) {
    if (shape.dimensions()[dim] > 1) {
      return {dim, position_of_first_non_trivial_dim};
    }
    ++position_of_first_non_trivial_dim;
  }
  return {-1, position_of_first_non_trivial_dim};
}
}  
bool TransposesMinorDimension(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kFusion:
      return absl::c_any_of(instr->fused_instructions(),
                            TransposesMinorDimension);
    case HloOpcode::kCopy: {
      int64_t first_non_trivial_operand_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).first;
      int64_t first_non_trivial_output_dim =
          MostMinorNonTrivialDimension(instr->shape()).first;
      return first_non_trivial_operand_dim != first_non_trivial_output_dim;
    }
    case HloOpcode::kTranspose: {
      auto position_in_minor_to_major = InversePermutation(
          instr->operand(0)->shape().layout().minor_to_major());
      int64_t position_of_first_non_trivial_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).second;
      for (int64_t output_dim : instr->shape().layout().minor_to_major()) {
        if (instr->shape().dimensions()[output_dim] == 1) {
          continue;
        }
        int64_t operand_dim = instr->dimensions().at(output_dim);
        return position_in_minor_to_major[operand_dim] >
               position_of_first_non_trivial_dim;
      }
      return false;
    }
    default:
      return false;
  }
}
bool IsReduceInputFusion(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion &&
         absl::c_any_of(GetFusionRoots(*instr.called_computations()[0]),
                        [](const HloInstruction* root) {
                          return IsRealReductionHero(*root,
                                                     FindNonTrivialHero(*root));
                        });
}
bool IsInputFusibleReduction(const HloInstruction& instr) {
  return IsReduceInputFusion(instr) ||
         IsReductionFromOrToContiguousDimensions(instr);
}
bool IsNestableVariadicReduction(const HloInstruction& instr) {
  return instr.shape().IsTuple() &&
         ((instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr)) ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop &&
           instr.fused_expression_root()->opcode() == HloOpcode::kReduce));
}
bool IsInputFusibleTranspose(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kBitcast || instr.IsCustomFusion()) {
    return false;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return HasAnyTiledTransposeRoot(*instr.fused_instructions_computation());
  }
  return GetDescriptionForTiledTransposeEmitter(instr).has_value();
}
const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    const auto& hero = FindNonTrivialHero(*fused_expression_root);
    if (IsRealReductionHero(*fused_expression_root, hero) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return &hero;
    }
    return fused_expression_root;
  }
  for (auto* inst : fused_expression_root->mutable_operands()) {
    const auto& hero = FindNonTrivialHero(*inst);
    if (IsRealReductionHero(*inst, hero) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return &hero;
    }
  }
  return fused_expression_root->operands()[0];
}
FusionDecision FusionHeroesAreCompatible(const HloInstruction* hero1,
                                         const HloInstruction* hero2) {
  auto hero1_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero1);
  auto tiled_transpose_hero1 = GetDescriptionForTiledTransposeEmitter(*hero1);
  bool hero1_is_unnested_transpose = tiled_transpose_hero1.has_value();
  bool hero2_is_unnested_reduce =
      IsReductionFromOrToContiguousDimensions(*hero2);
  auto tiled_transpose_hero2 = GetDescriptionForTiledTransposeEmitter(*hero2);
  bool hero2_is_unnested_transpose = tiled_transpose_hero2.has_value();
  if (hero1_is_unnested_reduce && hero2_is_unnested_reduce &&
      !AreReductionsMultiOutputFusionCompatible(hero2, hero1)) {
    return FusionDecision::Forbid("tiled reductions with different shapes");
  } else if (hero1_is_unnested_transpose && hero2_is_unnested_transpose &&
             !tiled_transpose_hero1->IsEquivalent(*tiled_transpose_hero2)) {
    return FusionDecision::Forbid("tiled transposes with different shapes");
  } else if ((hero1_is_unnested_transpose && hero2_is_unnested_reduce) ||
             (hero1_is_unnested_reduce && hero2_is_unnested_transpose)) {
    return FusionDecision::Forbid("MOF-fusion of a transpose and a reduction");
  }
  if (hero1_is_unnested_transpose || hero2_is_unnested_transpose) {
    auto check_path_of_intermediate_ops = [](HloInstruction* param) {
      if (param->user_count() != 1) {
        return false;
      }
      HloInstruction* hlo = param->users()[0];
      while (hlo->user_count() > 0) {
        if (!IsIntermediate(hlo)) {
          return false;
        }
        hlo = hlo->users()[0];
      }
      return true;
    };
    HloInstruction* fusion1 = hero1->parent()->FusionInstruction();
    HloInstruction* fusion2 = hero2->parent()->FusionInstruction();
    if (fusion1 != nullptr && fusion2 != nullptr) {
      if (hero1_is_unnested_transpose && fusion2->IsUserOf(fusion1)) {
        int64_t operand_idx = fusion2->operand_index(fusion1);
        auto hlo = fusion2->fused_parameter(operand_idx);
        if (!check_path_of_intermediate_ops(hlo)) {
          return FusionDecision::Forbid("tiled transpose would become untiled");
        }
      } else if (hero2_is_unnested_transpose && fusion1->IsUserOf(fusion2)) {
        int64_t operand_idx = fusion1->operand_index(fusion2);
        auto hlo = fusion1->fused_parameter(operand_idx);
        if (!check_path_of_intermediate_ops(hlo)) {
          return FusionDecision::Forbid("tiled transpose would become untiled");
        }
      }
    }
  }
  return FusionDecision::Allow();
}
FusionDecision ShapesCompatibleForMultiOutputFusion(
    const HloInstruction& instr1, const HloInstruction& instr2) {
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    const auto& hero = element_instr->parent()->IsFusionComputation()
                           ? FindNonTrivialHero(*element_instr)
                           : *element_instr;
    if (IsReductionFromOrToContiguousDimensions(*element_instr) ||
        GetDescriptionForTiledTransposeEmitter(hero).has_value()) {
      return hero.operand(0)->shape();
    }
    return element_instr->shape();
  };
  const HloInstruction* hero1 = GetRealHeroForMultiOutputFusion(instr1);
  const HloInstruction* hero2 = GetRealHeroForMultiOutputFusion(instr2);
  if (auto compatible = FusionHeroesAreCompatible(hero1, hero2); !compatible) {
    return compatible;
  }
  const Shape& l1 = get_loop_shape(hero1);
  const Shape& l2 = get_loop_shape(hero2);
  bool accept_unequal_shape = !l1.IsTuple() && !l2.IsTuple();
  if (!ShapeUtil::EqualIgnoringElementType(l1, l2) &&
      (!accept_unequal_shape ||
       !ShapeUtil::IsReshapeOrTransposeBitcast(l1, l2,
                                               true))) {
    return FusionDecision::Forbid("different loop shapes");
  }
  return FusionDecision::Allow();
}
bool IsInputFusibleScatter(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}
bool IsInputFusible(const HloInstruction& instr) {
  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleScatter(instr) ||
          IsInputFusibleTranspose(instr));
}
bool IsUniversallyLoopFusible(const HloInstruction& instr) {
  if (instr.IsElementwise() && instr.operand_count() > 0 &&
      instr.opcode() != HloOpcode::kCopy) {
    return true;
  }
  switch (instr.opcode()) {
    case HloOpcode::kCopy:
      return !GetDescriptionForTiledTransposeEmitter(instr).has_value();
    case HloOpcode::kFusion:
      return instr.fusion_kind() == HloInstruction::FusionKind::kLoop;
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}
bool IsLoopFusibleAsConsumer(const HloInstruction& instr) {
  if (!instr.IsFusible()) return false;
  if (instr.opcode() == HloOpcode::kBitcast) return false;
  if (instr.opcode() == HloOpcode::kReduce) return true;
  if (!IsInputFusible(instr) && instr.opcode() == HloOpcode::kFusion &&
      instr.fusion_kind() == HloInstruction::FusionKind::kInput) {
    return true;
  }
  return IsUniversallyLoopFusible(instr);
}
bool IsLoopFusibleAsProducer(const HloInstruction& instr) {
  if (!instr.IsFusible()) return false;
  switch (instr.opcode()) {
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
      return true;
    case HloOpcode::kReduce:
      return !instr.shape().IsTuple();
    default:
      return IsUniversallyLoopFusible(instr);
  }
}
static bool AllSatisfy(const HloInstruction& instr,
                       const HloPredicate& predicate) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return predicate(&instr);
  }
  return absl::c_all_of(
      instr.fused_instructions(), [&](const HloInstruction* i) {
        return i->opcode() == HloOpcode::kParameter || predicate(i);
      });
}
FusionDecision CanEmitInputFusedScatter(const HloInstruction& producer,
                                        const HloInstruction& consumer) {
  if (IsInputFusibleScatter(producer)) {
    return FusionDecision::Forbid("do not fuse into the output of scatter");
  }
  if (!IsInputFusibleScatter(consumer)) {
    return FusionDecision::Allow();
  }
  const HloInstruction* inplace_operand;
  if (consumer.opcode() == HloOpcode::kFusion) {
    const HloInstruction* scatter = consumer.fused_expression_root();
    CHECK_EQ(scatter->opcode(), HloOpcode::kScatter);
    CHECK_EQ(scatter->operand(0)->opcode(), HloOpcode::kParameter);
    inplace_operand = consumer.operand(scatter->operand(0)->parameter_number());
  } else {
    inplace_operand = consumer.operand(0);
  }
  if (inplace_operand == &producer) {
    return FusionDecision::Forbid(
        "do not fuse into the in-place operand of scatter");
  }
  if (absl::c_linear_search(producer.operands(), inplace_operand)) {
    return FusionDecision::Forbid(
        "Producer uses the in-place operand of a scatter");
  }
  return FusionDecision::Allow();
}
FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer) {
  if (!IsLoopFusibleAsProducer(producer) &&
      !IsInputFusibleTranspose(producer)) {
    return FusionDecision::Forbid("the producer is not loop-fusible");
  }
  if (IsInputFusibleReduction(producer)) {
    if (!producer.GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_reduction_epilogue_fusion()) {
      return FusionDecision::Forbid(
          "Reduction epilogue fusion is not enabled.");
    }
    const HloInstruction& reduce_hero =
        producer.opcode() == HloOpcode::kFusion
            ? FindNonTrivialHero(*producer.fused_expression_root())
            : producer;
    if (!ReductionIsRaceFree(
            reduce_hero.GetModule()->config(),
            GetReductionKindAndContiguousComponents(reduce_hero))) {
      return FusionDecision::Forbid(
          "Reduction output fusion only works for race free reductions");
    }
    if (!AllSatisfy(consumer, [](const HloInstruction* hlo) {
          return IsIntermediate(hlo, 1);
        })) {
      return FusionDecision::Forbid(
          "Reductions from/to continuous dims epilogue not fusible");
    }
    if (producer.user_count() > 1) {
      return FusionDecision::Forbid(
          "reduction output fusion only works for single user");
    }
  }
  if (auto can_fuse = CanEmitInputFusedScatter(producer, consumer); !can_fuse) {
    return can_fuse;
  }
  if (!IsInputFusible(consumer) && !IsLoopFusibleAsConsumer(consumer)) {
    return FusionDecision::Forbid(
        "the consumer is not input-fusible and not loop-fusible");
  }
  if (producer.IsMultiOutputFusion()) {
    return FusionDecision::Forbid(
        "the producer is not fusible as it is a multi-output fusion");
  }
  if (producer.opcode() == HloOpcode::kConstant &&
      (!ShapeUtil::IsEffectiveScalar(producer.shape()) ||
       consumer.opcode() != HloOpcode::kFusion)) {
    return FusionDecision::Forbid("not fusing constant");
  }
  return InstructionFusion::ShouldFuseInPlaceOp(&producer, &consumer);
}
FusionDecision IsProducerMultiOutputFusible(const HloInstruction& producer) {
  if (producer.IsMultiOutputFusion()) {
    return FusionDecision::Forbid("Producer is a multi-output fusion");
  }
  if (!HloDataflowAnalysis::GetInPlaceInputOutputPairs(&producer).empty()) {
    return FusionDecision::Forbid("In-place operations are present");
  }
  if (!IsLoopFusibleAsProducer(producer)) {
    return FusionDecision::Forbid("producer is not loop-fusible");
  }
  if (IsPhysicallyTransposing(producer)) {
    return FusionDecision::Forbid("producer is physically transposing");
  }
  return FusionDecision::Allow();
}
static int64_t SharedMemoryUsageNoCache(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo);
    }
    return sum;
  } else if (instr.opcode() == HloOpcode::kReduce &&
             IsReductionFromOrToContiguousDimensions(instr)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(
        instr.operand(0)->shape().element_type());
    int num_variadic =
        instr.shape().IsTuple() ? instr.shape().tuple_shapes_size() : 1;
    if (reduction_info.is_row_reduction) {
      return 32 * primitive_size * num_variadic;
    } else {
      return 4 * 32 * 33 * primitive_size * num_variadic;
    }
  } else if (auto tr = GetDescriptionForTiledTransposeEmitter(instr)) {
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    int64_t bytes_required = 32 * 33 * primitive_size;
    if (tr->permutation.back() == tr->permutation.size() - 1) {
      bytes_required *= tr->dimensions.back();
    }
    return bytes_required;
  }
  return 0;
}
int64_t FusionInfoCache::GetSharedMemoryUsage(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = shared_memory_usage_.find(&instr);
    if (it != shared_memory_usage_.end()) {
      return it->second;
    }
  }
  int64_t shared_memory_usage = SharedMemoryUsageNoCache(instr);
  absl::MutexLock lock(&mutex_);
  shared_memory_usage_.emplace(&instr, shared_memory_usage);
  return shared_memory_usage;
}
int64_t SharedMemoryUsage(const HloInstruction& instr, FusionInfoCache* cache) {
  if (!cache) {
    return SharedMemoryUsageNoCache(instr);
  }
  return cache->GetSharedMemoryUsage(instr);
}
constexpr int64_t kMaxUnnestedReductionOutputsPerFusion = 8;
static int64_t NumUnnestedReductionsNoCache(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr)) {
    return 1;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += NumUnnestedReductionsNoCache(*hlo);
    }
    return sum;
  }
  return 0;
}
int64_t FusionInfoCache::GetNumUnnestedReductions(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = num_unnested_reductions_.find(&instr);
    if (it != num_unnested_reductions_.end()) {
      return it->second;
    }
  }
  int64_t num_unnested_reductions = NumUnnestedReductionsNoCache(instr);
  absl::MutexLock lock(&mutex_);
  num_unnested_reductions_.emplace(&instr, num_unnested_reductions);
  return num_unnested_reductions;
}
static int64_t NumUnnestedReductions(const HloInstruction& instr,
                                     FusionInfoCache* cache) {
  if (!cache) {
    return NumUnnestedReductionsNoCache(instr);
  }
  return cache->GetNumUnnestedReductions(instr);
}
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  const se::DeviceDescription& device_info,
                                  bool is_consumer_producer_fusion,
                                  FusionInfoCache* cache ) {
  if (SharedMemoryUsage(instr1, cache) + SharedMemoryUsage(instr2, cache) >
      device_info.shared_memory_per_block()) {
    return FusionDecision::Forbid(
               "shared memory usage would be over the budget of ")
           << device_info.shared_memory_per_block() << "B";
  }
  if (NumUnnestedReductions(instr1, cache) +
          NumUnnestedReductions(instr2, cache) >
      kMaxUnnestedReductionOutputsPerFusion) {
    return FusionDecision::Forbid("over ")
           << kMaxUnnestedReductionOutputsPerFusion
           << " unnested reductions in fusion";
  }
  int64_t num_output_buffers = ShapeUtil::SubshapeCount(instr1.shape()) +
                               ShapeUtil::SubshapeCount(instr2.shape());
  if (instr1.operand_count() + instr2.operand_count() - 1 +
          num_output_buffers <=
      MaxOperandsAndOutputsPerFusion()) {
    return FusionDecision::Allow();
  } else {
    VLOG(5) << "Operand count of " << "(" << instr1.ToString()
            << " ) = " << instr1.operand_count() << " and ( "
            << instr2.ToString() << " ) = " << instr2.operand_count()
            << " and num_output_buffers = " << num_output_buffers
            << " is bigger than the bound of "
            << MaxOperandsAndOutputsPerFusion();
  }
  absl::flat_hash_set<const HloInstruction*> operands(instr1.operands().begin(),
                                                      instr1.operands().end());
  operands.insert(instr2.operands().begin(), instr2.operands().end());
  operands.erase(&instr1);
  operands.erase(&instr2);
  if (is_consumer_producer_fusion &&
      operands.size() <= instr1.operands().size()) {
    return FusionDecision::Allow();
  }
  if (operands.size() + num_output_buffers > MaxOperandsAndOutputsPerFusion()) {
    return FusionDecision::Forbid(
        "Number of operands and output buffers is larger than allowed budget "
        "per fusion");
  }
  return FusionDecision::Allow();
}
bool CreatesHeavyComputation(const HloInstruction& producer,
                             const HloInstruction& consumer) {
  auto producer_is_heavy = [&](const HloInstruction& instr) {
    if (producer.opcode() != HloOpcode::kFusion) {
      return IsExpensiveToRepeat(producer);
    }
    for (const auto& instr : producer.fused_instructions()) {
      if (IsExpensiveToRepeat(*instr)) {
        return true;
      }
    }
    return false;
  };
  if (!producer_is_heavy(producer)) {
    return false;
  }
  if (consumer.opcode() != HloOpcode::kFusion) {
    return IfFusedReadsElementsMultipleTimes(consumer);
  }
  for (const HloInstruction* operand : consumer.operands()) {
    if (operand != &producer) {
      continue;
    }
    const HloInstruction* root =
        consumer.fused_instructions_computation()->parameter_instruction(
            consumer.operand_index(operand));
    std::stack<const HloInstruction*> dfs;
    dfs.push(root);
    absl::flat_hash_set<const HloInstruction*> visited;
    while (!dfs.empty()) {
      const HloInstruction* cur = dfs.top();
      dfs.pop();
      if (!visited.insert(cur).second) {
        continue;
      }
      if (IfFusedReadsElementsMultipleTimes(*cur)) {
        return true;
      }
      for (const auto& user : cur->users()) {
        if (visited.contains(user)) {
          continue;
        }
        dfs.push(user);
      }
    }
  }
  return false;
}
bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr) {
  return instr.IsFusible() && !instr.IsCustomFusion() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleTranspose(instr) ||
          instr.IsLoopFusion() ||  
          instr.IsElementwise());
}
HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& producer,
                                            const HloInstruction& consumer) {
  return (IsInputFusible(consumer) || IsInputFusible(producer))
             ? HloInstruction::FusionKind::kInput
             : HloInstruction::FusionKind::kLoop;
}
bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    return user == &consumer || user == user->parent()->root_instruction();
  });
}
size_t GetInstrCountOfFusible(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion ? instr.fused_instruction_count()
                                              : 1;
}
absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return {&instr};
  }
  HloInstruction* root = instr.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return {root};
  } else {
    auto v = root->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(v.begin(), v.end());
  }
}
size_t GetOutputSizeOfFusible(const HloInstruction& instr) {
  if (!instr.IsMultiOutputFusion()) {
    return 1;
  }
  const HloInstruction* root = instr.fused_expression_root();
  return ShapeUtil::TupleElementCount(root->shape());
}
static void GetFusionRootsRec(const HloInstruction* root,
                              std::vector<const HloInstruction*>& out) {
  if (root->opcode() == HloOpcode::kGetTupleElement &&
      root->operand(0)->opcode() == HloOpcode::kTuple) {
    return GetFusionRootsRec(root->operand(0)->operand(root->tuple_index()),
                             out);
  } else if (root->opcode() == HloOpcode::kGetTupleElement) {
    out.push_back(root->operand(0));
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < root->operand_count(); i++) {
      GetFusionRootsRec(root->operand(i), out);
    }
  } else {
    out.push_back(root);
  }
}
std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation) {
  std::vector<const HloInstruction*> out;
  GetFusionRootsRec(computation.root_instruction(), out);
  return out;
}
bool IsGenericTritonFusion(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion &&
         instr.fusion_kind() == HloInstruction::FusionKind::kCustom &&
         instr.backend_config<GpuBackendConfig>().ok() &&
         instr.backend_config<GpuBackendConfig>()
                 ->fusion_backend_config()
                 .kind() == kTritonFusionKind;
}
bool MayPreventVectorization(const HloFusionAdaptor& fusion) {
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;
  return HloAnyOf(fusion, [&](auto node) {
    switch (node.opcode()) {
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSort:
      case HloOpcode::kDot:
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kTan:
      case HloOpcode::kPower:
      case HloOpcode::kAtan2:
        return true;
      case HloOpcode::kConcatenate:
        return node.instruction().operand_count() >
               kMaxConcatArgumentsForUnrolling;
      case HloOpcode::kReduce:
        return node.instruction().shape().tuple_shapes_size() > 1;
      default:
        return false;
    }
  });
}
std::vector<HloComputation*> GetFusibleComputations(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto result = module.MakeComputationPostOrder(execution_threads);
  absl::flat_hash_set<const HloComputation*> computations_not_to_fuse;
  for (const auto* computation : result) {
    for (const auto* instr : computation->instructions()) {
      if (HloInstruction::MightHaveCalledComputations(instr->opcode()) &&
          instr->opcode() != HloOpcode::kWhile &&
          instr->opcode() != HloOpcode::kConditional &&
          instr->opcode() != HloOpcode::kFusion) {
        for (auto* called : instr->called_computations()) {
          computations_not_to_fuse.insert(called);
        }
      }
    }
  }
  result.erase(
      std::remove_if(result.begin(), result.end(),
                     [&](HloComputation* computation) {
                       return computation->IsFusionComputation() ||
                              computations_not_to_fuse.contains(computation);
                     }),
      result.end());
  return result;
}
LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis) {
  return ComputeLoopFusionConfig(analysis, GetElementShape(analysis));
}
LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis, const Shape& element_shape) {
  int unroll_factor = 1;
  int64_t num_elements = ShapeUtil::ElementsIn(element_shape);
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayPreventVectorization(analysis.fusion())) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  unroll_factor = std::max(
      unroll_factor,
      CeilOfRatio(8, analysis.input_output_info().smallest_output_dtype_bits));
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  VLOG(2) << "Unroll factor: " << unroll_factor;
  LaunchDimensionsConfig launch_config{unroll_factor};
  return launch_config;
}
}  
}  