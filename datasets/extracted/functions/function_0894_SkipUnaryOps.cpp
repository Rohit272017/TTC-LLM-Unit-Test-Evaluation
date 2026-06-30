#include "xla/service/gpu/transforms/cudnn_norm_rewriter.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include "google/protobuf/wrappers.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  
#include "third_party/gpus/cudnn/cudnn.h"        
#include "third_party/gpus/cudnn/cudnn_version.h"
#endif
namespace xla {
namespace gpu {
namespace {
namespace m = match;
const HloInstruction* SkipUnaryOps(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kConvert ||
         instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape) {
    instr = instr->operand(0);
  }
  return instr;
}
void SkipUnaryOpsTopDownRecursive(HloInstruction* instr,
                                  std::vector<HloInstruction*>& instrs) {
  if (instr->opcode() == HloOpcode::kConvert ||
      instr->opcode() == HloOpcode::kBitcast ||
      instr->opcode() == HloOpcode::kReshape) {
    for (HloInstruction* user : instr->users()) {
      SkipUnaryOpsTopDownRecursive(user, instrs);
    }
  } else {
    instrs.emplace_back(instr);
  }
}
struct NormMetadata {
  HloInstruction *x_transpose, *y_transpose;
  std::vector<int64_t> norm_dims_adjusted, non_norm_dims_adjusted;
};
using NormMetadataMap = absl::flat_hash_map<HloInstruction*, NormMetadata>;
class UniqueHloInstruction {
 public:
  UniqueHloInstruction()
      : is_set_(false), instr_(nullptr), capture_or_verify_() {}
  HloInstruction* Instr() const { return instr_; }
  void SetInstr(HloInstruction* instr) {
    is_set_ = true;
    instr_ = instr;
  }
  bool CaptureOrVerify(HloInstruction* instr) {
    if (is_set_ && instr != instr_) {
      instr_ = nullptr;
    }
    if (!is_set_) {
      is_set_ = true;
      instr_ = instr;
    }
    return instr_;
  }
  std::function<bool(const HloInstruction*)> GetCaptureOrVerifyFn() {
    if (!capture_or_verify_) {
      capture_or_verify_ = [this](const HloInstruction* instr) -> bool {
        return CaptureOrVerify(const_cast<HloInstruction*>(instr));
      };
    }
    return capture_or_verify_;
  }
 private:
  bool is_set_;
  HloInstruction* instr_;
  std::function<bool(const HloInstruction*)> capture_or_verify_;
};
absl::StatusOr<int64_t> CConstant(
    se::CudaComputeCapability cuda_compute_capability) {
  if (cuda_compute_capability.major == se::CudaComputeCapability::AMPERE) {
    return 32 * 128;
  } else if (cuda_compute_capability.major ==
             se::CudaComputeCapability::HOPPER) {
    return 32 * 144;
  }
  return xla::Internal("Norm kernels require Ampere or Hopper architecture.");
}
bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F16 || element_type == F32;
}
std::vector<int64_t> AdjustedDimensions(const Shape& shape,
                                        absl::Span<const int64_t> dimensions) {
  absl::flat_hash_map<int64_t, int64_t> dimension_map;
  for (int64_t dimension = 0, non_degen_dimension = 0; dimension < shape.rank();
       ++dimension) {
    if (shape.dimensions(dimension) > 1) {
      dimension_map.insert({dimension, non_degen_dimension});
      non_degen_dimension++;
    }
  }
  std::vector<int64_t> adjusted_dimensions;
  for (int64_t dimension : dimensions) {
    auto non_degenerate_dimension = dimension_map.find(dimension);
    if (non_degenerate_dimension != dimension_map.end()) {
      adjusted_dimensions.emplace_back(non_degenerate_dimension->second);
    }
  }
  return adjusted_dimensions;
}
std::vector<int64_t> AdjustedDimensions(const HloInstruction* instr) {
  Shape shape;
  if (instr->opcode() == HloOpcode::kBroadcast) {
    shape = instr->shape();
  } else if (instr->opcode() == HloOpcode::kReduce) {
    shape = instr->operand(0)->shape();
  } else {
    return {};
  }
  return AdjustedDimensions(shape, instr->dimensions());
}
bool AppliesAddReduce(const HloInstruction* instr,
                      absl::Span<const int64_t> reduce_dims = {}) {
  if (instr->opcode() != HloOpcode::kReduce) {
    return false;
  }
  if (!reduce_dims.empty() && AdjustedDimensions(instr) != reduce_dims) {
    return false;
  }
  HloComputation* reduce_comp = instr->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return instr->operand_count() == 2 &&
         instr->operand(1)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsScalar(instr->operand(1)->shape()) &&
         instr->operand(1)->literal().GetAsDouble({}) == 0. &&
         reduce_comp_root->opcode() == HloOpcode::kAdd &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}
bool CalculatesExpectation(const HloInstruction* instr) {
  instr = SkipUnaryOps(instr);
  if (instr->opcode() != HloOpcode::kMultiply) {
    return false;
  }
  bool bcast_operand = instr->operand(0)->opcode() != HloOpcode::kBroadcast;
  const HloInstruction *broadcast = instr->operand(bcast_operand),
                       *reduce = SkipUnaryOps(instr->operand(!bcast_operand));
  if (reduce->opcode() != HloOpcode::kReduce ||
      broadcast->opcode() != HloOpcode::kBroadcast ||
      broadcast->operand(0)->opcode() != HloOpcode::kConstant) {
    return false;
  }
  float actual_r_nelems =
      broadcast->operand(0)->literal().GetAsDouble({}).value();
  int64_t nelems = 1;
  for (int64_t norm_dim : reduce->dimensions()) {
    nelems *= reduce->operand(0)->shape().dimensions()[norm_dim];
  }
  float r_nelems = 1. / static_cast<float>(nelems);
  float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
  return abs(actual_r_nelems - r_nelems) <
         ((actual_r_nelems + r_nelems) * numerical_epsilon);
}
bool FindTargetRecursive(
    const HloInstruction* instr, const HloInstruction* target,
    absl::flat_hash_set<const HloInstruction*>& visited_instrs,
    const HloInstruction* transpose) {
  visited_instrs.emplace(instr);
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape};
  if (instr == target) {
    return true;
  }
  for (HloInstruction* user : instr->users()) {
    if ((supported_ops.contains(user->opcode()) || user == transpose) &&
        !visited_instrs.contains(user)) {
      return FindTargetRecursive(user, target, visited_instrs, transpose);
    }
  }
  if (supported_ops.contains(instr->opcode())) {
    return FindTargetRecursive(instr->operand(0), target, visited_instrs,
                               transpose);
  }
  return false;
}
bool FindTarget(const HloInstruction* custom_call, const HloInstruction* instr,
                const HloInstruction* target,
                const NormMetadataMap& norm_metadata) {
  absl::flat_hash_set<const HloInstruction*> visited_instrs;
  auto custom_call_metadata = norm_metadata.find(custom_call);
  if (custom_call_metadata == norm_metadata.end()) {
    return false;
  }
  return FindTargetRecursive(instr, target, visited_instrs,
                             custom_call_metadata->second.x_transpose);
}
std::vector<int64_t> MapDimensions(const Shape& original_shape,
                                   const Shape& reshaped_shape,
                                   const absl::Span<const int64_t> dimensions) {
  auto dimension_product =
      [](const Shape& shape,
         absl::Span<const int64_t> product_dimensions) -> int64_t {
    int64_t product = 1;
    for (int64_t product_dimension : product_dimensions) {
      product *= shape.dimensions(product_dimension);
    }
    return product;
  };
  absl::flat_hash_map<int64_t, std::vector<int64_t>> dimensions_map;
  std::vector<int64_t> original_dimensions, reshaped_dimensions;
  for (int64_t original_dimension = 0, reshaped_dimension = 0;
       original_dimension < original_shape.rank(); ++original_dimension) {
    original_dimensions.emplace_back(original_dimension);
    while ((reshaped_dimensions.empty() ||
            dimension_product(reshaped_shape, reshaped_dimensions) <
                dimension_product(original_shape, original_dimensions)) &&
           reshaped_dimension < reshaped_shape.rank()) {
      reshaped_dimensions.emplace_back(reshaped_dimension++);
    }
    if (original_dimensions.size() > 1 && reshaped_dimensions.size() > 1) {
      return {};
    }
    if (dimension_product(original_shape, original_dimensions) ==
        dimension_product(reshaped_shape, reshaped_dimensions)) {
      std::vector<int64_t> original_dimensions_in_dimensions;
      std::set_intersection(
          original_dimensions.begin(), original_dimensions.end(),
          dimensions.begin(), dimensions.end(),
          std::back_inserter(original_dimensions_in_dimensions));
      if (!original_dimensions_in_dimensions.empty() &&
          original_dimensions_in_dimensions.size() !=
              original_dimensions.size()) {
        return {};
      }
      for (int64_t dimension : original_dimensions) {
        dimensions_map.insert({dimension, reshaped_dimensions});
      }
      original_dimensions.clear();
      reshaped_dimensions.clear();
    }
  }
  std::vector<int64_t> mapped_dimensions;
  for (int64_t dimension : dimensions) {
    auto mapped_dimension = dimensions_map.find(dimension);
    if (mapped_dimension == dimensions_map.end()) {
      return {};
    }
    mapped_dimensions.insert(mapped_dimensions.end(),
                             mapped_dimension->second.begin(),
                             mapped_dimension->second.end());
  }
  mapped_dimensions.erase(
      std::unique(mapped_dimensions.begin(), mapped_dimensions.end()),
      mapped_dimensions.end());
  return mapped_dimensions;
}
HloInstruction* FindAddReduceRecursive(
    HloInstruction* instr, const Shape& orig_instr_shape,
    const absl::Span<const int64_t> reduce_dims,
    absl::flat_hash_set<HloInstruction*>& visited_instrs) {
  visited_instrs.emplace(instr);
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape};
  for (HloInstruction* user : instr->users()) {
    if (user->opcode() == HloOpcode::kReduce) {
      std::vector<int64_t> mapped_reduce_dims =
          MapDimensions(orig_instr_shape, instr->shape(), reduce_dims);
      if (!mapped_reduce_dims.empty() &&
          AppliesAddReduce(user, mapped_reduce_dims)) {
        return user;
      }
    }
    if (supported_ops.contains(user->opcode()) &&
        !visited_instrs.contains(user)) {
      return FindAddReduceRecursive(user, orig_instr_shape, reduce_dims,
                                    visited_instrs);
    }
  }
  if (supported_ops.contains(instr->opcode())) {
    return FindAddReduceRecursive(instr->mutable_operand(0), orig_instr_shape,
                                  reduce_dims, visited_instrs);
  }
  return nullptr;
}
HloInstruction* FindAddReduce(HloInstruction* instr,
                              const absl::Span<const int64_t> reduce_dims) {
  absl::flat_hash_set<HloInstruction*> visited_instrs;
  return FindAddReduceRecursive(instr, instr->shape(), reduce_dims,
                                visited_instrs);
}
template <typename Pattern>
auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}
template <typename Pattern>
auto SupportedBitcastOrReshape(Pattern pattern) {
  auto supported_bitcast_or_reshape = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::Equal(
        ShapeUtil::DropDegenerateDimensions(instr->shape()),
        ShapeUtil::DropDegenerateDimensions(instr->operand(0)->shape()));
  };
  return m::AnyOf<HloInstruction>(
      m::Bitcast(pattern).WithPredicate(supported_bitcast_or_reshape),
      m::Reshape(pattern).WithPredicate(supported_bitcast_or_reshape));
}
template <typename Pattern>
auto OptionalSupportedTransform(Pattern pattern) {
  auto shared_subpattern = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(
      SupportedConvert(SupportedBitcastOrReshape(shared_subpattern)),
      SupportedBitcastOrReshape(SupportedConvert(shared_subpattern)),
      SupportedConvert(shared_subpattern),
      SupportedBitcastOrReshape(shared_subpattern), shared_subpattern);
}
template <typename Pattern>
auto BitcastOrReshape(Pattern pattern) {
  return OptionalSupportedTransform(
      m::AnyOf<HloInstruction>(m::Bitcast(pattern), m::Reshape(pattern)));
}
template <typename Pattern>
auto Transpose(Pattern pattern) {
  return OptionalSupportedTransform(m::Transpose(pattern));
}
template <typename Pattern>
auto Rsqrt(HloInstruction** rsqrt, Pattern pattern) {
  return OptionalSupportedTransform(m::Rsqrt(rsqrt, pattern));
}
template <typename Pattern0, typename Pattern1>
auto AddAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::AddAnyOrder(pattern0, pattern1));
}
template <typename Pattern0, typename Pattern1>
auto Subtract(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(pattern0, pattern1));
}
template <typename Pattern0, typename Pattern1>
auto Subtract(HloInstruction** subtract, Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(subtract, pattern0, pattern1));
}
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::MultiplyAnyOrder(pattern0, pattern1));
}
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(HloInstruction** multiply, Pattern0 pattern0,
                      Pattern1 pattern1) {
  return OptionalSupportedTransform(
      m::MultiplyAnyOrder(multiply, pattern0, pattern1));
}
template <typename Pattern>
auto Square(Pattern pattern) {
  return MultiplyAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}
template <typename Pattern>
auto Cube(Pattern pattern) {
  auto unique_cube = [](const HloInstruction* instr) -> bool {
    bool square_operand = instr->operand(0)->opcode() != HloOpcode::kMultiply;
    return instr->operand(!square_operand)->opcode() != HloOpcode::kMultiply &&
           instr->operand(square_operand)->operand(0) ==
               instr->operand(!square_operand);
  };
  return MultiplyAnyOrder(Square(pattern), pattern).WithPredicate(unique_cube);
}
template <typename Pattern>
auto AddReduce(Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}
template <typename Pattern>
auto AddReduce(HloInstruction** reduction, Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(reduction, pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}
template <typename Pattern>
auto NegateAddReduce(HloInstruction** reduction, Pattern pattern) {
  return m::AnyOf<HloInstruction>(AddReduce(reduction, m::Negate(pattern)),
                                  m::Negate(AddReduce(reduction, pattern)));
}
template <typename Pattern>
auto Expectation(Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}
template <typename Pattern>
auto Expectation(UniqueHloInstruction* expectation, Pattern pattern) {
  auto shared_subpattern = OptionalSupportedTransform(
      m::MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          })
          .WithPredicate(expectation->GetCaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}
template <typename Pattern>
auto Expectation(UniqueHloInstruction* expectation, HloInstruction** reduce,
                 Pattern pattern) {
  auto shared_subpattern = OptionalSupportedTransform(
      m::MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()),
                          AddReduce(reduce, pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          })
          .WithPredicate(expectation->GetCaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}
auto Variance(UniqueHloInstruction* variance, UniqueHloInstruction* expectation,
              UniqueHloInstruction* x) {
  return m::AnyOf<HloInstruction>(
      Subtract(
          Expectation(Square(OptionalSupportedTransform(
              m::Op().WithPredicate(x->GetCaptureOrVerifyFn())))),
          Square(Expectation(expectation,
                             OptionalSupportedTransform(m::Op().WithPredicate(
                                 x->GetCaptureOrVerifyFn())))))
          .WithPredicate(variance->GetCaptureOrVerifyFn()),
      Expectation(
          Square(Subtract(
              OptionalSupportedTransform(
                  m::Op().WithPredicate(x->GetCaptureOrVerifyFn())),
              Expectation(expectation,
                          OptionalSupportedTransform(m::Op().WithPredicate(
                              x->GetCaptureOrVerifyFn()))))))
          .WithPredicate(variance->GetCaptureOrVerifyFn()));
}
auto NormFactor(HloInstruction** norm_factor, UniqueHloInstruction* x,
                UniqueHloInstruction* variance,
                UniqueHloInstruction* expectation,
                UniqueHloInstruction* epsilon) {
  auto shared_subpattern = m::SharedSubpattern(Rsqrt(
      norm_factor, AddAnyOrder(Variance(variance, expectation, x),
                               m::Broadcast(m::ConstantScalar().WithPredicate(
                                   epsilon->GetCaptureOrVerifyFn())))));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}
template <typename P0, typename P1, typename P2>
auto MultiplyMultiplyAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, MultiplyAnyOrder(p1, p2)),
      MultiplyAnyOrder(p1, MultiplyAnyOrder(p0, p2)),
      MultiplyAnyOrder(p2, MultiplyAnyOrder(p0, p1)));
}
template <typename P0, typename P1, typename P2>
auto AddAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(p0, AddAnyOrder(p1, p2)),
                                  AddAnyOrder(p1, AddAnyOrder(p0, p2)),
                                  AddAnyOrder(p2, AddAnyOrder(p0, p1)));
}
template <typename P0, typename P1, typename P2>
auto MultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, AddAnyOrder(p1, p2)),
      AddAnyOrder(MultiplyAnyOrder(p0, p1), MultiplyAnyOrder(p0, p2)));
}
template <typename P0, typename P1, typename P2>
auto SubtractAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(Subtract(p0, p1), p2),
                                  AddAnyOrder(Subtract(p2, p1), p0),
                                  Subtract(AddAnyOrder(p0, p2), p1));
}
template <typename P0, typename P1, typename P2, typename P3, typename P4>
auto SubtractMultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2, P3 p3, P4 p4) {
  return m::AnyOf<HloInstruction>(
      SubtractAddAnyOrder(MultiplyMultiplyAnyOrder(p0, p2, p3),
                          MultiplyMultiplyAnyOrder(p1, p2, p3), p4),
      AddAnyOrder(MultiplyMultiplyAnyOrder(Subtract(p0, p1), p2, p3), p4));
}
auto FusedExpectation(UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
      m::CustomCall({kCudnnNormCallTarget})
          .WithPredicate(custom_call->GetCaptureOrVerifyFn()),
      1));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}
auto FusedExpectation(UniqueHloInstruction* fused_expectation,
                      UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(
          m::CustomCall({kCudnnNormCallTarget})
              .WithPredicate(custom_call->GetCaptureOrVerifyFn()),
          1)
          .WithPredicate(fused_expectation->GetCaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}
auto FusedNormFactor(UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
      m::CustomCall({kCudnnNormCallTarget})
          .WithPredicate(custom_call->GetCaptureOrVerifyFn()),
      2));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}
auto FusedNormFactor(UniqueHloInstruction* fused_norm_factor,
                     UniqueHloInstruction* custom_call) {
  auto shared_subpattern = m::SharedSubpattern(
      m::GetTupleElement(
          m::CustomCall({kCudnnNormCallTarget})
              .WithPredicate(custom_call->GetCaptureOrVerifyFn()),
          2)
          .WithPredicate(fused_norm_factor->GetCaptureOrVerifyFn()));
  return m::AnyOf<HloInstruction>(shared_subpattern,
                                  BitcastOrReshape(shared_subpattern));
}
auto DNormFactor(UniqueHloInstruction* custom_call) {
  return MultiplyAnyOrder(m::Broadcast(m::ConstantScalar(-0.5)),
                          Cube(FusedNormFactor(custom_call)));
}
auto XCenter(UniqueHloInstruction* x, UniqueHloInstruction* custom_call,
             const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_x =
      [x, custom_call, &norm_metadata](const HloInstruction* instr) -> bool {
    return x->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(0), norm_metadata)
            ? custom_call->Instr()->mutable_operand(0)
            : nullptr);
  };
  return Subtract(m::Op(), m::Broadcast(FusedExpectation(custom_call)))
      .WithPredicate(capture_or_verify_x);
}
auto XCenter(UniqueHloInstruction* x_center, UniqueHloInstruction* x,
             UniqueHloInstruction* fused_expectation,
             UniqueHloInstruction* custom_call,
             const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_x =
      [x, custom_call, &norm_metadata](const HloInstruction* instr) -> bool {
    return x->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(0), norm_metadata)
            ? custom_call->Instr()->mutable_operand(0)
            : nullptr);
  };
  return Subtract(m::Op(), m::Broadcast(FusedExpectation(fused_expectation,
                                                         custom_call)))
      .WithPredicate(x_center->GetCaptureOrVerifyFn())
      .WithPredicate(capture_or_verify_x);
}
auto F0(UniqueHloInstruction* custom_call, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, UniqueHloInstruction* x,
        HloInstruction** reduce, const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_scale = [scale, custom_call, &norm_metadata](
                                     const HloInstruction* instr) -> bool {
    return scale->CaptureOrVerify(FindTarget(custom_call->Instr(), instr,
                                             custom_call->Instr()->operand(1),
                                             norm_metadata)
                                      ? custom_call->Instr()->mutable_operand(1)
                                      : nullptr);
  };
  return AddReduce(
      reduce, MultiplyMultiplyAnyOrder(
                  XCenter(x, custom_call, norm_metadata),
                  m::Broadcast(m::Op().WithPredicate(capture_or_verify_scale)),
                  m::Op().WithPredicate(dy->GetCaptureOrVerifyFn())));
}
auto F1(UniqueHloInstruction* x, UniqueHloInstruction* x_center,
        UniqueHloInstruction* fused_expectation,
        UniqueHloInstruction* custom_call, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, HloInstruction** reduce,
        const NormMetadataMap& norm_metadata) {
  auto broadcasts_two_over_nelems = [](const HloInstruction* instr) -> bool {
    const HloInstruction* multiply = SkipUnaryOps(instr->operand(0));
    bool bcast_operand =
        multiply->operand(0)->opcode() != HloOpcode::kBroadcast;
    float actual_two_over_nelems = multiply->operand(bcast_operand)
                                       ->operand(0)
                                       ->literal()
                                       .GetAsDouble({})
                                       .value();
    int64_t nelems = 1;
    for (int i = 0; i < instr->shape().dimensions_size(); ++i) {
      if (!absl::c_linear_search(instr->dimensions(), i)) {
        nelems *= instr->shape().dimensions()[i];
      }
    }
    float two_over_nelems = 2. / static_cast<float>(nelems);
    float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
    return abs(actual_two_over_nelems - two_over_nelems) <
           ((actual_two_over_nelems + two_over_nelems) * numerical_epsilon);
  };
  return MultiplyAnyOrder(
      XCenter(x_center, x, fused_expectation, custom_call, norm_metadata),
      m::Broadcast(
          MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()),
                           MultiplyAnyOrder(DNormFactor(custom_call),
                                            F0(custom_call, scale, dy, x,
                                               reduce, norm_metadata))))
          .WithPredicate(broadcasts_two_over_nelems));
}
auto F2(UniqueHloInstruction* fused_norm_factor, UniqueHloInstruction* scale,
        UniqueHloInstruction* dy, UniqueHloInstruction* custom_call,
        const NormMetadataMap& norm_metadata) {
  auto capture_or_verify_scale = [scale, custom_call, &norm_metadata](
                                     const HloInstruction* instr) -> bool {
    return scale->CaptureOrVerify(
        FindTarget(custom_call->Instr(), instr->operand(0),
                   custom_call->Instr()->operand(1), norm_metadata)
            ? custom_call->Instr()->mutable_operand(1)
            : nullptr);
  };
  return MultiplyAnyOrder(
      m::Broadcast(
          BitcastOrReshape(FusedNormFactor(fused_norm_factor, custom_call))),
      MultiplyAnyOrder(m::Broadcast().WithPredicate(capture_or_verify_scale),
                       m::Op().WithPredicate(dy->GetCaptureOrVerifyFn())));
}
class CudnnNormRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNormRewriterVisitor(
      const se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}
  absl::Status HandleAdd(HloInstruction* instr) override {
    TF_RETURN_IF_ERROR(MatchLayerNorm(instr));
    TF_RETURN_IF_ERROR(MatchLayerNormGradient(instr));
    return absl::OkStatus();
  }
  absl::Status HandleSubtract(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }
  absl::Status MatchLayerNorm(HloInstruction* instr) {
    UniqueHloInstruction x, expectation, variance, epsilon;
    HloInstruction *scale, *bias, *reduce, *norm_factor, *broadcast_scale,
        *broadcast_bias;
    if (Match(
            instr,
            SubtractMultiplyAddAnyOrder(
                OptionalSupportedTransform(
                    m::Op().WithPredicate(x.GetCaptureOrVerifyFn())),
                Expectation(&expectation, &reduce,
                            OptionalSupportedTransform(m::Op().WithPredicate(
                                x.GetCaptureOrVerifyFn()))),
                NormFactor(&norm_factor, &x, &variance, &expectation, &epsilon),
                m::Broadcast(&broadcast_scale, m::Op(&scale)),
                m::Broadcast(&broadcast_bias, m::Op(&bias))))) {
#if CUDNN_VERSION < 8905
      VLOG(1) << "Layer norm Custom Calls require cuDNN 8.9.5.";
      return absl::OkStatus();
#endif  
      if (!instr->GetModule()
               ->config()
               .debug_options()
               .xla_gpu_enable_cudnn_layer_norm()) {
        VLOG(1) << "Layer norm Custom Calls disabled.";
        return absl::OkStatus();
      }
      if (cuda_compute_capability_.major != se::CudaComputeCapability::AMPERE &&
          cuda_compute_capability_.major != se::CudaComputeCapability::HOPPER) {
        VLOG(1) << "Layer norm Custom Calls require Ampere or Hopper "
                   "architectures.";
        return absl::OkStatus();
      }
      if (!x.Instr() || !expectation.Instr() || !variance.Instr() ||
          !epsilon.Instr()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }
      if (!LayoutUtil::IsMonotonicWithDim0Major(x.Instr()->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(scale->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(bias->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout())) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }
      if (!CompatibleElementType(instr) || !CompatibleElementType(scale) ||
          !CompatibleElementType(bias) ||
          !ShapeUtil::SameElementType(instr->shape(), x.Instr()->shape()) ||
          !ShapeUtil::Equal(scale->shape(), bias->shape())) {
        VLOG(1) << "Layer norm input types or shapes not supported.";
        return absl::OkStatus();
      }
      std::vector<int64_t> norm_dims(reduce->dimensions().begin(),
                                     reduce->dimensions().end());
      std::vector<int64_t> norm_dims_adjusted = AdjustedDimensions(reduce);
      if (norm_dims_adjusted.size() !=
          ShapeUtil::DropDegenerateDimensions(scale->shape())
              .dimensions_size()) {
        VLOG(1) << "Layer norm input dimensions not supported.";
        return absl::OkStatus();
      }
      if (!ShapeUtil::EqualIgnoringElementType(
              ShapeUtil::DropDegenerateDimensions(reduce->operand(0)->shape()),
              ShapeUtil::DropDegenerateDimensions(broadcast_scale->shape())) ||
          !ShapeUtil::EqualIgnoringElementType(
              ShapeUtil::DropDegenerateDimensions(reduce->operand(0)->shape()),
              ShapeUtil::DropDegenerateDimensions(broadcast_bias->shape())) ||
          norm_dims_adjusted != AdjustedDimensions(broadcast_scale) ||
          norm_dims_adjusted != AdjustedDimensions(broadcast_bias)) {
        VLOG(1) << "Layer norm operand broadcast not supported.";
        return absl::OkStatus();
      }
      std::vector<int64_t> non_norm_dims;
      for (int64_t x_dim = 0; x_dim < x.Instr()->shape().rank(); ++x_dim) {
        if (std::find(norm_dims.begin(), norm_dims.end(), x_dim) ==
            norm_dims.end()) {
          non_norm_dims.emplace_back(x_dim);
        }
      }
      std::vector<int64_t> non_norm_dims_adjusted =
          AdjustedDimensions(x.Instr()->shape(), non_norm_dims);
      std::vector<int64_t> x_transpose_order = non_norm_dims;
      x_transpose_order.insert(x_transpose_order.end(), norm_dims.begin(),
                               norm_dims.end());
      bool apply_transpose = false;
      for (int i = 0; i < x_transpose_order.size(); ++i) {
        if (x_transpose_order[i] != i) {
          apply_transpose = true;
          break;
        }
      }
      std::optional<HloInstruction*> x_transpose;
      std::vector<int64_t> y_transpose_order(x_transpose_order.size());
      if (apply_transpose) {
        for (int k = 0; k < x_transpose_order.size(); ++k) {
          y_transpose_order[x_transpose_order[k]] = k;
        }
        TF_ASSIGN_OR_RETURN(x_transpose,
                            MakeTransposeHlo(x.Instr(), x_transpose_order));
      }
      std::vector<int64_t> reshaped_dims = {1};
      for (auto non_norm_dim : non_norm_dims) {
        reshaped_dims[0] *= x.Instr()->shape().dimensions(non_norm_dim);
      }
      for (auto norm_dim : norm_dims) {
        reshaped_dims.emplace_back(x.Instr()->shape().dimensions(norm_dim));
      }
      while (reshaped_dims.size() < 4) {
        reshaped_dims.emplace_back(1);
      }
      Shape reshaped_shape = ShapeUtil::MakeShape(
          x.Instr()->shape().element_type(), reshaped_dims);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * x_reshape,
          MakeReshapeHlo(reshaped_shape, x_transpose.value_or(x.Instr())));
      std::vector<int64_t> reshaped_scale_dims = reshaped_dims;
      reshaped_scale_dims[0] = 1;
      Shape scale_bias_shape = ShapeUtil::MakeShape(
          scale->shape().element_type(), reshaped_scale_dims);
      TF_ASSIGN_OR_RETURN(HloInstruction * scale_reshape,
                          MakeReshapeHlo(scale_bias_shape, scale));
      TF_ASSIGN_OR_RETURN(HloInstruction * bias_reshape,
                          MakeReshapeHlo(scale_bias_shape, bias));
      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_epsilon(
          epsilon.Instr()->literal().GetAsDouble({}).value());
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_INFER);
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      algorithm->set_is_cudnn_frontend(true);
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size =
          (2 * c_constant * (4 + 256)) + (2 * reshaped_dims[0] * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {x_reshape->shape(), ShapeUtil::MakeShape(U8, {workspace_size})});
      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape, {x_reshape, scale_reshape, bias_reshape},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(custom_call, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * y_reshape,
          MakeReshapeHlo(x_transpose.value_or(instr)->shape(), gte));
      std::optional<HloInstruction*> y_transpose;
      if (apply_transpose) {
        TF_ASSIGN_OR_RETURN(y_transpose,
                            MakeTransposeHlo(y_reshape, y_transpose_order));
      }
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, y_transpose.value_or(y_reshape)));
      norm_metadata_.insert(
          {custom_call,
           NormMetadata({x_transpose.value_or(nullptr),
                         y_transpose.value_or(nullptr), norm_dims_adjusted,
                         non_norm_dims_adjusted})});
      VLOG(1) << "Layer norm rewritten into Custom Call.";
      for (HloInstruction* user : norm_factor->users()) {
        if (user->opcode() == HloOpcode::kDivide &&
            user->operand_index(norm_factor) == 0) {
          TF_ASSIGN_OR_RETURN(bool changed,
                              MatchNormFactor(user, custom_call, variance,
                                              expectation, epsilon));
          if (changed) {
            break;
          }
        }
      }
    }
    return absl::OkStatus();
  }
  absl::StatusOr<bool> MatchNormFactor(HloInstruction* instr,
                                       HloInstruction* custom_call,
                                       UniqueHloInstruction& variance,
                                       UniqueHloInstruction& expectation,
                                       UniqueHloInstruction& epsilon) {
    HloInstruction* gte = custom_call->users()[0];
    if (Match(instr,
              m::Divide(
                  m::Op(),
                  AddAnyOrder(
                      m::Op().WithPredicate(variance.GetCaptureOrVerifyFn()),
                      m::Broadcast(m::ConstantScalar().WithPredicate(
                          epsilon.GetCaptureOrVerifyFn())))))) {
      if (!variance.Instr() || !epsilon.Instr()) {
        VLOG(1) << "Layer norm operands not unique.";
        return false;
      }
      if (!CompatibleElementType(instr) ||
          !CompatibleElementType(expectation.Instr())) {
        VLOG(1) << "Layer norm input types not compatible.";
        return false;
      }
      auto norm_metadata = norm_metadata_.extract(custom_call);
      if (!norm_metadata) {
        VLOG(1) << "Unable to retrieve norm metadata of forward Custom Call.";
        return false;
      }
      auto make_compatible_shape = [](Shape shape) -> Shape {
        return ShapeUtil::MakeShape(shape.element_type(),
                                    {ShapeUtil::ElementsIn(shape), 1, 1, 1});
      };
      Shape expectation_shape =
          make_compatible_shape(expectation.Instr()->shape());
      Shape norm_factor_shape = make_compatible_shape(instr->shape());
      std::vector<Shape> tuple_shapes = custom_call->shape().tuple_shapes();
      tuple_shapes.insert(tuple_shapes.begin() + 1,
                          {expectation_shape, norm_factor_shape});
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);
      HloInstruction* new_custom_call = instr->AddInstruction(
          custom_call->CloneWithNewShape(custom_call_shape));
      TF_ASSIGN_OR_RETURN(
          GpuBackendConfig gpu_backend_config,
          custom_call->backend_config<xla::gpu::GpuBackendConfig>());
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_TRAIN);
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) + 32;
      backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
          workspace_size);
      TF_RETURN_IF_ERROR(
          new_custom_call->set_backend_config(gpu_backend_config));
      auto replace_with_new_cc = [new_custom_call, this](
                                     HloInstruction* old_instr,
                                     int tuple_index) -> absl::Status {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * new_gte,
            MakeGetTupleElementHlo(new_custom_call, tuple_index));
        HloInstruction* new_instr = new_gte;
        if (!ShapeUtil::Equal(new_gte->shape(), old_instr->shape())) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), new_gte));
        }
        if (old_instr->opcode() != HloOpcode::kDivide) {
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        } else {
          TF_RETURN_IF_ERROR(
              ReplaceInstruction(old_instr->mutable_operand(0), new_instr));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply0,
              MakeBinaryHlo(HloOpcode::kMultiply, new_instr, new_instr));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply1,
              MakeBinaryHlo(HloOpcode::kMultiply, new_multiply0, new_instr));
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_multiply1));
        }
        return absl::OkStatus();
      };
      TF_RETURN_IF_ERROR(replace_with_new_cc(gte, 0));
      TF_RETURN_IF_ERROR(replace_with_new_cc(expectation.Instr(), 1));
      TF_RETURN_IF_ERROR(replace_with_new_cc(instr, 2));
      norm_metadata.key() = new_custom_call;
      norm_metadata_.insert(std::move(norm_metadata));
      VLOG(1)
          << "Expectation and norm factor fused into layer norm Custom Call.";
    }
    return true;
  }
  absl::Status MatchLayerNormGradient(HloInstruction* instr) {
    UniqueHloInstruction fwd_custom_call, x, x_center, scale, dy,
        fused_expectation, fused_norm_factor;
    HloInstruction *broadcast, *scalar, *dscale, *dbias, *reduce0, *reduce1,
        *reduce2, *reduce3;
    if (Match(instr,
              AddAddAnyOrder(
                  m::Broadcast(
                      &broadcast,
                      MultiplyAddAnyOrder(
                          m::Broadcast(m::ConstantScalar(&scalar)),
                          NegateAddReduce(&reduce0,
                                          F1(&x, &x_center, &fused_expectation,
                                             &fwd_custom_call, &scale, &dy,
                                             &reduce2, norm_metadata_)),
                          NegateAddReduce(
                              &reduce1, F2(&fused_norm_factor, &scale, &dy,
                                           &fwd_custom_call, norm_metadata_)))),
                  F2(&fused_norm_factor, &scale, &dy, &fwd_custom_call,
                     norm_metadata_),
                  F1(&x, &x_center, &fused_expectation, &fwd_custom_call,
                     &scale, &dy, &reduce3, norm_metadata_)))) {
      if (instr->user_count() == 1 &&
          instr->users()[0]->opcode() == HloOpcode::kConvert &&
          CompatibleElementType(instr->users()[0])) {
        instr = instr->users()[0];
      }
      if (!fwd_custom_call.Instr() || !x.Instr() || !dy.Instr() ||
          !x_center.Instr() || !scale.Instr() || !fused_expectation.Instr() ||
          !fused_norm_factor.Instr()) {
        VLOG(1) << "Layer norm gradient inputs not unique.";
        return absl::OkStatus();
      }
      auto norm_metadata = norm_metadata_.find(fwd_custom_call.Instr());
      if (norm_metadata == norm_metadata_.end()) {
        VLOG(1) << "Unable to retrieve norm metadata of forward Custom Call.";
        return absl::OkStatus();
      }
      if (AdjustedDimensions(reduce0) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce1) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce2) !=
              norm_metadata->second.norm_dims_adjusted ||
          AdjustedDimensions(reduce3) !=
              norm_metadata->second.norm_dims_adjusted) {
        VLOG(1) << "Unexpected reductions dimensions in layer norm gradient.";
        return absl::OkStatus();
      }
      float actual_r_nelems = scalar->literal().GetAsDouble({}).value();
      int64_t nelems = 1;
      for (int i = 0; i < broadcast->shape().dimensions_size(); ++i) {
        if (!absl::c_linear_search(broadcast->dimensions(), i)) {
          nelems *= broadcast->shape().dimensions()[i];
        }
      }
      float r_nelems = 1. / static_cast<float>(nelems);
      float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
      if (!(abs(actual_r_nelems - r_nelems) <
            ((actual_r_nelems + r_nelems) * numerical_epsilon))) {
        VLOG(1)
            << "Layer norm backward broadcast operand outside expected range.";
        return absl::OkStatus();
      }
      auto find_dscale =
          [&fused_norm_factor, &norm_metadata](
              const UniqueHloInstruction& factor0,
              const UniqueHloInstruction& factor1) -> HloInstruction* {
        for (HloInstruction* factor0_user : factor0.Instr()->users()) {
          std::vector<HloInstruction*> users;
          SkipUnaryOpsTopDownRecursive(factor0_user, users);
          for (HloInstruction* user : users) {
            if (Match(user,
                      MultiplyAnyOrder(
                          m::Op(), MultiplyAnyOrder(
                                       m::Broadcast(BitcastOrReshape(m::Op().Is(
                                           fused_norm_factor.Instr()))),
                                       m::Op().Is(factor1.Instr()))))) {
              for (HloInstruction* multiply_user : user->users()) {
                if (AppliesAddReduce(
                        multiply_user,
                        norm_metadata->second.non_norm_dims_adjusted)) {
                  return multiply_user;
                }
              }
            }
          }
        }
        return nullptr;
      };
      if (!(dscale = find_dscale(x_center, dy)) &&
          !(dscale = find_dscale(dy, x_center))) {
        VLOG(1) << "Unable to identify Dscale in graph.";
        return absl::OkStatus();
      }
      dbias = FindAddReduce(dy.Instr(),
                            norm_metadata->second.non_norm_dims_adjusted);
      if (!LayoutUtil::IsMonotonicWithDim0Major(dy.Instr()->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(dscale->shape().layout()) ||
          (dbias &&
           !LayoutUtil::IsMonotonicWithDim0Major(dbias->shape().layout()))) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }
      if (x.Instr()->shape().element_type() != instr->shape().element_type()) {
        VLOG(1) << "The types of X and DX must match.";
        return absl::OkStatus();
      }
      if (!ShapeUtil::Equal(
              ShapeUtil::DropDegenerateDimensions(scale.Instr()->shape()),
              ShapeUtil::DropDegenerateDimensions(dscale->shape())) ||
          (dbias &&
           !ShapeUtil::Equal(
               ShapeUtil::DropDegenerateDimensions(scale.Instr()->shape()),
               ShapeUtil::DropDegenerateDimensions(dbias->shape())))) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }
      if (!CompatibleElementType(dy.Instr())) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }
      if (ShapeUtil::ByteSizeOfPrimitiveType(
              x.Instr()->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(
                  dy.Instr()->shape().element_type()) ||
          ShapeUtil::ByteSizeOfPrimitiveType(
              x.Instr()->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(
                  scale.Instr()->shape().element_type())) {
        VLOG(1) << "Backward layer norm types not supported.";
        return absl::OkStatus();
      }
      HloInstruction* transposed_dy = dy.Instr();
      if (norm_metadata->second.x_transpose) {
        TF_ASSIGN_OR_RETURN(
            transposed_dy,
            MakeTransposeHlo(dy.Instr(),
                             norm_metadata->second.x_transpose->dimensions()));
      }
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_dy,
                          MakeReshapeHlo(x.Instr()->shape(), transposed_dy));
      Shape dx_shape = ShapeUtil::MakeShape(instr->shape().element_type(),
                                            x.Instr()->shape().dimensions());
      Shape dscale_dbias_shape = ShapeUtil::MakeShape(
          dscale->shape().element_type(), scale.Instr()->shape().dimensions());
      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_BWD);
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      algorithm->set_is_cudnn_frontend(true);
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size =
          (2 * c_constant * (4 + 256)) +
          (2 * x.Instr()->shape().dimensions(0) * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {dx_shape, dscale_dbias_shape, dscale_dbias_shape,
           ShapeUtil::MakeShape(U8, {workspace_size})});
      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape,
              {x.Instr(), scale.Instr(), reshaped_dy, fused_expectation.Instr(),
               fused_norm_factor.Instr()},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
      auto replace_with_cc = [custom_call, norm_metadata, transposed_dy, this](
                                 HloInstruction* old_instr,
                                 int tuple_index) -> absl::Status {
        TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                            MakeGetTupleElementHlo(custom_call, tuple_index));
        HloInstruction* new_instr;
        if (tuple_index == 0 && norm_metadata->second.y_transpose) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(transposed_dy->shape(), gte));
          TF_ASSIGN_OR_RETURN(
              new_instr,
              MakeTransposeHlo(
                  new_instr, norm_metadata->second.y_transpose->dimensions()));
        } else {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), gte));
        }
        TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        return absl::OkStatus();
      };
      TF_RETURN_IF_ERROR(replace_with_cc(instr, 0));
      TF_RETURN_IF_ERROR(replace_with_cc(dscale, 1));
      if (dbias) {
        TF_RETURN_IF_ERROR(replace_with_cc(dbias, 2));
      }
      VLOG(1) << "Gradients w.r.t. x"
              << (dbias ? ", scale and bias" : " and scale")
              << " rewritten into layer norm backward Custom Call.";
    }
    return absl::OkStatus();
  }
 private:
  se::CudaComputeCapability cuda_compute_capability_;
  NormMetadataMap norm_metadata_;
};
absl::StatusOr<bool> RunOnComputation(
    HloComputation* computation,
    se::CudaComputeCapability cuda_compute_capability) {
  CudnnNormRewriterVisitor visitor(cuda_compute_capability);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}
}  
CudnnNormRewriter::CudnnNormRewriter(
    se::CudaComputeCapability cuda_compute_capability)
    : cuda_compute_capability_(cuda_compute_capability) {}
absl::StatusOr<bool> CudnnNormRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}
}  
}  