#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
namespace m = match;
bool IsConvCustomCall(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCustomCall &&
         (instr->custom_call_target() == kCudnnConvForwardCallTarget ||
          instr->custom_call_target() ==
              kCudnnConvBiasActivationForwardCallTarget);
}
bool IsConvDepthwise(const HloInstruction* instr) {
  int64_t feature_group_count = instr->feature_group_count();
  if (feature_group_count == 1) {
    return false;
  }
  const HloInstruction* input = instr->operand(0);
  int64_t input_feature_dimension =
      instr->convolution_dimension_numbers().input_feature_dimension();
  int64_t input_feature_count =
      input->shape().dimensions(input_feature_dimension);
  return input_feature_count == feature_group_count;
}
bool IsNonDepthwiseConvCustomCall(const HloInstruction* instr) {
  return IsConvCustomCall(instr) && !IsConvDepthwise(instr);
}
bool IsROCm(se::GpuComputeCapability cc) {
  return std::holds_alternative<se::RocmComputeCapability>(cc);
}
bool ShouldUseCudnnRuntimeFusion(const DebugOptions& debug_opts,
                                 se::GpuComputeCapability cc) {
  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&cc);
  if (cuda_cc != nullptr)
    return debug_opts.xla_gpu_use_runtime_fusion() && cuda_cc->IsAtLeast(7, 5);
  else
    return true;
}
bool IsSuitableForCudnnRuntimeFusion(HloInstruction* conv) {
  if (conv->operands().size() > 3) {
    return false;
  }
  if (conv->operand(0)->shape().element_type() != F16) {
    return false;
  }
  const Shape& shape = conv->operand(1)->shape();
  int64_t num_input_features = shape.dimensions(
      conv->convolution_dimension_numbers().kernel_input_feature_dimension());
  int64_t num_output_features = shape.dimensions(
      conv->convolution_dimension_numbers().kernel_output_feature_dimension());
  if (num_input_features % 2 != 0 || num_output_features % 2 != 0) {
    return false;
  }
  return true;
}
bool IsLosslesslyConvertibleTo(const HloInstruction* instr,
                               PrimitiveType dst_ty) {
  if (instr->shape().element_type() == dst_ty) {
    return true;
  }
  if (Match(instr, m::Convert(m::Op().WithElementType(dst_ty)))) {
    return primitive_util::CastPreservesValues(dst_ty,
                                               instr->shape().element_type());
  }
  if (instr->opcode() == HloOpcode::kConstant) {
    if (!instr->shape().IsArray()) {
      return false;
    }
    PrimitiveType orig_ty = instr->shape().element_type();
    absl::StatusOr<Literal> converted1 = instr->literal().Convert(dst_ty);
    if (!converted1.ok()) {
      return false;
    }
    absl::StatusOr<Literal> converted2 = converted1->Convert(orig_ty);
    if (!converted2.ok()) {
      return false;
    }
    return instr->literal() == *converted2;
  }
  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kReshape ||
      instr->opcode() == HloOpcode::kTranspose) {
    return IsLosslesslyConvertibleTo(instr->operand(0), dst_ty);
  }
  return false;
}
bool IsLosslesslyConvertibleToS8(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, S8);
}
bool IsLosslesslyConvertibleToF16(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, F16);
}
absl::StatusOr<HloInstruction*> EnsureIsConvBiasActivation(
    HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kCustomCall);
  if (conv->custom_call_target() == kCudnnConvBiasActivationForwardCallTarget) {
    return conv;
  }
  if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
    HloComputation* comp = conv->parent();
    const Shape& shape = conv->shape().tuple_shapes(0);
    int64_t num_output_features = shape.dimensions(
        conv->convolution_dimension_numbers().output_feature_dimension());
    PrimitiveType bias_ty;
    if (primitive_util::IsIntegralType(shape.element_type())) {
      bias_ty = F32;
    } else {
      bias_ty = shape.element_type();
    }
    auto bias = BroadcastZeros(comp, bias_ty, {num_output_features});
    absl::InlinedVector<HloInstruction*, 3> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands.push_back(bias);
    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    new_conv->set_custom_call_target(kCudnnConvBiasActivationForwardCallTarget);
    comp->parent()->SetAndUniquifyInstrName(new_conv,
                                            "cudnn-conv-bias-activation");
    return new_conv;
  }
  return FailedPrecondition("Unsupported conv: %s", conv->ToString());
}
absl::StatusOr<bool> FuseConvertTypeIntoConv(HloComputation* comp,
                                             PrimitiveType conv_type,
                                             PrimitiveType cvt_type) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    auto tuple_elem =
        m::GetTupleElement(m::Op(&conv).WithPredicate(IsConvCustomCall), 0)
            .WithElementType(conv_type);
    auto pattern =
        m::Convert(tuple_elem.WithOneUser()).WithElementType(cvt_type);
    if (!Match(instr, pattern)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertTypeIntoConv: ", conv->ToString());
        })) {
      continue;
    }
    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(cvt_type);
    HloInstruction* new_conv =
        comp->AddInstruction(conv->CloneWithNewShape(new_shape));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_gte));
    changed = true;
  }
  return changed;
}
struct ConvConvertTypes {
  PrimitiveType convolution_type;
  PrimitiveType conversion_type;
};
absl::StatusOr<bool> FuseRemoveConvertInConv(HloComputation* comp) {
  bool changed = false;
  std::array<ConvConvertTypes, 3> types{{
      {S32, F32},
      {S8, F32},
      {F32, S8},
  }};
  for (auto [conv_type, cvt_type] : types) {
    TF_ASSIGN_OR_RETURN(bool curr_change,
                        FuseConvertTypeIntoConv(comp, conv_type, cvt_type));
    changed |= curr_change;
  }
  return changed;
}
absl::StatusOr<bool> FuseConvAlpha(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* alpha = nullptr;
    auto pattern = m::MultiplyAnyOrder(
        m::GetTupleElement(
            &gte, m::Op(&conv).WithPredicate(IsNonDepthwiseConvCustomCall), 0)
            .WithOneUse(),
        m::Broadcast(m::ConstantEffectiveScalar(&alpha)));
    if (!Match(instr, pattern)) {
      continue;
    }
    PrimitiveType alpha_ty = gte->shape().element_type() == F64 ? F64 : F32;
    if (!IsLosslesslyConvertibleTo(alpha, alpha_ty)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.conv_result_scale() != 1) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvAlpha: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_conv_result_scale(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}
class GraphString {
 public:
  GraphString() = default;
  bool AppendOp(std::string op_name, HloInstruction* op,
                std::vector<HloInstruction*> operands = {}) {
    std::optional<int64_t> operand_uid;
    int num_operands_in_graph = 0;
    for (HloInstruction* operand : operands) {
      if (OpInGraph(operand->unique_id())) {
        num_operands_in_graph++;
        if (num_operands_in_graph > 1) {
          return false;
        }
        operand_uid = operand->unique_id();
      }
    }
    graph_.emplace_back(OpDescriptor(
        {op->unique_id(), op->shape().element_type(), op_name, operand_uid}));
    return true;
  }
  void ChangeDataType(PrimitiveType type) {
    DCHECK(!graph_.empty());
    graph_.back().output_type = type;
  }
  std::string Graph() const {
    std::string graph;
    for (OpDescriptor op : graph_) {
      graph.append(std::to_string(op.uid));
      graph.append(":[" +
                   primitive_util::LowercasePrimitiveTypeName(op.output_type) +
                   "]");
      graph.append(op.name);
      graph.append("(");
      if (op.operand.has_value()) {
        graph.append(std::to_string(*op.operand));
      }
      graph.append(");");
    }
    return graph;
  }
  bool OpInGraph(int64_t uid, std::string op_name = "") const {
    auto op_filter = [&](OpDescriptor op) -> bool {
      if (op_name.empty()) {
        return op.uid == uid;
      } else {
        return op.uid == uid && op.name == op_name;
      }
    };
    return std::find_if(graph_.begin(), graph_.end(), op_filter) !=
           graph_.end();
  }
 private:
  struct OpDescriptor {
    int64_t uid;
    PrimitiveType output_type;
    std::string name;
    std::optional<int64_t> operand;
  };
  std::vector<OpDescriptor> graph_;
};
bool IsF8Type(const HloInstruction* instr) {
  return primitive_util::IsF8Type(instr->shape().element_type());
}
bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsScalar(instr->shape());
}
std::optional<PrimitiveType> IsSaturatingCastToF8(HloInstruction* instr) {
  HloInstruction *op, *clamp_lower, *clamp_upper;
  if (Match(instr,
            m::Convert(
                &op,
                m::Clamp(m::Broadcast(m::ConstantScalar(&clamp_lower)), m::Op(),
                         m::Broadcast(m::ConstantScalar(&clamp_upper))))) &&
      ((op->shape().element_type() == F8E4M3FN &&
        clamp_lower->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e4m3fn>::lowest())) &&
        clamp_upper->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e4m3fn>::max()))) ||
       (op->shape().element_type() == F8E5M2 &&
        clamp_lower->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e5m2>::lowest())) &&
        clamp_upper->literal().IsAllFloat(static_cast<float>(
            std::numeric_limits<tsl::float8_e5m2>::max()))))) {
    return op->shape().element_type();
  }
  return std::nullopt;
}
bool AppliesMaxReduce(HloInstruction* op) {
  HloComputation* reduce_comp = op->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return ShapeUtil::IsScalar(op->shape()) &&
         ShapeUtil::IsScalar(op->operand(1)->shape()) &&
         op->operand(1)->IsConstant() &&
         op->operand(1)->literal().GetAsDouble({}) <= 0. &&
         reduce_comp_root->opcode() == HloOpcode::kMaximum &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}
void CaptureConvGraphRecursive(HloInstruction* instr,
                               std::vector<HloInstruction*>& operands,
                               std::vector<HloInstruction*>& aux_outputs,
                               GraphString& graph_string,
                               absl::flat_hash_set<int>& visited_instrs,
                               HloInstruction*& final_instr) {
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return;
  }
  final_instr = instr;
  GraphString init_graph_string = graph_string;
  std::vector<HloInstruction*> init_operands = operands,
                               init_aux_outputs = aux_outputs;
  int num_linear_users = 0, num_nonlinear_users = 0;
  for (HloInstruction* user : instr->users()) {
    HloInstruction *op, *operand0, *operand1;
    if (Match(user, m::AddAnyOrder(&op, m::Op(&operand0), m::Op(&operand1)))) {
      if (graph_string.AppendOp("add", op, {operand0, operand1})) {
        operands.push_back(operand0 == instr ? operand1 : operand0);
        num_linear_users++;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr);
      }
      continue;
    }
    if (Match(user, m::MultiplyAnyOrder(&op, m::Op(&operand0),
                                        m::Broadcast(m::Op(&operand1)))) &&
        ShapeUtil::IsScalar(operand1->shape())) {
      if (graph_string.AppendOp("scale", op, {operand0, operand1})) {
        operands.push_back(operand1);
        num_linear_users++;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr);
      }
      continue;
    }
    if (Match(user, m::Divide(&op, m::Op(&operand0),
                              m::Broadcast(m::Op(&operand1)))) &&
        ShapeUtil::IsScalar(operand1->shape())) {
      if (graph_string.AppendOp("invscale", op, {operand0, operand1})) {
        operands.push_back(operand1);
        num_linear_users++;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr);
      }
      continue;
    }
    if (Match(user, m::MaximumAnyOrder(&op, m::Op(&operand0),
                                       m::Broadcast(m::ConstantScalar(0))))) {
      if (graph_string.AppendOp("relu", op, {operand0})) {
        num_linear_users++;
        CaptureConvGraphRecursive(user, operands, aux_outputs, graph_string,
                                  visited_instrs, final_instr);
      }
      continue;
    }
    if (Match(user, m::Reduce(&op, m::Op(&operand0), m::Op())) &&
        graph_string.OpInGraph(operand0->unique_id(), "relu") &&
        AppliesMaxReduce(op)) {
      if (graph_string.AppendOp("amax", op, {operand0})) {
        aux_outputs.emplace_back(op);
        num_nonlinear_users++;
      }
      continue;
    }
    if (!user->users().empty()) {
      HloInstruction* users_user = user->users()[0];
      std::optional<PrimitiveType> f8_type = IsSaturatingCastToF8(users_user);
      if (f8_type.has_value()) {
        graph_string.ChangeDataType(f8_type.value());
        num_linear_users++;
        CaptureConvGraphRecursive(users_user, operands, aux_outputs,
                                  graph_string, visited_instrs, final_instr);
        continue;
      }
      if (Match(users_user,
                m::Reduce(&op, m::Abs(m::Op(&operand0)), m::Op())) &&
          AppliesMaxReduce(op)) {
        if (graph_string.AppendOp("amax", op, {operand0})) {
          aux_outputs.emplace_back(op);
          num_nonlinear_users++;
        }
        continue;
      }
    }
  }
  if (num_linear_users > 1 || num_nonlinear_users > 1 ||
      num_linear_users + num_nonlinear_users < instr->user_count()) {
    graph_string = init_graph_string;
    operands = init_operands;
    aux_outputs = init_aux_outputs;
    final_instr = instr;
  }
}
absl::StatusOr<
    std::tuple<std::vector<HloInstruction*>, std::vector<HloInstruction*>,
               GraphString, HloInstruction*>>
CaptureConvGraph(HloInstruction* instr, HloInstruction* convolution,
                 HloInstruction* wide_input, HloInstruction* wide_filter,
                 HloInstruction* input_scale, HloInstruction* filter_scale,
                 bool x_mult_scale, bool w_mult_scale) {
  GraphString graph_string;
  graph_string.AppendOp("conv", instr);
  HloInstruction *input_scaled_conv, *filter_scaled_conv;
  if (input_scale) {
    TF_RETURN_IF_ERROR(convolution->ReplaceOperandWith(0, wide_input));
    HloInstruction* bcast_input_scale = instr->AddInstruction(
        HloInstruction::CreateBroadcast(instr->shape(), input_scale, {}));
    input_scaled_conv = instr->AddInstruction(HloInstruction::CreateBinary(
        instr->shape(),
        x_mult_scale ? HloOpcode::kMultiply : HloOpcode::kDivide, instr,
        bcast_input_scale));
    TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(input_scaled_conv));
  }
  if (filter_scale) {
    TF_RETURN_IF_ERROR(convolution->ReplaceOperandWith(1, wide_filter));
    HloInstruction* bcast_filter_scale = instr->AddInstruction(
        HloInstruction::CreateBroadcast(instr->shape(), filter_scale, {}));
    filter_scaled_conv = instr->AddInstruction(HloInstruction::CreateBinary(
        instr->shape(),
        w_mult_scale ? HloOpcode::kMultiply : HloOpcode::kDivide,
        input_scale ? input_scaled_conv : instr, bcast_filter_scale));
    TF_RETURN_IF_ERROR((input_scale ? input_scaled_conv : instr)
                           ->ReplaceAllUsesWith(filter_scaled_conv));
  }
  std::vector<HloInstruction*> operands, aux_outputs;
  absl::flat_hash_set<int> visited_instrs;
  HloInstruction* final_instr;
  CaptureConvGraphRecursive(instr, operands, aux_outputs, graph_string,
                            visited_instrs, final_instr);
  return std::make_tuple(operands, aux_outputs, graph_string, final_instr);
}
absl::StatusOr<bool> F8GraphConv(HloComputation* comp,
                                 se::CudaComputeCapability cc,
                                 se::dnn::VersionInfo dnn_version,
                                 const se::SemanticVersion& toolkit_version) {
  bool changed = false;
  if (dnn_version < se::dnn::VersionInfo(8, 9, 0)) {
    return false;
  }
  if (toolkit_version < se::SemanticVersion{12, 0, 0}) {
    return false;
  }
  if (!cc.IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    return false;
  }
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *convolution, *gte, *input, *filter,
        *input_scale = nullptr, *filter_scale = nullptr,
        *input_scale_op = nullptr, *filter_scale_op = nullptr,
        *wide_input = nullptr, *wide_filter = nullptr;
    auto conv_operand_maybe_scaled = [](HloInstruction** operand,
                                        HloInstruction** wide_operand,
                                        HloInstruction** scale_op,
                                        HloInstruction** scale) {
      return m::AnyOf<HloInstruction>(
          m::Op(operand).WithPredicate(IsF8Type),
          m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
          m::Divide(
              scale_op,
              m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
              m::Broadcast(m::Op(scale).WithPredicate(IsScalar))),
          m::MultiplyAnyOrder(
              scale_op,
              m::Convert(wide_operand, m::Op(operand).WithPredicate(IsF8Type)),
              m::Broadcast(m::Op(scale).WithPredicate(IsScalar))));
    };
    auto pattern = m::GetTupleElement(
        &gte,
        m::CustomCall(
            &convolution,
            conv_operand_maybe_scaled(&input, &wide_input, &input_scale_op,
                                      &input_scale),
            conv_operand_maybe_scaled(&filter, &wide_filter, &filter_scale_op,
                                      &filter_scale))
            .WithPredicate(IsConvCustomCall),
        0);
    if (Match(instr, pattern)) {
      if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
            return absl::StrCat("F8GraphConv: ", convolution->ToString());
          })) {
        continue;
      }
      std::vector<HloInstruction*> operands, aux_outputs;
      GraphString graph_string;
      HloInstruction* final_instr;
      TF_ASSIGN_OR_RETURN(
          std::tie(operands, aux_outputs, graph_string, final_instr),
          CaptureConvGraph(
              instr, convolution, wide_input, wide_filter, input_scale,
              filter_scale,
              input_scale_op ? input_scale_op->opcode() == HloOpcode::kMultiply
                             : false,
              filter_scale_op
                  ? filter_scale_op->opcode() == HloOpcode::kMultiply
                  : false));
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          convolution->backend_config<GpuBackendConfig>());
      CudnnConvBackendConfig& config =
          *gpu_config.mutable_cudnn_conv_backend_config();
      config.set_serialized_graph(graph_string.Graph());
      operands.insert(operands.begin(), input);
      operands.insert(operands.begin() + 1, filter);
      std::vector<Shape> output_shapes;
      output_shapes.emplace_back(ShapeUtil::ChangeElementType(
          ShapeUtil::GetTupleElementShape(convolution->shape(), 0),
          final_instr->shape().element_type()));
      for (HloInstruction* aux_output : aux_outputs) {
        output_shapes.emplace_back(aux_output->shape());
      }
      output_shapes.emplace_back(
          ShapeUtil::GetTupleElementShape(convolution->shape(), 1));
      HloInstruction* new_convolution =
          comp->AddInstruction(convolution->CloneWithNewOperands(
              ShapeUtil::MakeTupleShape(output_shapes), operands));
      new_convolution->set_custom_call_target(kCudnnConvForwardGraphCallTarget);
      TF_RETURN_IF_ERROR(new_convolution->set_backend_config(gpu_config));
      TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                          MakeGetTupleElementHlo(new_convolution, 0));
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(final_instr, new_gte));
      for (int i = 0; i < aux_outputs.size(); ++i) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                            MakeGetTupleElementHlo(new_convolution, i + 1));
        TF_RETURN_IF_ERROR(comp->ReplaceInstruction(aux_outputs[i], new_gte));
      }
      changed = true;
    }
  }
  return changed;
}
absl::StatusOr<bool> FuseBiasOrSideInput(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* addend = nullptr;
    auto pattern = m::AddAnyOrder(
        m::GetTupleElement(&gte,
                           m::Op(&conv)
                               .WithPredicate(IsNonDepthwiseConvCustomCall)
                               .WithOneUse(),
                           0)
            .WithOneUse(),
        m::Op(&addend));
    if (!Match(instr, pattern)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseBiasOrSideInput: ", conv->ToString());
        })) {
      continue;
    }
    if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
      TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }
    bool can_accept_bias =
        Match(conv->operand(2), m::Broadcast(m::ConstantEffectiveScalar(0)));
    bool can_accept_side_input = conv->operand_count() < 4;
    PrimitiveType conv_ty = gte->shape().element_type();
    PrimitiveType bias_ty =
        primitive_util::IsFloatingPointType(conv_ty) ? conv_ty : F32;
    bool addend_may_be_rank1_bias =
        addend->opcode() == HloOpcode::kBroadcast &&
        addend->dimensions().size() == 1 &&
        addend->dimensions(0) ==
            conv->convolution_dimension_numbers().output_feature_dimension() &&
        IsLosslesslyConvertibleTo(addend, bias_ty);
    bool addend_may_be_rank0_bias = addend->opcode() == HloOpcode::kBroadcast &&
                                    addend->dimensions().empty() &&
                                    IsLosslesslyConvertibleTo(addend, bias_ty);
    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    if (can_accept_bias && addend_may_be_rank1_bias) {
      new_operands[2] = MakeConvertToHlo(addend->mutable_operand(0), bias_ty,
                                         &addend->operand(0)->metadata());
    } else if (can_accept_bias && addend_may_be_rank0_bias) {
      new_operands[2] = MakeBroadcastHlo(
          MakeConvertToHlo(addend->mutable_operand(0), bias_ty,
                           &addend->operand(0)->metadata()),
          {},
          {gte->shape().dimensions(conv->convolution_dimension_numbers()
                                       .output_feature_dimension())});
    } else if (can_accept_side_input) {
      CHECK_EQ(new_operands.size(), 3);
      new_operands.push_back(addend);
      config.set_side_input_scale(1);
    } else {
      continue;
    }
    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseSideInputAlpha(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv;
    HloInstruction* side_input;
    auto pattern = m::Op(&conv)
                       .WithPredicate(IsConvCustomCall)
                       .WithOperand(3, m::Op(&side_input));
    if (!Match(instr, pattern)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.side_input_scale() != 1) {
      continue;
    }
    HloInstruction* before_reshape = side_input;
    while (before_reshape->opcode() == HloOpcode::kReshape ||
           before_reshape->opcode() == HloOpcode::kTranspose) {
      before_reshape = before_reshape->mutable_operand(0);
    }
    PrimitiveType conv_ty = conv->shape().tuple_shapes(0).element_type();
    PrimitiveType alpha_ty = conv_ty == F64 ? F64 : F32;
    HloInstruction* base;
    HloInstruction* alpha;
    if (!Match(
            before_reshape,
            m::MultiplyAnyOrder(
                m::Op(&base),
                m::Broadcast(m::ConstantEffectiveScalar(&alpha).WithPredicate(
                    [&](const HloInstruction* instr) {
                      return IsLosslesslyConvertibleTo(instr, alpha_ty);
                    }))))) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseSideInputAlpha: ", conv->ToString());
        })) {
      continue;
    }
    std::function<HloInstruction*(const HloInstruction*)> clone =
        [&](const HloInstruction* instr) {
          if (instr == before_reshape) {
            return base;
          }
          CHECK(instr->opcode() == HloOpcode::kReshape ||
                instr->opcode() == HloOpcode::kTranspose)
              << "Must be reshape or transpose: " << instr->ToString();
          return comp->AddInstruction(instr->CloneWithNewOperands(
              instr->shape(), {clone(instr->operand(0))}));
        };
    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[3] = clone(side_input);
    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_side_input_scale(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseElu(HloComputation* comp,
                             se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte1, *gte2, *gte3;
    HloInstruction* conv;
    HloInstruction* expm1;
    if (!Match(instr,
               m::Select(m::Compare(m::GetTupleElement(&gte1, m::Op()),
                                    m::Broadcast(m::ConstantEffectiveScalar(0)))
                             .WithComparisonDirection(ComparisonDirection::kGt)
                             .WithOneUse(),
                         m::GetTupleElement(
                             &gte2,
                             m::Op(&conv)
                                 .WithPredicate(IsNonDepthwiseConvCustomCall)
                                 .WithOneUse(),
                             0)
                             .WithElementType(F16),
                         m::Op(&expm1)
                             .WithOpcode(HloOpcode::kExpm1)
                             .WithOperand(0, m::GetTupleElement(&gte3, m::Op()))
                             .WithOneUse()))) {
      continue;
    }
    if (gte1 != gte2 || gte2 != gte3 || gte1->user_count() != 3) {
      continue;
    }
    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseElu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kElu);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte1));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseRelu(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte;
    HloInstruction* conv;
    if (!Match(instr,
               m::MaximumAnyOrder(
                   m::Broadcast(m::ConstantEffectiveScalar(0)),
                   m::GetTupleElement(
                       &gte, m::Op(&conv)
                                 .WithPredicate(IsNonDepthwiseConvCustomCall)
                                 .WithOneUse())
                       .WithOneUse()))) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseRelu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kRelu);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseRelu6(HloComputation* comp,
                               se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte, *conv;
    if (!Match(
            instr,
            m::Clamp(m::Broadcast(m::ConstantEffectiveScalar(0)),
                     m::GetTupleElement(
                         &gte, m::Op(&conv)
                                   .WithPredicate(IsNonDepthwiseConvCustomCall)
                                   .WithOneUse())
                         .WithElementType(F16)
                         .WithOneUse(),
                     m::Broadcast(m::ConstantEffectiveScalar(6))))) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }
    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseRelu6: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kRelu6);
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseLeakyRelu(HloComputation* comp,
                                   se::GpuComputeCapability cc) {
  if (!ShouldUseCudnnRuntimeFusion(comp->parent()->config().debug_options(),
                                   cc)) {
    return false;
  }
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction *gte1, *gte2, *gte3, *conv, *alpha;
    if (!Match(instr,
               m::Select(
                   m::Compare(m::GetTupleElement(&gte1, m::Op()),
                              m::Broadcast(m::ConstantEffectiveScalar(0)))
                       .WithComparisonDirection(ComparisonDirection::kGt)
                       .WithOneUse(),
                   m::GetTupleElement(
                       &gte2, m::Op(&conv)
                                  .WithPredicate(IsNonDepthwiseConvCustomCall)
                                  .WithOneUse())
                       .WithElementType(F16),
                   m::Multiply(m::GetTupleElement(&gte3, m::Op()),
                               m::Broadcast(m::ConstantEffectiveScalar(&alpha)))
                       .WithOneUse()))) {
      continue;
    }
    if (gte1 != gte2 || gte2 != gte3 || gte1->user_count() != 3) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        conv->backend_config<GpuBackendConfig>());
    CudnnConvBackendConfig& config =
        *gpu_config.mutable_cudnn_conv_backend_config();
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }
    if (!IsSuitableForCudnnRuntimeFusion(conv)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseLeakyRelu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kLeakyRelu);
    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_leakyrelu_alpha(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte1));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseConvertToF16(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;
    auto f32_convertible_to_f16_pat =
        m::Op().WithElementType(F32).WithPredicate(
            IsLosslesslyConvertibleToF16);
    if (!MatchAndLogIfFailed(
            instr, "f16 conv",
            m::Convert(
                m::GetTupleElement(
                    &gte,
                    m::Op(&conv)
                        .WithPredicate(IsConvCustomCall)
                        .WithOperand(0, f32_convertible_to_f16_pat)
                        .WithOperand(1, f32_convertible_to_f16_pat)
                        .WithOperandIfPresent(2, f32_convertible_to_f16_pat)
                        .WithOperandIfPresent(3, f32_convertible_to_f16_pat),
                    0)
                    .WithOneUse())
                .WithElementType(F16),
            VLOG_IS_ON(3),
            m::Op().WithOperand(0, m::GetTupleElement(m::Op().WithPredicate(
                                       IsConvCustomCall))))) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToF16: ", conv->ToString());
        })) {
      continue;
    }
    VLOG(2) << "Matched fp16 conv: " << conv->ToString();
    absl::InlinedVector<HloInstruction*, 4> new_operands;
    for (HloInstruction* operand : conv->operands()) {
      new_operands.push_back(
          MakeConvertToHlo(operand, F16, &operand->metadata()));
    }
    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(F16);
    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}
absl::StatusOr<bool> FuseConvertToS8(HloComputation* comp,
                                     se::GpuComputeCapability cc) {
  if (IsROCm(cc)) return false;
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;
    auto conv_pattern =
        m::Op(&conv)
            .WithPredicate(IsConvCustomCall)
            .WithOperand(0, m::Op().WithPredicate(IsLosslesslyConvertibleToS8))
            .WithOperand(1, m::Op().WithPredicate(IsLosslesslyConvertibleToS8));
    PrimitiveType conv_output_ty;
    if (MatchAndLogIfFailed(
            instr, "s8->s8 conv",
            m::Convert(m::Clamp(m::Broadcast(m::ConstantEffectiveScalar(-128)),
                                m::GetTupleElement(
                                    &gte,
                                    conv_pattern.WithOperandIfPresent(
                                        3, m::Op().WithPredicate(
                                               IsLosslesslyConvertibleToS8)),
                                    0)
                                    .WithOneUse(),
                                m::Broadcast(m::ConstantEffectiveScalar(127))))
                .WithElementType(S8),
            VLOG_IS_ON(3),
            m::Convert(m::Clamp(m::Op(),
                                m::GetTupleElement(
                                    m::Op().WithPredicate(IsConvCustomCall)),
                                m::Op()))
                .WithElementType(S8))) {
      conv_output_ty = S8;
    } else if (MatchAndLogIfFailed(
                   instr, "s8->f32 conv",
                   m::GetTupleElement(&gte,
                                      conv_pattern.WithOperandIfPresent(
                                          3, m::Op().WithElementType(F32)),
                                      0)
                       .WithElementType(F32),
                   VLOG_IS_ON(3),
                   m::GetTupleElement(m::Op().WithPredicate(IsConvCustomCall))
                       .WithElementType(F32))) {
      conv_output_ty = F32;
    } else {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToS8: ", conv->ToString());
        })) {
      continue;
    }
    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[0] =
        MakeConvertToHlo(new_operands[0], S8, &new_operands[0]->metadata());
    new_operands[1] =
        MakeConvertToHlo(new_operands[1], S8, &new_operands[1]->metadata());
    if (new_operands.size() >= 4) {
      new_operands[3] = MakeConvertToHlo(new_operands[3], conv_output_ty,
                                         &new_operands[3]->metadata());
    }
    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(conv_output_ty);
    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}
absl::Status CheckNoIllegalIntegerConvs(HloComputation* comp) {
  auto is_integral_not_s8 = [](const Shape& s) {
    return primitive_util::IsIntegralType(s.element_type()) &&
           s.element_type() != S8;
  };
  std::vector<HloInstruction*> bad_convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (!IsConvCustomCall(instr)) {
      continue;
    }
    if (is_integral_not_s8(instr->shape().tuple_shapes(0)) ||
        is_integral_not_s8(instr->operand(0)->shape()) ||
        is_integral_not_s8(instr->operand(1)->shape()) ||
        (instr->operand_count() >= 4 &&
         is_integral_not_s8(instr->operand(3)->shape()))) {
      bad_convs.push_back(instr);
    }
  }
  if (bad_convs.empty()) {
    return absl::OkStatus();
  }
  return Unimplemented(
      R"(
Can't lower one or more integer convolutions to idioms supported by CuDNN.
CuDNN integer convolutions must have:
  - s8 input and filter,
  - f32 bias (if present),
  - s8 or f32 output, and
  - s8 side_input (if present) if output is s8.
For each of the unsupported convs below, we weren't able to lower one of the
operands or the output to the appropriate type.
See specific HLO idioms in cudnn_fused_conv_rewriter.h, and see cudnn semantics:
https:
https:
Unsupported convs:
%s
******* Full HLO module *******
%s
)",
      absl::StrJoin(bad_convs, "\n",
                    [](std::string* out, HloInstruction* instr) {
                      absl::StrAppend(out, " - ", instr->ToString());
                    }),
      comp->parent()->ToString());
}
void VlogStats(HloModule* module) {
  if (!VLOG_IS_ON(1)) {
    return;
  }
  VLOG(1) << "Results of CudnnFusedConvRewriter for " << module->name();
  absl::flat_hash_map<std::string, int> stats;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!Match(instr, m::Op().WithPredicate(IsConvCustomCall))) {
        continue;
      }
      VLOG(3) << instr->ToString();
      if (instr->custom_call_target() == kCudnnConvForwardCallTarget) {
        ++stats["01 non-fused forward convs"];
      } else if (instr->custom_call_target() ==
                 kCudnnConvBiasActivationForwardCallTarget) {
        ++stats["02 fused forward convs"];
      }
      PrimitiveType conv_in_ty = instr->operand(0)->shape().element_type();
      PrimitiveType conv_out_ty = instr->shape().tuple_shapes(0).element_type();
      if (conv_in_ty == F32) {
        ++stats["10 f32 convs"];
      } else if (conv_in_ty == F16) {
        ++stats["11 f16 convs"];
      } else if (conv_in_ty == S8) {
        if (conv_out_ty == S8) {
          ++stats["12 s8->s8 convs"];
        } else if (conv_out_ty == F32) {
          ++stats["13 s8->f32 convs"];
        } else {
          LOG(ERROR) << "Unexpected conv: " << instr->ToString();
        }
      }
      if (instr->operand_count() > 2) {
        ++stats["20 convs with bias"];
        if (Match(instr->operand(2),
                  m::Broadcast(m::ConstantEffectiveScalar(0)))) {
          ++stats["21 convs with 0 bias"];
        }
      }
      if (instr->operand_count() > 3) {
        ++stats["22 convs with side-input"];
      }
      auto gpu_config = instr->backend_config<GpuBackendConfig>();
      if (!gpu_config.ok()) {
        LOG(ERROR) << "Couldn't parse backend config for " << instr->ToString();
        continue;
      }
      const CudnnConvBackendConfig& config =
          gpu_config->cudnn_conv_backend_config();
      if (config.conv_result_scale() != 1) {
        ++stats["30 convs with result scale"];
      }
      if (config.side_input_scale() != 0 && config.side_input_scale() != 1) {
        ++stats["31 convs with side-input scale"];
      }
      ++stats[absl::StrCat(
          "32 convs with activation mode ",
          se::dnn::ActivationMode_Name(config.activation_mode()))];
    }
  }
  std::vector<std::pair<std::string, int>> stats_sorted(stats.begin(),
                                                        stats.end());
  absl::c_sort(stats_sorted);
  for (const auto& kv : stats_sorted) {
    VLOG(1) << absl::StreamFormat("%4d %s", kv.second,
                                  absl::string_view(kv.first).substr(3));
  }
}
}  
absl::StatusOr<bool> CudnnFusedConvRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    bool changed = false;
    if (!IsROCm(compute_capability_)) {
      auto cc = std::get<se::CudaComputeCapability>(compute_capability_);
      TF_ASSIGN_OR_RETURN(
          changed, F8GraphConv(comp, cc, dnn_version_, toolkit_version_));
      if (changed) {
        return changed;
      }
    }
    TF_ASSIGN_OR_RETURN(changed, FuseRemoveConvertInConv(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseConvAlpha(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseElu(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu6(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseLeakyRelu(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseConvertToF16(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseConvertToS8(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseElu(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseRelu6(comp, compute_capability_));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseLeakyRelu(comp, compute_capability_));
    any_changed |= changed;
    TF_RETURN_IF_ERROR(CheckNoIllegalIntegerConvs(comp));
  }
  VlogStats(module);
  return any_changed;
}
}  
}  