#include "xla/hlo/builder/value_inference.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
Literal CreatePredLiteral(bool pred, const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    const auto& reference_shape_tuple_shapes = reference_shape.tuple_shapes();
    sub_literals.reserve(reference_shape_tuple_shapes.size());
    for (const Shape& shape : reference_shape_tuple_shapes) {
      sub_literals.emplace_back(CreatePredLiteral(pred, shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateR0(pred);
  }
  Literal literal = LiteralUtil::CreateR0(pred);
  Literal literal_broadcast =
      literal.Broadcast(ShapeUtil::ChangeElementType(reference_shape, PRED), {})
          .value();
  return literal_broadcast;
}
Literal CreateS64Literal(int64_t value, const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    const auto& reference_shape_tuple_shapes = reference_shape.tuple_shapes();
    sub_literals.reserve(reference_shape_tuple_shapes.size());
    for (const Shape& shape : reference_shape_tuple_shapes) {
      sub_literals.emplace_back(CreateS64Literal(value, shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateToken();
  }
  Literal literal = LiteralUtil::CreateR0<int64_t>(value);
  return literal
      .Broadcast(ShapeUtil::ChangeElementType(reference_shape, S64), {})
      .value();
}
Literal CreateGarbageLiteral(const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    for (const Shape& shape : reference_shape.tuple_shapes()) {
      sub_literals.emplace_back(CreateGarbageLiteral(shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateToken();
  }
  Literal literal = LiteralUtil::One(element_type);
  return literal.Broadcast(reference_shape, {}).value();
}
struct HloProtoEvaluator {
  explicit HloProtoEvaluator(HloEvaluator& evaluator, HloInstructionProto inst)
      : evaluator(evaluator),
        inst(std::move(inst)),
        module("EmptyModuleForEvaluation", HloModuleConfig()) {}
  HloProtoEvaluator& WithComputation(
      std::unique_ptr<HloComputation> new_computation) {
    computation = new_computation.get();
    computation->ClearUniqueIdInternal();
    for (HloInstruction* inst : computation->instructions()) {
      inst->ClearUniqueIdInternal();
    }
    module.AddEmbeddedComputation(std::move(new_computation));
    return *this;
  }
  HloProtoEvaluator& WithPrimitiveType(PrimitiveType new_primitive_type) {
    primitive_type = new_primitive_type;
    return *this;
  }
  HloProtoEvaluator& WithOpCode(HloOpcode new_opcode) {
    opcode = new_opcode;
    return *this;
  }
  HloProtoEvaluator& WithOperands(absl::Span<Literal> operands) {
    this->operands = operands;
    return *this;
  }
  HloProtoEvaluator& WithSubshape(ShapeIndex shape_index) {
    this->shape_index = std::move(shape_index);
    return *this;
  }
  absl::StatusOr<Literal> Evaluate() {
    HloComputation::Builder builder("EmptyComputation");
    absl::flat_hash_map<int64_t, HloInstruction*> operand_map;
    for (int64_t i = 0; i < inst.operand_ids_size(); ++i) {
      int64_t operand_handle = inst.operand_ids(i);
      std::unique_ptr<HloInstruction> operand =
          HloInstruction::CreateConstant(operands[i].Clone());
      operand_map[operand_handle] = operand.get();
      builder.AddInstruction(std::move(operand));
    }
    if (primitive_type.has_value()) {
      *inst.mutable_shape() = ShapeUtil::ChangeElementType(
                                  Shape(inst.shape()), primitive_type.value())
                                  .ToProto();
    }
    if (opcode.has_value()) {
      *inst.mutable_opcode() = std::string(HloOpcodeString(opcode.value()));
    }
    absl::flat_hash_map<int64_t, HloComputation*> computation_map;
    if (inst.called_computation_ids_size() != 0) {
      TF_RET_CHECK(inst.called_computation_ids_size() == 1 &&
                   computation != nullptr)
          << inst.DebugString();
      computation_map[inst.called_computation_ids(0)] = computation;
    }
    TF_ASSIGN_OR_RETURN(
        auto new_instruction,
        HloInstruction::CreateFromProto(inst, operand_map, computation_map));
    new_instruction->ClearUniqueIdInternal();
    builder.AddInstruction(std::move(new_instruction));
    auto computation = builder.Build();
    module.AddEntryComputation(std::move(computation));
    if (shape_index.empty()) {
      return evaluator.Evaluate(module.entry_computation()->root_instruction());
    } else {
      TF_ASSIGN_OR_RETURN(
          auto result,
          evaluator.Evaluate(module.entry_computation()->root_instruction()));
      return result.SubLiteral(this->shape_index);
    }
  }
  HloEvaluator& evaluator;
  HloInstructionProto inst;
  HloModule module;
  absl::Span<Literal> operands;
  ShapeIndex shape_index = {};
  HloComputation* computation = nullptr;
  std::optional<PrimitiveType> primitive_type = std::nullopt;
  std::optional<HloOpcode> opcode = std::nullopt;
};
enum PostorderDFSNodeType {
  kConstantValue = 0,
  kConstantUpperBound,
  kConstantLowerBound,
  kValueIsDynamic,
  kBoundIsDynamic,
};
std::string PostorderDFSNodeTypeToString(PostorderDFSNodeType type) {
  switch (type) {
    case kConstantValue:
      return "kConstantValue";
    case kConstantUpperBound:
      return "kConstantUpperBound";
    case kConstantLowerBound:
      return "kConstantLowerBound";
    case kValueIsDynamic:
      return "kValueIsDynamic";
    case kBoundIsDynamic:
      return "kBoundIsDynamic";
  }
}
struct InferenceContext {
  explicit InferenceContext(ShapeIndex shape_index,
                            std::vector<int64_t> caller_operand_handles)
      : shape_index(std::move(shape_index)),
        caller_operand_handles(std::move(caller_operand_handles)) {}
  ShapeIndex shape_index;
  std::vector<int64_t> caller_operand_handles;
};
struct PostorderDFSDep {
  explicit PostorderDFSDep(int64_t handle, PostorderDFSNodeType type,
                           InferenceContext context, std::string annotation)
      : handle(handle),
        type(type),
        context(std::move(context)),
        annotation(std::move(annotation)) {}
  int64_t handle;
  PostorderDFSNodeType type;
  InferenceContext context;
  std::string annotation;
};
using Visit = std::function<absl::StatusOr<Literal>(absl::Span<Literal>)>;
using Visit0D = std::function<absl::StatusOr<Literal>()>;
using Visit1D = std::function<absl::StatusOr<Literal>(Literal)>;
using Visit2D = std::function<absl::StatusOr<Literal>(Literal, Literal)>;
struct [[nodiscard]] PostorderDFSNode {
  PostorderDFSNode& AddDependency(int64_t handle, PostorderDFSNodeType type,
                                  InferenceContext context,
                                  std::string annotation = "") {
    dependencies.emplace_back(handle, type, std::move(context),
                              std::move(annotation));
    return *this;
  }
  PostorderDFSNode& AddVisit(const Visit& visit) {
    this->visit = visit;
    return *this;
  }
  PostorderDFSNode& AddVisit(const Visit0D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) { return visit(); };
    return *this;
  }
  PostorderDFSNode& AddVisit(const Visit1D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) {
      return visit(std::move(literals[0]));
    };
    return *this;
  }
  PostorderDFSNode& AddVisit(const Visit2D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) {
      return visit(std::move(literals[0]), std::move(literals[1]));
    };
    return *this;
  }
  std::vector<PostorderDFSDep> dependencies;
  Visit visit;
};
using HandleToInstruction =
    std::function<absl::StatusOr<const HloInstructionProto*>(int64_t)>;
using HandleToComputation = std::function<const HloComputationProto*(int64_t)>;
struct PostorderDFSVisitor {
  PostorderDFSVisitor(HloEvaluator& evaluator,
                      HandleToInstruction handle_to_instruction,
                      HandleToComputation handle_to_computation)
      : evaluator(evaluator),
        handle_to_instruction(handle_to_instruction),
        handle_to_computation(handle_to_computation) {}
  absl::StatusOr<PostorderDFSNode> AnalyzeUpperBound(int64_t handle,
                                                     InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeLowerBound(int64_t handle,
                                                     InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeIsDynamic(int64_t handle,
                                                    PostorderDFSNodeType type,
                                                    InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeConstant(int64_t handle,
                                                   InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeConstantValueFallback(
      int64_t handle, PostorderDFSNodeType type, InferenceContext context);
  absl::StatusOr<Literal> PostOrderDFSVisit(int64_t handle,
                                            PostorderDFSNodeType type);
  bool IsValueEffectiveInteger(int64_t handle) {
    const HloInstructionProto* instr = handle_to_instruction(handle).value();
    if (primitive_util::IsIntegralType(instr->shape().element_type())) {
      return true;
    }
    HloOpcode opcode = StringToHloOpcode(instr->opcode()).value();
    if (opcode != HloOpcode::kConvert) {
      return false;
    }
    const HloInstructionProto* parent =
        handle_to_instruction(instr->operand_ids(0)).value();
    if (primitive_util::IsIntegralType(parent->shape().element_type())) {
      return true;
    }
    return false;
  }
  bool IsInstructionOverLimit(const HloInstructionProto* proto,
                              const InferenceContext& context) {
    auto subshape = std::make_unique<Shape>(
        ShapeUtil::GetSubshape(Shape(proto->shape()), context.shape_index));
    if (subshape->IsArray() &&
        ShapeUtil::ElementsIn(*subshape) > kLargeShapeElementLimit) {
      return true;
    }
    HloOpcode opcode = StringToHloOpcode(proto->opcode()).value();
    for (int64_t operand_id : proto->operand_ids()) {
      const HloInstructionProto* operand =
          handle_to_instruction(operand_id).value();
      auto operand_shape = std::make_unique<Shape>(operand->shape());
      if (operand_shape->IsArray() &&
          ShapeUtil::ElementsIn(*operand_shape) > kLargeShapeElementLimit &&
          opcode != HloOpcode::kGetDimensionSize &&
          opcode != HloOpcode::kSetDimensionSize) {
        return true;
      }
    }
    return false;
  }
  struct CacheKey {
    CacheKey(int64_t handle, InferenceContext context,
             PostorderDFSNodeType type)
        : handle(handle), context(context), type(type) {}
    int64_t handle;
    InferenceContext context;
    PostorderDFSNodeType type;
    template <typename H>
    friend H AbslHashValue(H h, const CacheKey& key) {
      h = H::combine(std::move(h), key.handle);
      h = H::combine(std::move(h), key.context.shape_index.ToString());
      h = H::combine(std::move(h),
                     VectorString(key.context.caller_operand_handles));
      h = H::combine(std::move(h), key.type);
      return h;
    }
    friend bool operator==(const CacheKey& lhs, const CacheKey& rhs) {
      return lhs.handle == rhs.handle &&
             lhs.context.shape_index == rhs.context.shape_index &&
             lhs.context.caller_operand_handles ==
                 rhs.context.caller_operand_handles &&
             lhs.type == rhs.type;
    }
  };
  HloEvaluator& evaluator;
  absl::flat_hash_map<CacheKey, Literal> evaluated;
  HandleToInstruction handle_to_instruction;
  HandleToComputation handle_to_computation;
  static constexpr int64_t kLargeShapeElementLimit = 1000 * 1000;
};
PostorderDFSNode CreateAllDynamicResult(const Shape& shape,
                                        const PostorderDFSNodeType& type) {
  return PostorderDFSNode().AddVisit(
      [shape, type](absl::Span<Literal>) -> Literal {
        if (type == PostorderDFSNodeType::kConstantValue ||
            type == PostorderDFSNodeType::kConstantUpperBound ||
            type == PostorderDFSNodeType::kConstantLowerBound) {
          return CreateGarbageLiteral(shape);
        } else {
          return CreatePredLiteral(true, shape);
        }
      });
}
}  
absl::StatusOr<PostorderDFSNode>
PostorderDFSVisitor::AnalyzeConstantValueFallback(int64_t handle,
                                                  PostorderDFSNodeType type,
                                                  InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  Shape subshape =
      ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index);
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    InferenceContext dep_context = context;
    dep_context.shape_index = {};
    result.AddDependency(operand_id, type, dep_context);
  }
  switch (opcode) {
    case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64_t caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return CreateAllDynamicResult(subshape, type);
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide:
    case HloOpcode::kGetDimensionSize: {
      return InvalidArgument(
          "AnalyzeConstantValueFallback can't handle opcode: %s",
          root->opcode());
    }
    case HloOpcode::kCall: {
      auto node = PostorderDFSNode();
      auto* call_proto = root;
      if (call_proto->operand_ids_size() != 1) {
        return CreateAllDynamicResult(subshape, type);
      }
      int64_t called_root =
          handle_to_computation(call_proto->called_computation_ids(0))
              ->root_id();
      InferenceContext call_context = context;
      call_context.caller_operand_handles.push_back(call_proto->operand_ids(0));
      node.AddDependency(called_root, PostorderDFSNodeType::kConstantValue,
                         call_context, "callee's root instruction");
      return node.AddVisit([](Literal operand) -> absl::StatusOr<Literal> {
        return std::move(operand);
      });
    }
    case HloOpcode::kConditional: {
      auto node = PostorderDFSNode();
      auto* conditional_proto = root;
      InferenceContext predicate_context = context;
      predicate_context.shape_index = {};
      node.AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue,
                         predicate_context)
          .AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic,
                         predicate_context);
      const int64_t branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64_t i = 0; i < branch_size; ++i) {
        int64_t branch_root =
            handle_to_computation(conditional_proto->called_computation_ids(i))
                ->root_id();
        InferenceContext branch_context = context;
        branch_context.caller_operand_handles.push_back(
            conditional_proto->operand_ids(i + 1));
        node.AddDependency(branch_root, PostorderDFSNodeType::kConstantValue,
                           branch_context);
      }
      return node.AddVisit(
          [](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            int64_t pred_is_dynamic = operands[1].Get<bool>({});
            if (pred_is_dynamic) {
              return std::move(operands[2]);
            } else {
              int64_t branch_index = 0;
              if (operands[0].shape().element_type() == PRED) {
                if (operands[0].Get<bool>({})) {
                  branch_index = 0;
                } else {
                  branch_index = 1;
                }
              } else {
                branch_index = operands[0].GetIntegralAsS64({}).value();
              }
              const int64_t branch_dynamism_index = 2 + branch_index;
              return std::move(operands[branch_dynamism_index]);
            }
          });
    }
    case HloOpcode::kGetTupleElement: {
      int64_t operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }
    case HloOpcode::kReduce:
    case HloOpcode::kSort:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow: {
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, computation_proto, context,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithComputation(std::move(computation))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    default: {
      if (opcode == HloOpcode::kTuple && !context.shape_index.empty()) {
        int64_t tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
            .Evaluate();
      });
    }
  }
}
absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeUpperBound(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  Shape subshape =
      ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index);
  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantUpperBound);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle).value();
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension]() -> absl::StatusOr<Literal> {
            return LiteralUtil::CreateR0<int32_t>(
                operand_proto->shape().dimensions(dimension));
          });
    }
    case HloOpcode::kAbs: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddVisit([this](Literal lower_bound,
                           Literal upper_bound) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(auto lower_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, lower_bound));
            TF_ASSIGN_OR_RETURN(auto upper_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, upper_bound));
            return evaluator.EvaluateElementwiseBinaryOp(
                HloOpcode::kMaximum, lower_bound_abs, upper_bound_abs);
          });
    }
    case HloOpcode::kSort: {
      auto dfs = PostorderDFSNode();
      InferenceContext dep_context = context;
      dep_context.shape_index = {};
      if (!context.shape_index.empty()) {
        dfs.AddDependency(root->operand_ids(context.shape_index[0]),
                          PostorderDFSNodeType::kConstantUpperBound,
                          dep_context);
      } else {
        for (int64_t i = 0; i < root->operand_ids_size(); ++i) {
          dfs.AddDependency(root->operand_ids(i),
                            PostorderDFSNodeType::kConstantUpperBound,
                            dep_context);
        }
      }
      return dfs.AddVisit(
          [root,
           context](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            std::vector<Literal> results;
            results.reserve(operands.size());
            for (int64_t i = 0; i < operands.size(); ++i) {
              auto max = LiteralUtil::MaxElement(operands[i]);
              results.emplace_back(
                  max.Broadcast(operands[i].shape(), {}).value());
            }
            if (ShapeUtil::GetSubshape(Shape(root->shape()),
                                       context.shape_index)
                    .IsTuple()) {
              return LiteralUtil::MakeTupleOwned(std::move(results));
            } else {
              return std::move(results[0]);
            }
          });
    }
    case HloOpcode::kNegate: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddVisit([this](Literal lower_bound) -> absl::StatusOr<Literal> {
            return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                        lower_bound);
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddVisit([root, opcode, this](
                        Literal upper_bound,
                        Literal lower_bound) -> absl::StatusOr<Literal> {
            if (opcode == HloOpcode::kDivide &&
                this->IsValueEffectiveInteger(root->operand_ids(1))) {
              auto zero = LiteralUtil::Zero(lower_bound.shape().element_type());
              zero = zero.Broadcast(lower_bound.shape(), {}).value();
              TF_ASSIGN_OR_RETURN(
                  auto lower_bound_is_zero,
                  evaluator.EvaluateElementwiseCompareOp(
                      ComparisonDirection::kEq, lower_bound, zero));
              auto one = LiteralUtil::One(lower_bound.shape().element_type());
              one = one.Broadcast(lower_bound.shape(), {}).value();
              TF_ASSIGN_OR_RETURN(
                  lower_bound, evaluator.EvaluateElementwiseTernaryOp(
                                   HloOpcode::kSelect, lower_bound_is_zero, one,
                                   lower_bound));
            }
            std::vector<Literal> new_operands;
            new_operands.emplace_back(std::move(upper_bound));
            new_operands.emplace_back(std::move(lower_bound));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(absl::MakeSpan(new_operands))
                .Evaluate();
          });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([root]() -> absl::StatusOr<Literal> {
          if (root->literal().shape().element_type() == TUPLE) {
            return Literal::CreateFromProto(root->literal().tuple_literals(0));
          } else {
            return Literal::CreateFromProto(root->literal());
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(0),
                           PostorderDFSNodeType::kConstantUpperBound, context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return InvalidArgument(
          "Upper-bound inferencing on custom call %s is not supported",
          root->DebugString());
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddVisit([root, this](absl::Span<Literal> operands) {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantUpperBound, context);
  }
}
absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeLowerBound(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  Shape subshape =
      ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index);
  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantLowerBound);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [dimension, operand_proto]() -> absl::StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              return LiteralUtil::CreateR0<int32_t>(0);
            } else {
              return LiteralUtil::CreateR0<int32_t>(
                  operand_proto->shape().dimensions(dimension));
            }
          });
    }
    case HloOpcode::kAbs: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddVisit([this](Literal lower_bound,
                           Literal upper_bound) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(auto lower_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, lower_bound));
            TF_ASSIGN_OR_RETURN(auto upper_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, upper_bound));
            return evaluator.EvaluateElementwiseBinaryOp(
                HloOpcode::kMinimum, lower_bound_abs, upper_bound_abs);
          });
    }
    case HloOpcode::kNegate: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddVisit([this](Literal upper_bound) -> absl::StatusOr<Literal> {
            return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                        upper_bound);
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddVisit(
              [root,
               this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
                return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                    ->WithOperands(operands)
                    .Evaluate();
              });
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddVisit([root, this](absl::Span<Literal> operands) {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantLowerBound, context);
  }
}
absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstant(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  HloOpcode opcode = StringToHloOpcode(root->opcode()).value();
  Shape subshape =
      ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index);
  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantValue);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, root]() -> absl::StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              return CreateGarbageLiteral(Shape(root->shape()));
            } else {
              return LiteralUtil::CreateR0<int32_t>(
                  operand_proto->shape().dimensions(dimension));
            }
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide: {
      PostorderDFSNode result;
      for (auto operand_id : root->operand_ids()) {
        result.AddDependency(operand_id, PostorderDFSNodeType::kConstantValue,
                             context);
      }
      return result.AddVisit(
          [root,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([root]() -> absl::StatusOr<Literal> {
          if (root->literal().shape().element_type() == TUPLE) {
            return Literal::CreateFromProto(root->literal().tuple_literals(0));
          } else {
            return Literal::CreateFromProto(root->literal());
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(0),
                           PostorderDFSNodeType::kConstantValue, context)
            .AddVisit([](Literal operand) { return operand; });
      } else {
        return PostorderDFSNode().AddVisit(
            [root, context](absl::Span<Literal>) {
              return CreateGarbageLiteral(ShapeUtil::GetSubshape(
                  Shape(root->shape()), context.shape_index));
            });
      }
    }
    case HloOpcode::kSort: {
      PostorderDFSNode result;
      InferenceContext dep_context = context;
      dep_context.shape_index = {};
      for (auto operand_id : root->operand_ids()) {
        result.AddDependency(operand_id, PostorderDFSNodeType::kConstantValue,
                             dep_context);
      }
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, context, computation_proto,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithComputation(std::move(computation))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantValue, context);
  }
}
absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeIsDynamic(
    int64_t handle, PostorderDFSNodeType type, InferenceContext context) {
  TF_RETURN_IF_ERROR(handle_to_instruction(handle).status());
  TF_RET_CHECK(handle_to_instruction(handle).value());
  VLOG(1) << "Analyzing IsDynamic on "
          << handle_to_instruction(handle).value()->DebugString();
  if (IsInstructionOverLimit(handle_to_instruction(handle).value(), context)) {
    return CreateAllDynamicResult(
        ShapeUtil::GetSubshape(
            Shape(handle_to_instruction(handle).value()->shape()),
            context.shape_index),
        type);
  }
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    InferenceContext dep_context = context;
    dep_context.shape_index = {};
    result.AddDependency(operand_id, type, dep_context);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, type]() -> absl::StatusOr<Literal> {
            if (type == PostorderDFSNodeType::kBoundIsDynamic) {
              return LiteralUtil::CreateR0<bool>(false);
            }
            return LiteralUtil::CreateR0<bool>(
                operand_proto->shape().is_dynamic_dimension(dimension));
          });
    }
    case HloOpcode::kSort: {
      auto dfs = PostorderDFSNode();
      InferenceContext dep_context = context;
      dep_context.shape_index = {};
      for (int64_t i = 0; i < root->operand_ids_size(); ++i) {
        dfs.AddDependency(root->operand_ids(i), type, dep_context);
      }
      return dfs.AddVisit([root, context, type](absl::Span<Literal> operands)
                              -> absl::StatusOr<Literal> {
        bool all_operands_values_static = true;
        for (int64_t i = 0; i < operands.size(); ++i) {
          all_operands_values_static &= operands[i].IsAll(0);
        }
        if (type == PostorderDFSNodeType::kValueIsDynamic) {
          return CreatePredLiteral(!all_operands_values_static,
                                   ShapeUtil::GetSubshape(Shape(root->shape()),
                                                          context.shape_index));
        }
        CHECK(type == PostorderDFSNodeType::kBoundIsDynamic);
        if (!context.shape_index.empty()) {
          int64_t index = context.shape_index[0];
          bool all_values_static = operands[index].IsAll(0);
          return CreatePredLiteral(!all_values_static, operands[index].shape());
        }
        std::vector<Literal> results;
        results.reserve(operands.size());
        for (int64_t i = 0; i < operands.size(); ++i) {
          bool all_values_static = operands[i].IsAll(0);
          results.emplace_back(
              CreatePredLiteral(!all_values_static, operands[i].shape()));
        }
        if (!ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index)
                 .IsTuple()) {
          return std::move(results[0]);
        }
        return LiteralUtil::MakeTupleOwned(std::move(results));
      });
    }
    case HloOpcode::kSetDimensionSize:
      return result.AddVisit([root, type](absl::Span<Literal> operands) {
        bool any_dynamic_operand = absl::c_any_of(
            operands, [](Literal& operand) { return !operand.IsAll(0); });
        return CreatePredLiteral(
            type == PostorderDFSNodeType::kValueIsDynamic &&
                any_dynamic_operand,
            ShapeUtil::MakeStaticShape(Shape(root->shape())));
      });
    case HloOpcode::kDynamicSlice: {
      return result.AddVisit([root](absl::Span<Literal> operands) {
        bool any_dynamic_operand = absl::c_any_of(
            operands, [](Literal& operand) { return !operand.IsAll(0); });
        return CreatePredLiteral(any_dynamic_operand, Shape(root->shape()));
      });
    }
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCos:
    case HloOpcode::kClz:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kConvert:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh: {
      return result.AddVisit([](Literal operand) { return operand; });
    }
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kComplex:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kCompare:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical: {
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
            .WithPrimitiveType(PRED)
            .WithOpCode(HloOpcode::kOr)
            .Evaluate();
      });
    }
    case HloOpcode::kTuple:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kPad: {
      if (opcode == HloOpcode::kTuple && !context.shape_index.empty()) {
        int64_t tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
            .WithPrimitiveType(PRED)
            .Evaluate();
      });
    }
    case HloOpcode::kCall: {
      auto node = PostorderDFSNode();
      auto* call_proto = root;
      if (call_proto->operand_ids_size() != 1) {
        return CreateAllDynamicResult(
            Shape(handle_to_instruction(handle).value()->shape()), type);
      }
      int64_t call_root =
          handle_to_computation(call_proto->called_computation_ids(0))
              ->root_id();
      InferenceContext branch_context = context;
      branch_context.caller_operand_handles.push_back(
          call_proto->operand_ids(0));
      node.AddDependency(call_root, PostorderDFSNodeType::kValueIsDynamic,
                         branch_context, "callee's root instruction");
      return node.AddVisit(
          [context](Literal operand) -> absl::StatusOr<Literal> {
            return operand;
          });
    }
    case HloOpcode::kConditional: {
      auto node = PostorderDFSNode();
      auto* conditional_proto = root;
      InferenceContext predicate_context = context;
      predicate_context.shape_index = {};
      node.AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue,
                         predicate_context)
          .AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic,
                         predicate_context);
      const int64_t branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64_t i = 0; i < branch_size; ++i) {
        int64_t branch_root =
            handle_to_computation(conditional_proto->called_computation_ids(i))
                ->root_id();
        InferenceContext branch_context = context;
        branch_context.caller_operand_handles.push_back(
            conditional_proto->operand_ids(i + 1));
        node.AddDependency(branch_root, PostorderDFSNodeType::kConstantValue,
                           branch_context,
                           absl::StrFormat("branch %lld's value", i))
            .AddDependency(branch_root, PostorderDFSNodeType::kValueIsDynamic,
                           branch_context,
                           absl::StrFormat("branch %lld's dynamism", i));
      }
      return node.AddVisit([root, branch_size,
                            context](absl::Span<Literal> operands)
                               -> absl::StatusOr<Literal> {
        int64_t pred_is_dynamic = operands[1].Get<bool>({});
        auto result = CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
        if (pred_is_dynamic) {
          VLOG(1) << "predict is dynamic value" << result.ToString();
          result.MutableEachCell<bool>(
              [&](absl::Span<const int64_t> indices, bool value) {
                std::string branch_value = operands[2].GetAsString(indices, {});
                for (int64_t i = 0; i < branch_size; ++i) {
                  const int64_t branch_value_index = 2 + 2 * i;
                  const int64_t branch_dynamism_index = 2 + 2 * i + 1;
                  auto branch_is_dynamic =
                      operands[branch_dynamism_index].Get<bool>(indices);
                  if (branch_is_dynamic) {
                    return true;
                  }
                  if (branch_value !=
                      operands[branch_value_index].GetAsString(indices, {})) {
                    return true;
                  }
                }
                return false;
              });
          return result;
        } else {
          VLOG(1) << "predict is constant value";
          int64_t branch_index = 0;
          if (operands[0].shape().element_type() == PRED) {
            if (operands[0].Get<bool>({})) {
              branch_index = 0;
            } else {
              branch_index = 1;
            }
          } else {
            branch_index = operands[0].GetIntegralAsS64({}).value();
          }
          const int64_t branch_dynamism_index = 2 + 2 * branch_index + 1;
          return std::move(operands[branch_dynamism_index]);
        }
      });
    }
    case HloOpcode::kGetTupleElement: {
      int64_t operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }
    case HloOpcode::kReduce: {
      return result.AddVisit(
          [root, context, this](absl::Span<Literal> operands) {
            Shape root_shape = Shape(root->shape());
            Shape scalar_shape = ShapeUtil::MakeScalarShape(xla::PRED);
            std::unique_ptr<HloComputation> reduce_or;
            if (root_shape.IsTuple()) {
              HloComputation::Builder b("reduce_or");
              auto accum = b.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<bool>(false)));
              for (int i = 0; i < root_shape.tuple_shapes_size(); ++i) {
                auto lhs = b.AddInstruction(
                    HloInstruction::CreateParameter(i, scalar_shape, "lhs"));
                auto rhs = b.AddInstruction(HloInstruction::CreateParameter(
                    i + root_shape.tuple_shapes_size(), scalar_shape, "rhs"));
                accum = b.AddInstruction(HloInstruction::CreateBinary(
                    scalar_shape, HloOpcode::kOr, accum, lhs));
                accum = b.AddInstruction(HloInstruction::CreateBinary(
                    scalar_shape, HloOpcode::kOr, accum, rhs));
              }
              std::vector<HloInstruction*> results(
                  root_shape.tuple_shapes_size(), accum);
              b.AddInstruction(HloInstruction::CreateTuple(results));
              reduce_or = b.Build();
            } else {
              HloComputation::Builder b("reduce_or");
              auto lhs = b.AddInstruction(
                  HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
              auto rhs = b.AddInstruction(
                  HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
              b.AddInstruction(HloInstruction::CreateBinary(
                  scalar_shape, HloOpcode::kOr, lhs, rhs));
              reduce_or = b.Build();
            }
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithPrimitiveType(PRED)
                .WithComputation(std::move(reduce_or))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    case HloOpcode::kConstant:
    case HloOpcode::kIota: {
      return result.AddVisit(
          [root]() { return CreatePredLiteral(false, Shape(root->shape())); });
    }
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64_t caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return result.AddVisit([root, context]() {
        return CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
      });
    }
    case HloOpcode::kSelect: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic, context)
          .AddDependency(root->operand_ids(1), type, context)
          .AddDependency(root->operand_ids(2), type, context)
          .AddVisit([root](absl::Span<Literal> operands)
                        -> absl::StatusOr<Literal> {
            OptionalLiteral optional_selector_literal(std::move(operands[0]),
                                                      std::move(operands[1]));
            Literal lhs = std::move(operands[2]);
            Literal rhs = std::move(operands[3]);
            auto result = CreatePredLiteral(true, Shape(root->shape()));
            result.MutableEachCell<bool>(
                [&](absl::Span<const int64_t> indices, bool value) {
                  std::optional<bool> optional_selector =
                      optional_selector_literal.Get<bool>(indices);
                  bool lhs_value = lhs.Get<bool>(indices);
                  bool rhs_value = rhs.Get<bool>(indices);
                  if (optional_selector.has_value()) {
                    if (*optional_selector) {
                      return lhs_value;
                    } else {
                      return rhs_value;
                    }
                  } else {
                    return true;
                  }
                });
            return result;
          });
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0), type, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kValueIsDynamic, context)
          .AddVisit(
              [root,
               this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
                OptionalLiteral optional_selector_literal(
                    std::move(operands[1]), std::move(operands[2]));
                if (!optional_selector_literal.AllValid()) {
                  return CreatePredLiteral(true, Shape(root->shape()));
                }
                std::vector<Literal> new_operands;
                new_operands.emplace_back(std::move(operands[0]));
                new_operands.emplace_back(
                    optional_selector_literal.GetValue()->Clone());
                return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                    ->WithOperands(absl::MakeSpan(new_operands))
                    .WithPrimitiveType(PRED)
                    .Evaluate();
              });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([type,
                                            root]() -> absl::StatusOr<Literal> {
          if (type == PostorderDFSNodeType::kBoundIsDynamic) {
            return CreatePredLiteral(false, Shape(root->shape()));
          } else {
            if (root->literal().shape().element_type() == TUPLE) {
              return Literal::CreateFromProto(
                  root->literal().tuple_literals(1));
            } else if (type == PostorderDFSNodeType::kValueIsDynamic) {
              return CreatePredLiteral(true, Shape(root->shape()));
            } else {
              return Literal::CreateFromProto(root->literal());
            }
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return result.AddVisit([](Literal operand) { return operand; });
      } else {
        return InvalidArgument(
            "Dynamic inferencing on custom call %s is not supported",
            root->DebugString());
      }
      break;
    }
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kWhile: {
      return PostorderDFSNode().AddVisit([root, context]()
                                             -> absl::StatusOr<Literal> {
        return CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
      });
      break;
    }
    default:
      return PostorderDFSNode().AddVisit([root, context]()
                                             -> absl::StatusOr<Literal> {
        return CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
      });
  }
}
absl::StatusOr<Literal> PostorderDFSVisitor::PostOrderDFSVisit(
    int64_t handle, PostorderDFSNodeType type) {
  enum VisitState {
    kUnvisited = 0,
    kVisiting,
    kVisited,
  };
  int64_t unique_id = 0;
  struct WorkItem {
    explicit WorkItem(int64_t handle, InferenceContext context,
                      PostorderDFSNodeType type, VisitState state, int64_t id)
        : handle(handle),
          context(std::move(context)),
          type(type),
          state(state),
          id(id) {}
    int64_t handle;  
    InferenceContext context;
    PostorderDFSNodeType type;
    VisitState state;
    Visit visit;  
    int64_t id;   
    std::vector<CacheKey> dependencies;
    CacheKey GetCacheKey() { return CacheKey(handle, context, type); }
  };
  std::vector<WorkItem> stack;
  WorkItem root(handle, InferenceContext({}, {}), type, kUnvisited,
                unique_id++);
  stack.push_back(root);
  while (!stack.empty()) {
    WorkItem& item = stack.back();
    VLOG(1) << "stack top shape index: " << item.context.shape_index.ToString();
    if (VLOG_IS_ON(1)) {
      TF_RETURN_IF_ERROR(handle_to_instruction(item.handle).status());
      VLOG(1) << "stack top "
              << handle_to_instruction(item.handle).value()->DebugString();
    }
    if (item.state == kVisiting) {
      VLOG(1) << "visiting";
      std::vector<Literal> literals;
      literals.reserve(item.dependencies.size());
      for (CacheKey& dep_key : item.dependencies) {
        TF_RET_CHECK(evaluated.contains(dep_key));
        literals.emplace_back(evaluated.at(dep_key).Clone());
      }
      VLOG(1) << "Start visiting with dependency type: "
              << PostorderDFSNodeTypeToString(item.type);
      TF_ASSIGN_OR_RETURN(auto literal, item.visit(absl::MakeSpan(literals)));
      VLOG(1) << "End visiting: " << literal.ToString();
      evaluated[item.GetCacheKey()] = std::move(literal);
      stack.pop_back();
      continue;
    }
    VLOG(1) << "unvisited";
    if (evaluated.contains(item.GetCacheKey())) {
      stack.pop_back();
      continue;
    }
    item.state = kVisiting;
    PostorderDFSNode node;
    switch (item.type) {
      case PostorderDFSNodeType::kConstantValue: {
        VLOG(1) << "constant value";
        TF_ASSIGN_OR_RETURN(node, AnalyzeConstant(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kConstantLowerBound: {
        VLOG(1) << "constant lower bound";
        TF_ASSIGN_OR_RETURN(node, AnalyzeLowerBound(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kConstantUpperBound: {
        VLOG(1) << "constant upper bound";
        TF_ASSIGN_OR_RETURN(node, AnalyzeUpperBound(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kBoundIsDynamic:
      case PostorderDFSNodeType::kValueIsDynamic: {
        VLOG(1) << "value is dynamic";
        TF_ASSIGN_OR_RETURN(
            node, AnalyzeIsDynamic(item.handle, item.type, item.context));
        break;
      }
    }
    item.visit = node.visit;
    const int64_t current_item_id = stack.size() - 1;
    for (const PostorderDFSDep& dep : node.dependencies) {
      TF_ASSIGN_OR_RETURN(auto dependency_inst,
                          handle_to_instruction(dep.handle));
      VLOG(1) << "dependency " << dep.annotation
              << "::" << dependency_inst->DebugString() << "index"
              << dep.context.shape_index << " stack size:" << stack.size();
      stack.emplace_back(dep.handle, dep.context, dep.type, kUnvisited,
                         unique_id++);
      stack[current_item_id].dependencies.push_back(stack.back().GetCacheKey());
    }
  }
  VLOG(1) << "done" << evaluated[root.GetCacheKey()].ToString();
  return evaluated[root.GetCacheKey()].Clone();
}
absl::StatusOr<Literal> ValueInference::AnalyzeIsDynamic(XlaOp op) {
  PostorderDFSVisitor visitor(
      evaluator_,
      [&](int64_t handle) {
        return builder_->LookUpInstructionByHandle(handle);
      },
      [&](int64_t handle) { return &(builder_->embedded_[handle]); });
  auto result = visitor.PostOrderDFSVisit(
      op.handle(), PostorderDFSNodeType::kValueIsDynamic);
  return result;
}
absl::StatusOr<std::optional<int64_t>> ValueInference::CseOpHandle(
    int64_t handle) {
  TF_ASSIGN_OR_RETURN(auto inst, builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(inst->opcode()));
  if (opcode != HloOpcode::kGetDimensionSize) {
    return {std::nullopt};
  }
  int64_t hash = absl::HashOf(inst->operand_ids(0), inst->dimensions(0));
  auto lookup = cse_map_.find(hash);
  if (lookup == cse_map_.end()) {
    cse_map_[hash] = handle;
    return {std::nullopt};
  }
  TF_ASSIGN_OR_RETURN(auto equivalent_op,
                      builder_->LookUpInstructionByHandle(lookup->second));
  if (equivalent_op->opcode() != inst->opcode() ||
      equivalent_op->operand_ids(0) != inst->operand_ids(0) ||
      equivalent_op->dimensions(0) != inst->dimensions(0)) {
    return {std::nullopt};
  }
  int64_t cse = lookup->second;
  if (handle != cse) {
    return {cse};
  }
  return {std::nullopt};
}
absl::StatusOr<Literal> ValueInference::SimplifyOp(int64_t handle) {
  TF_ASSIGN_OR_RETURN(auto cse_handle, CseOpHandle(handle));
  if (cse_handle) {
    return SimplifyOp(*cse_handle);
  }
  TF_ASSIGN_OR_RETURN(auto* inst, builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(inst->opcode()));
  std::vector<Literal> operands;
  auto output_shape = std::make_unique<const Shape>(inst->shape());
  switch (opcode) {
    case HloOpcode::kSlice:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kBroadcast: {
      for (auto operand_id : inst->operand_ids()) {
        TF_ASSIGN_OR_RETURN(auto literal, SimplifyOp(operand_id));
        operands.emplace_back(std::move(literal));
      }
      return std::make_unique<HloProtoEvaluator>(evaluator_, *inst)
          ->WithOperands(absl::MakeSpan(operands))
          .WithPrimitiveType(S64)
          .Evaluate();
    }
    case HloOpcode::kConvert: {
      auto operand =
          builder_->LookUpInstructionByHandle(inst->operand_ids(0)).value();
      if (Shape::Equal()(*output_shape, Shape(operand->shape()))) {
        return SimplifyOp(inst->operand_ids(0));
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
    case HloOpcode::kAdd: {
      if (output_shape->rank() == 0) {
        TF_ASSIGN_OR_RETURN(auto lhs, SimplifyOp(inst->operand_ids(0)));
        TF_ASSIGN_OR_RETURN(auto rhs, SimplifyOp(inst->operand_ids(1)));
        int64_t lhs_handle = lhs.Get<int64_t>({});
        int64_t rhs_handle = rhs.Get<int64_t>({});
        if (lhs_handle == -1 || rhs_handle == -1) {
          return CreateS64Literal(-1, *output_shape);
        }
        std::function<std::optional<int64_t>(int64_t, int64_t)>
            can_be_optimized;
        can_be_optimized = [this, &can_be_optimized](
                               int64_t lhs,
                               int64_t rhs) -> std::optional<int64_t> {
          auto rhs_inst = builder_->LookUpInstructionByHandle(rhs).value();
          HloOpcode rhs_opcode = StringToHloOpcode(rhs_inst->opcode()).value();
          if (rhs_opcode == HloOpcode::kSubtract) {
            auto sub_lhs_handle =
                SimplifyOp(rhs_inst->operand_ids(0)).value().Get<int64_t>({});
            auto sub_rhs_handle =
                SimplifyOp(rhs_inst->operand_ids(1)).value().Get<int64_t>({});
            if (sub_rhs_handle == lhs) {
              return sub_lhs_handle;
            }
          }
          auto lhs_inst = builder_->LookUpInstructionByHandle(lhs).value();
          HloOpcode lhs_opcode = StringToHloOpcode(lhs_inst->opcode()).value();
          if (lhs_opcode == HloOpcode::kAdd) {
            auto add_lhs_handle =
                SimplifyOp(lhs_inst->operand_ids(0)).value().Get<int64_t>({});
            auto add_rhs_handle =
                SimplifyOp(lhs_inst->operand_ids(1)).value().Get<int64_t>({});
            if (auto optimized = can_be_optimized(add_lhs_handle, rhs)) {
              return Add(XlaOp(add_rhs_handle, builder_),
                         XlaOp(optimized.value(), builder_))
                  .handle();
            }
            if (auto optimized = can_be_optimized(add_rhs_handle, rhs)) {
              return Add(XlaOp(add_lhs_handle, builder_),
                         XlaOp(optimized.value(), builder_))
                  .handle();
            }
          }
          return std::nullopt;
        };
        if (auto optimized = can_be_optimized(lhs_handle, rhs_handle)) {
          return LiteralUtil::CreateR0<int64_t>(optimized.value());
        }
        if (auto optimized = can_be_optimized(rhs_handle, lhs_handle)) {
          return LiteralUtil::CreateR0<int64_t>(optimized.value());
        }
        XlaOp new_sum =
            Add(XlaOp(lhs_handle, builder_), XlaOp(rhs_handle, builder_));
        return LiteralUtil::CreateR0<int64_t>(new_sum.handle());
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
    default: {
      if (ShapeUtil::IsScalar(*output_shape)) {
        return LiteralUtil::CreateR0<int64_t>(handle);
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
  }
}
absl::StatusOr<OptionalLiteral> ValueInference::AnalyzeConstant(
    XlaOp op, ValueInferenceMode mode) {
  TF_RETURN_IF_ERROR(builder_->LookUpInstructionByHandle(op.handle()).status());
  PostorderDFSVisitor visitor(
      evaluator_,
      [&](int64_t handle) {
        return builder_->LookUpInstructionByHandle(handle);
      },
      [&](int64_t handle) { return &(builder_->embedded_[handle]); });
  TF_ASSIGN_OR_RETURN(Shape op_shape, builder_->GetShape(op));
  int64_t handle = op.handle();
  if (ShapeUtil::IsScalar(builder_->GetShape(op).value())) {
    TF_ASSIGN_OR_RETURN(auto result, SimplifyOp(handle));
    auto optimized_handle = result.Get<int64_t>({});
    if (optimized_handle != -1) {
      handle = optimized_handle;
    }
  }
  switch (mode) {
    case ValueInferenceMode::kLowerBound: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kBoundIsDynamic));
      if (mask.IsAll(1)) {
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(handle,
                                    PostorderDFSNodeType::kConstantLowerBound));
      return OptionalLiteral(std::move(value), std::move(mask));
    }
    case ValueInferenceMode::kUpperBound: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kBoundIsDynamic));
      if (mask.IsAll(1)) {
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(handle,
                                    PostorderDFSNodeType::kConstantUpperBound));
      return OptionalLiteral(std::move(value), std::move(mask));
    }
    case ValueInferenceMode::kValue: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kValueIsDynamic));
      if (mask.IsAll(1)) {
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(Literal value,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kConstantValue));
      return OptionalLiteral(std::move(value), std::move(mask));
    }
  }
}
}  