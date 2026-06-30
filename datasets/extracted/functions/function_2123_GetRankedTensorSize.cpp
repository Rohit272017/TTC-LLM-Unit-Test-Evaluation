#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"
#include <algorithm>
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Block.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/constants.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"  
namespace tensorflow {
namespace tfrt_compiler {
namespace {
constexpr int64_t kDefaultCheapCost = 1;
int64_t GetRankedTensorSize(mlir::TensorType type) {
  auto shape = type.getShape();
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= std::max(kDefaultCheapCost, dim);
  }
  return size;
}
int64_t InferTensorSize(const CostContext& context, mlir::TensorType type) {
  if (type.hasRank()) return GetRankedTensorSize(type);
  return context.default_unranked_tensor_size;
}
int64_t InferLookupTableFindV2Cost(const CostContext& context,
                                   mlir::TF::LookupTableFindV2Op op) {
  constexpr int64_t kLookupTableFindCostScale = 8;
  constexpr int64_t kLookupTableFindStringKeyCostScale = 16;
  auto value_type = mlir::cast<mlir::TensorType>(op.getValues().getType());
  auto key_type = mlir::cast<mlir::TensorType>(op.getKeys().getType());
  int64_t output_size = InferTensorSize(context, value_type);
  int64_t cost = kLookupTableFindCostScale * output_size;
  if (mlir::isa<mlir::TF::StringType>(key_type.getElementType()))
    cost *= kLookupTableFindStringKeyCostScale;
  return cost;
}
int64_t InferGatherV2Cost(const CostContext& context, mlir::TF::GatherV2Op op) {
  return InferTensorSize(
      context, mlir::cast<mlir::TensorType>(op.getOutput().getType()));
}
template <typename OpType>
int64_t InferSparseSegmentOpCost(const CostContext& context, OpType op) {
  return InferTensorSize(
      context, mlir::cast<mlir::TensorType>(op.getOutput().getType()));
}
using CostFunctionRegistry = absl::flat_hash_map<std::string, CostFunction>;
void RegisterCostFunction(CostFunctionRegistry& registry,
                          absl::string_view op_name,
                          CostFunction cost_function) {
  auto r = registry.try_emplace(op_name, std::move(cost_function));
  assert(r.second);
  (void)r;
}
template <typename OpType, typename F>
void RegisterCostFunction(CostFunctionRegistry& registry, F f) {
  RegisterCostFunction(
      registry, OpType::getOperationName().str(),
      [f = std::move(f)](const CostContext& context, mlir::Operation* op) {
        return f(context, llvm::cast<OpType>(op));
      });
}
CostFunctionRegistry& GetCostFunctionRegistry() {
  static auto* const registry = []() {
    auto* registry = new CostFunctionRegistry;
    RegisterCostFunction<mlir::TF::GatherV2Op>(*registry, InferGatherV2Cost);
    RegisterCostFunction<mlir::TF::SparseSegmentSumOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSumOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentMeanOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentMeanOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentSqrtNOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSqrtNOp>);
    RegisterCostFunction<mlir::TF::LookupTableFindV2Op>(
        *registry, InferLookupTableFindV2Cost);
    return registry;
  }();
  return *registry;
}
}  
void RegisterCostFunction(absl::string_view op_name,
                          CostFunction cost_function) {
  RegisterCostFunction(GetCostFunctionRegistry(), op_name,
                       std::move(cost_function));
}
bool HasCostFunctionRegistered(absl::string_view op_name) {
  return GetCostFunctionRegistry().contains(op_name);
}
int64_t CostAnalysis::GetCost(mlir::Operation* op) const {
  assert(cost_map_.count(op) > 0);
  return cost_map_.lookup(op);
}
void CostAnalysis::AnalyzeArguments(mlir::func::FuncOp func_op) {
  for (auto arg : func_op.getArguments()) {
    if (!mlir::isa<mlir::TensorType>(arg.getType())) continue;
    auto type = mlir::cast<mlir::TensorType>(arg.getType());
    if (type.hasRank()) {
      max_arg_size_ = std::max(max_arg_size_, GetRankedTensorSize(type));
    }
  }
}
void CostAnalysis::AnalyzeBlock(mlir::Block* block) {
  for (auto& op : *block) {
    EvaluateCost(&op);
  }
}
void CostAnalysis::EvaluateCost(mlir::Operation* op) {
  if (auto cost_function =
          mlir::dyn_cast<tfrt::compiler::CostFunctionInterface>(op)) {
    cost_map_[op] = cost_function.cost();
    return;
  }
  if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) {
    cost_map_[op] = max_arg_size_;
    return;
  }
  const auto& registry = GetCostFunctionRegistry();
  absl::string_view op_name = op->getName().getStringRef();
  auto iter = registry.find(op_name);
  if (iter != registry.end()) {
    CostContext context;
    context.default_unranked_tensor_size = max_arg_size_;
    cost_map_[op] = iter->second(context, op);
    return;
  }
  if (cost_recorder_ != nullptr) {
    const auto op_key_attr =
        op->getAttrOfType<mlir::IntegerAttr>(kOpKeyAttrName);
    if (op_key_attr) {
      cost_map_[op] = cost_recorder_->GetCost(op_key_attr.getInt());
      return;
    }
  }
  if (llvm::isa<mlir::TF::ShapeOp, mlir::TF::StridedSliceOp,
                mlir::TF::ReshapeOp, mlir::TF::ExpandDimsOp>(op)) {
    cost_map_[op] = kDefaultCheapCost;
    return;
  }
  int64_t cost = kDefaultCheapCost;
  for (auto operand : op->getOperands()) {
    auto type = mlir::cast<mlir::TensorType>(operand.getType());
    if (type.hasRank()) {
      cost += GetRankedTensorSize(type);
    } else {
      cost += max_arg_size_;
    }
  }
  cost_map_[op] = cost;
}
}  
}  