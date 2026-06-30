#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <stack>
#include <string>
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/IR/Location.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/TypeRange.h"  
#include "mlir/IR/Value.h"  
#include "mlir/IR/ValueRange.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/stablehlo_type_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/quantization_unit_loc.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/mutex.h"
#include "tsl/platform/protobuf.h"  
namespace mlir::quant {
using ::stablehlo::quantization::Method;
using ::tsl::protobuf::TextFormat;
constexpr int64_t kDefaultVersion = 9;
constexpr StringRef kPlatformCpu = "CPU";
constexpr StringRef kStablehloModuleAttrsAttrName = "_stablehlo_module_attrs";
constexpr StringRef kUsesShapePolymorphismAttr = "jax.uses_shape_polymorphism";
bool IsInLiftedFunc(Operation* op) {
  if (op == nullptr) return false;
  return op->getParentOfType<func::FuncOp>()->hasAttr(kFusedFunctionAttr);
}
bool IsInStableHloOpRegion(Operation* op) {
  if (op == nullptr) return false;
  auto parent_op = op->getParentOp();
  return parent_op != nullptr && stablehlo::IsStablehloOp(parent_op);
}
StringAttr InsertToSymbolTable(Operation& module, Operation& function,
                               const StringRef func_name) {
  static tensorflow::mutex* mtx = new tensorflow::mutex();
  tensorflow::mutex_lock lock(*mtx);
  SymbolTable symbol_table(&module);
  std::string unique_name = func_name.str();
  int32_t uniquing_counter = 0;
  while (symbol_table.lookup(unique_name) != nullptr) {
    ++uniquing_counter;
    unique_name = absl::StrCat(func_name.str(), "_", uniquing_counter);
  }
  function.setAttr("sym_name",
                   StringAttr::get(module.getContext(), unique_name));
  return symbol_table.insert(&function);
}
ValueRange CreateTFPartitionedCallOp(OpBuilder& builder,
                                     const Location location,
                                     const StringRef func_name,
                                     const TypeRange output_types,
                                     const ValueRange args) {
  TF::PartitionedCallOp call_op = builder.create<TF::PartitionedCallOp>(
      location, output_types, args,
      FlatSymbolRefAttr::get(builder.getStringAttr(func_name)),
      "", "", "");
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));
  return call_op.getOutput();
}
ValueRange CreateTFXlaCallModuleOp(OpBuilder& builder, const Location location,
                                   const StringRef func_name,
                                   const TypeRange output_types,
                                   const ValueRange args) {
  MLIRContext* ctx = builder.getContext();
  SmallVector<Attribute> shape_attrs;
  for (const Type result_type : output_types) {
    shape_attrs.push_back(
        tf_type::ShapeAttr::get(ctx, mlir::cast<ShapedType>(result_type)));
  }
  auto empty_array_attr = ArrayAttr::get(ctx, {});
  auto platforms = ArrayAttr::get(ctx, {StringAttr::get(ctx, kPlatformCpu)});
  auto call_op = builder.create<TF::XlaCallModuleOp>(
      location,
      output_types,
      args,
      kDefaultVersion, "",
      ArrayAttr::get(ctx, shape_attrs),
      empty_array_attr,
      platforms,
      empty_array_attr,
      false,
      empty_array_attr);
  call_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                   FlatSymbolRefAttr::get(builder.getStringAttr(func_name)));
  call_op->setAttr(kOriginalStablehloEntryFunctionAttrName,
                   builder.getStringAttr(func_name));
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));
  call_op->setAttr(kStablehloModuleAttrsAttrName,
                   builder.getDictionaryAttr(builder.getNamedAttr(
                       kUsesShapePolymorphismAttr, builder.getBoolAttr(true))));
  return call_op.getOutput();
}
ValueRange CreateFunctionCallOp(OpBuilder& builder, const Location location,
                                const FunctionCallOpType call_op_type,
                                const StringRef func_name,
                                const TypeRange output_types,
                                const ValueRange args) {
  switch (call_op_type) {
    case FunctionCallOpType::TFXlaCallModuleOp:
      return CreateTFXlaCallModuleOp(builder, location, func_name, output_types,
                                     args);
    case FunctionCallOpType::TFPartitionedCallOp:
      return CreateTFPartitionedCallOp(builder, location, func_name,
                                       output_types, args);
  }
}
SmallVector<Operation*> FindOpsFromArgumentsToResults(
    const ArrayRef<Value> arguments, const ArrayRef<Value> results) {
  std::queue<Value> value_queue;
  for (Value result : results) {
    value_queue.push(result);
  }
  absl::flat_hash_set<mlir::detail::ValueImpl*> argument_set;
  for (Value argument : arguments) {
    argument_set.insert(argument.getImpl());
  }
  std::stack<Operation*> op_stack;
  while (!value_queue.empty()) {
    Value current_value = value_queue.front();
    value_queue.pop();
    Operation* defining_node = current_value.getDefiningOp();
    if (defining_node == nullptr) continue;
    op_stack.push(defining_node);
    for (Value arg : defining_node->getOperands()) {
      if (!argument_set.contains(arg.getImpl())) {
        value_queue.push(arg);
      }
    }
  }
  SmallVector<Operation*> sorted_ops;
  absl::flat_hash_set<Operation*> unique_ops;
  while (!op_stack.empty()) {
    Operation* current_op = op_stack.top();
    op_stack.pop();
    if (unique_ops.contains(current_op)) continue;
    sorted_ops.push_back(current_op);
    unique_ops.insert(current_op);
  }
  return sorted_ops;
}
LogicalResult SetAttributeMap(MLIRContext& context,
                              const ArrayRef<NamedAttribute> attributes,
                              const ArrayRef<Operation*> ops) {
  llvm::SmallDenseMap<NamedAttribute, Operation*> attr_to_op_map;
  for (Operation* op : ops) {
    for (const NamedAttribute named_attr : op->getAttrs()) {
      attr_to_op_map.insert({named_attr, op});
    }
  }
  for (int idx : llvm::seq<int>(0, attributes.size())) {
    const NamedAttribute& attribute = attributes[idx];
    if (const auto string_attr =
            mlir::dyn_cast_or_null<StringAttr>(attribute.getValue());
        string_attr != nullptr &&
        string_attr.getValue() == kNullAttributeValue) {
      continue;
    }
    if (std::find_if(
            attr_to_op_map.begin(), attr_to_op_map.end(), [&](auto attr_op) {
              return std::get<0>(attr_op).getName() == attribute.getName();
            }) == attr_to_op_map.end()) {
      emitError(UnknownLoc::get(&context),
                "Could not find attribute: " + attribute.getName().str());
      return failure();
    }
    Operation* owner_op;
    for (const auto& [attr, val] : attr_to_op_map) {
      if (attr.getName() == attribute.getName()) owner_op = val;
    }
    if (stablehlo::IsStablehloOp(owner_op)) {
      owner_op->setAttr(StringRef(attribute.getName()), attribute.getValue());
    } else {
      owner_op = attr_to_op_map[attribute];
      std::string new_attr_map_str{};
      if (owner_op->hasAttr(kAttrMapAttribute)) {
        new_attr_map_str =
            owner_op->getAttrOfType<StringAttr>(kAttrMapAttribute).str();
        absl::StrAppend(&new_attr_map_str, ",");
      }
      const std::string identifier = std::to_string(idx);
      const StringAttr attribute_name = attribute.getName();
      absl::StrAppend(&new_attr_map_str, identifier, ":", attribute_name.str());
      owner_op->setAttr(kAttrMapAttribute,
                        StringAttr::get(&context, new_attr_map_str));
    }
  }
  return success();
}
SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder& builder, const Location location,
    const FunctionCallOpType call_op_type, const StringRef func_name,
    const ArrayRef<Value> arguments, const ArrayRef<Value> results,
    const ArrayRef<NamedAttribute> attributes) {
  MLIRContext* context = builder.getContext();
  if (results.empty()) {
    emitError(UnknownLoc::get(context), "No result values specified");
    return {};
  }
  Operation* result_op = results[0].getDefiningOp();
  auto module = result_op->getParentOfType<ModuleOp>();
  auto current_func = result_op->getParentOfType<func::FuncOp>();
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointAfter(current_func);
  TypeRange arg_types{ValueRange{arguments}};
  TypeRange result_types{ValueRange{results}};
  auto func_type = FunctionType::get(context, arg_types, result_types);
  SmallVector<Location> arg_locs;
  for (Value arg : arguments) {
    arg_locs.push_back(arg.getLoc());
  }
  auto wrap_func = builder.create<func::FuncOp>(location, func_name, func_type);
  wrap_func.setVisibility(SymbolTable::Visibility::Private);
  if (call_op_type == FunctionCallOpType::TFXlaCallModuleOp) {
    wrap_func->setAttr(TF::kFromXlaCallModuleAttrName, builder.getUnitAttr());
  }
  wrap_func->setAttr(kFusedFunctionAttr, builder.getUnitAttr());
  builder.createBlock(&wrap_func.getBody(), wrap_func.begin(), arg_types,
                      arg_locs);
  IRMapping mapping;
  for (int32_t i : llvm::seq<int32_t>(0, arguments.size())) {
    mapping.map(arguments[i], wrap_func.getArgument(i));
  }
  auto cloning_ops = FindOpsFromArgumentsToResults(arguments, results);
  Location call_op_loc = location;
  for (Operation* op : cloning_ops) {
    std::optional<QuantizationUnitLoc::QuantizationUnit> unit =
        FindQuantizationUnitFromLoc(op->getLoc());
    if (unit.has_value()) {
      call_op_loc = QuantizationUnitLoc(builder.getContext(), unit.value());
    }
  }
  if (failed(SetAttributeMap(*context, attributes, cloning_ops))) {
    current_func.emitError() << "Some attributes couldn't be found.";
  }
  for (Operation* op : cloning_ops) {
    builder.clone(*op, mapping);
  }
  SmallVector<Value> return_values;
  for (Value result : results) {
    return_values.push_back(mapping.lookupOrNull(result));
  }
  builder.create<func::ReturnOp>(location, return_values);
  StringAttr new_func_name =
      InsertToSymbolTable(*module, *wrap_func, func_name);
  builder.setInsertionPointAfter(result_op);
  ValueRange new_results =
      CreateFunctionCallOp(builder, call_op_loc, call_op_type,
                           new_func_name.getValue(), result_types, arguments);
  return SmallVector<Value, 4>(new_results.begin(), new_results.end());
}
SmallVector<Value, 4> LiftAsFunctionCall(OpBuilder& builder,
                                         const Location location,
                                         const FunctionCallOpType call_op_type,
                                         const StringRef func_name,
                                         const ArrayRef<Value> arguments,
                                         const ArrayRef<Value> results) {
  SmallVector<NamedAttribute> attributes;
  return LiftAsFunctionCall(builder, location, call_op_type, func_name,
                            arguments, results, attributes);
}
SmallVector<Value> AppendToVector(const ArrayRef<Value> arguments,
                                  Value append) {
  SmallVector<Value> ret(arguments);
  ret.push_back(append);
  return ret;
}
bool IsEinsumSupportedByXlaDotV2(StringAttr equation_attr) {
  StringRef equation = equation_attr.getValue();
  if (!absl::StrContains(equation, "->") || !absl::StrContains(equation, ",") ||
      absl::StrContains(equation, ".")) {
    return false;
  }
  int idx_arrow = equation.find("->");
  StringRef calc_eq = equation.substr(0, idx_arrow);
  StringRef out_eq = equation.substr(idx_arrow + 2);
  int idx_comma = calc_eq.find(',');
  StringRef lhs_eq = calc_eq.substr(0, idx_comma);
  StringRef rhs_eq = calc_eq.substr(idx_comma + 1);
  if (absl::StrContains(rhs_eq, ",")) return false;
  int lhs_out_idx_start = out_eq.size();
  int lhs_out_idx_end = -1;
  int rhs_out_idx_start = out_eq.size();
  int rhs_out_idx_end = -1;
  int lhs_batch_dim_size = 0;
  int rhs_batch_dim_size = 0;
  for (const char c : lhs_eq) {
    if (absl::StrContains(out_eq, c) && absl::StrContains(rhs_eq, c)) {
      lhs_batch_dim_size++;
    } else if (absl::StrContains(out_eq, c)) {
      const int out_idx = out_eq.find(c);
      if (out_idx < lhs_out_idx_end) {
        return false;
      }
      lhs_out_idx_start = std::min(lhs_out_idx_start, out_idx);
      lhs_out_idx_end = std::max(lhs_out_idx_end, out_idx);
    }
  }
  for (const char c : rhs_eq) {
    if (absl::StrContains(out_eq, c) && absl::StrContains(lhs_eq, c)) {
      rhs_batch_dim_size++;
    } else if (absl::StrContains(out_eq, c)) {
      int out_idx = out_eq.find(c);
      if (out_idx < rhs_out_idx_end) {
        return false;
      }
      if (out_idx < rhs_out_idx_start) rhs_out_idx_start = out_idx;
      if (out_idx > rhs_out_idx_end) rhs_out_idx_end = out_idx;
    }
  }
  if (lhs_batch_dim_size != rhs_batch_dim_size && lhs_batch_dim_size != 0 &&
      rhs_batch_dim_size != 0) {
    return false;
  }
  if (lhs_out_idx_end > rhs_out_idx_start) return false;
  int batch_dim_size = std::max(rhs_batch_dim_size, lhs_batch_dim_size);
  return lhs_out_idx_start >= batch_dim_size &&
         rhs_out_idx_start >= batch_dim_size;
}
absl::StatusOr<Method> GetQuantizationMethod(absl::Nonnull<Operation*> op) {
  const auto quantization_method_attr =
      op->getAttrOfType<StringAttr>(kQuantizationMethodAttr);
  if (!quantization_method_attr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Attribute ", kQuantizationMethodAttr.str(), " is not found."));
  }
  Method quantization_method;
  const std::string method_txtpb = quantization_method_attr.getValue().str();
  if (!TextFormat::ParseFromString(method_txtpb, &quantization_method)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse Method from textproto: ", method_txtpb));
  }
  return quantization_method;
}
Method GetQuantizationMethodOrDefault(absl::Nonnull<Operation*> op) {
  absl::StatusOr<Method> method = GetQuantizationMethod(op);
  if (method.status().code() == absl::StatusCode::kInternal) {
    op->emitError(absl::StrCat("Failed to get quantization method: ",
                               method.status().ToString()));
  }
  return method.ok() ? *method : Method::default_instance();
}
bool HasWeightOnlyPtqMethod(TF::XlaCallModuleOp xla_call_module_op) {
  Method method = GetQuantizationMethodOrDefault(xla_call_module_op);
  return method.has_weight_only_ptq();
}
bool IsWeightOnlyQuantizableOp(const Operation& op) {
  if (auto call_op = dyn_cast<TF::XlaCallModuleOp>(op)) {
    StringRef entry_function_name = GetEntryFunctionName(call_op);
    absl::StatusOr<Method> quantization_method = GetQuantizationMethod(call_op);
    return ContainsConvOrDot(entry_function_name) && quantization_method.ok() &&
           quantization_method->has_weight_only_ptq();
  }
  return false;
}
SmallVector<func::FuncOp> GetSortedFunctions(ModuleOp module_op) {
  auto iterator_range = module_op.getOps<func::FuncOp>();
  SmallVector<func::FuncOp> func_ops(iterator_range.begin(),
                                     iterator_range.end());
  absl::c_sort(func_ops, [](func::FuncOp op1, func::FuncOp op2) {
    return op1.getName() < op2.getName();
  });
  return func_ops;
}
}  