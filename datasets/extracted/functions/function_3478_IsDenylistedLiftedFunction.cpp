#include "tensorflow/compiler/mlir/quantization/stablehlo/ops/stablehlo_op_quant_spec.h"
#include <memory>
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "stablehlo/dialect/StablehloOps.h"  
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  
#define DEBUG_TYPE "stablehlo_opt_quant_spec"
namespace mlir::quant::stablehlo {
namespace {
using ::mlir::stablehlo::DotGeneralOp;
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::StaticRangePtq;
bool IsDenylistedLiftedFunction(Operation* op) {
  if (auto xla_call_module_op = dyn_cast_or_null<TF::XlaCallModuleOp>(op);
      xla_call_module_op != nullptr) {
    absl::StatusOr<Method> method = GetQuantizationMethod(xla_call_module_op);
    if (method.ok() && method->has_no_quantization()) {
      return true;
    }
  }
  return false;
}
void PopulateCoeffOpQuantDimIfPerChannelQuantized(
    TF::XlaCallModuleOp xla_call_module_op, OpQuantSpec& spec) {
  absl::StatusOr<Method> method = GetQuantizationMethod(xla_call_module_op);
  if (method.ok() && method->has_static_range_ptq()) {
    const StaticRangePtq& static_range_ptq_spec = method->static_range_ptq();
    for (const auto& [operand_idx, quantized_type] :
         static_range_ptq_spec.input_quantized_types()) {
      if (quantized_type.has_dimension_specs()) {
        spec.coeff_op_quant_dim[operand_idx] =
            quantized_type.dimension_specs().dimension();
      }
    }
  }
}
}  
std::unique_ptr<OpQuantSpec> GetStableHloOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast_or_null<TF::XlaCallModuleOp>(op)) {
    auto entry_function =
        call_op->getAttrOfType<FlatSymbolRefAttr>("_entry_function");
    StringRef function_name = entry_function.getValue();
    if (!function_name.starts_with("composite_")) {
      return spec;
    }
    if (function_name.contains("conv")) {
      PopulateCoeffOpQuantDimIfPerChannelQuantized(call_op, *spec);
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    } else if (function_name.contains("dot_general")) {
      const auto module_op = call_op->getParentOfType<ModuleOp>();
      const SymbolTable symbol_table(module_op);
      auto entry_func_op =
          dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(function_name));
      auto dot_general_op = *entry_func_op.getOps<DotGeneralOp>().begin();
      if (auto optional_dim = GetDotGeneralQuantizationDim(dot_general_op);
          optional_dim) {
        spec->coeff_op_quant_dim[1] = optional_dim.value();
      } else {
        spec->coeff_op_quant_dim[1] = -1;
      }
      if (function_name.contains("with_bias")) {
        spec->biases_params[2] = {{0, 1},
                                  quant::GetUniformQuantizedTypeForBias};
      }
    }
    for (const auto [operand_idx, per_channel_dim] : spec->coeff_op_quant_dim) {
      spec->quantizable_operands.insert(operand_idx);
    }
  }
  return spec;
}
std::unique_ptr<OpQuantScaleSpec> GetStableHloQuantConstraints(Operation* op) {
  auto scale_spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<mlir::stablehlo::BroadcastInDimOp,
                mlir::stablehlo::ConcatenateOp,
                mlir::stablehlo::DynamicReshapeOp,
                mlir::stablehlo::DynamicSliceOp, mlir::stablehlo::GatherOp,
                mlir::stablehlo::PadOp, mlir::stablehlo::ReduceWindowOp,
                mlir::stablehlo::ReshapeOp, mlir::stablehlo::SelectOp,
                mlir::stablehlo::SliceOp, mlir::stablehlo::TransposeOp>(op)) {
    scale_spec->has_same_scale_requirement = true;
  }
  if (llvm::isa<mlir::stablehlo::DynamicSliceOp, mlir::stablehlo::GatherOp,
                mlir::stablehlo::PadOp, mlir::stablehlo::SliceOp>(op)) {
    scale_spec->has_same_operand_and_result_type_requirement = true;
  }
  return scale_spec;
}
bool IsOpQuantizableStableHlo(Operation* op) {
  if (isa<func::ConstantOp, mlir::stablehlo::ConstantOp>(op)) {
    return true;
  } else if (op->hasTrait<OpTrait::IsTerminator>() ||
             isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(op)) {
    return false;
  }
  if (IsDenylistedLiftedFunction(op)) {
    LLVM_DEBUG(llvm::errs() << "Denylisted quantizable unit: \n" << op << "\n");
    return false;
  }
  if (GetStableHloQuantConstraints(op)->has_same_scale_requirement) {
    return true;
  }
  const bool attr_enforced_quantizable =
      op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
      op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
          QuantTraitValues[QuantizationTrait::FullyQuantizable];
  return attr_enforced_quantizable;
}
}  