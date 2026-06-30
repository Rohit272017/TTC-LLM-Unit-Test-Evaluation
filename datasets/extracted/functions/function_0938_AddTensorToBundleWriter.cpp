#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace tensorflow {
namespace quantization {
namespace {
using ::mlir::func::FuncOp;
using ::mlir::tf_saved_model::GetInitializerFunction;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
absl::StatusOr<std::string> AddTensorToBundleWriter(
    mlir::TF::AssignVariableOp assign_var_op, BundleWriter& bundle_writer) {
  auto resource_operand = assign_var_op.getOperand(0);
  auto var_handle_op =
      llvm::dyn_cast<mlir::TF::VarHandleOp>(resource_operand.getDefiningOp());
  if (!var_handle_op) {
    assign_var_op->emitRemark(
        "Operand idx 0 is not a tf.VarHandleOp. The initializing tensor is not "
        "saved to checkpoint.");
    return "";
  }
  auto assigned_value_operand = assign_var_op.getOperand(1);
  auto const_op =
      llvm::dyn_cast<mlir::TF::ConstOp>(assigned_value_operand.getDefiningOp());
  if (!const_op) {
    assign_var_op->emitRemark(
        "Operand idx 1 is not a tf.ConstOp. The initializing tensor is not "
        "saved to checkpoint.");
    return "";
  }
  Tensor const_tensor{};
  if (const absl::Status status = mlir::tfg::ConvertToTensor(
          const_op.getValue(), &const_tensor);
      !status.ok()) {
    return status;
  }
  if (!bundle_writer.Add(var_handle_op.getSharedName(), const_tensor)
           .ok()) {
    return bundle_writer.status();
  }
  return var_handle_op.getSharedName().str();
}
}  
absl::StatusOr<std::vector<std::string>> SaveVariablesToCheckpoint(
    const absl::string_view prefix, mlir::ModuleOp module_op) {
  FuncOp session_init_func_type_restore_op = GetInitializerFunction(
      module_op, kTfSavedModelInitializerRestoreType);
  if (!session_init_func_type_restore_op) {
    LOG(INFO) << "No session initializer function with type 'restore_op'. No "
                 "variables are saved to checkpoint.";
    return std::vector<std::string>{};
  }
  BundleWriter bundle_writer(Env::Default(), prefix);
  if (!bundle_writer.status().ok()) {
    return bundle_writer.status();
  }
  std::vector<std::string> saved_variable_shared_names;
  for (auto assign_variable_op :
       session_init_func_type_restore_op.getOps<mlir::TF::AssignVariableOp>()) {
    if (const absl::StatusOr<std::string> variable_shared_name =
            AddTensorToBundleWriter(assign_variable_op, bundle_writer);
        !variable_shared_name.ok()) {
      return variable_shared_name.status();
    } else if (!variable_shared_name->empty()) {
      saved_variable_shared_names.emplace_back(
          std::move(*variable_shared_name));
      VLOG(1) << "Saved a variable with shared_name: " << *variable_shared_name;
    }
  }
  if (saved_variable_shared_names.empty()) {
    LOG(INFO) << "No variables are saved to checkpoint";
    return saved_variable_shared_names;
  }
  if (!bundle_writer.Finish().ok()) {
    return bundle_writer.status();
  }
  return saved_variable_shared_names;
}
}  
}  