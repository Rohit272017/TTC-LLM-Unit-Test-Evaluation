#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gelu.h"
#include <cmath>
#include <cstdlib>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LogicalResult.h"  
#include "stablehlo/dialect/ChloOps.h"  
#include "stablehlo/dialect/StablehloOps.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
namespace mlir::odml {
constexpr float kOne = 1.0;
const float kOneOverRoot2 = kOne / std::sqrt(2);
constexpr float kHalf = kOne / 2.0;
constexpr float kTolerance = kOne / 1000.0;
Operation* GetUserIfOnlyOne(Operation* op) {
  if (op->getNumResults() != 1) return nullptr;
  auto result = op->getResult(0);
  if (!result.hasOneUse()) return nullptr;
  return (*result.getUses().begin()).getOwner();
}
Operation* GetInputOpWithOneUse(Operation* op, int opr_num) {
  if (opr_num >= op->getNumOperands()) return nullptr;
  auto opr = op->getOperand(opr_num);
  if (llvm::isa<BlockArgument>(opr)) return nullptr;
  auto* res = opr.getDefiningOp();
  if (!res->hasOneUse()) return nullptr;
  return res;
}
bool HasSplatArg(Operation* op, float val, int opr_num) {
  auto* cst_input = GetInputOpWithOneUse(op, 1);
  if (!cst_input) return false;
  auto cst_op = llvm::dyn_cast_or_null<stablehlo::ConstantOp>(cst_input);
  if (!cst_op) return false;
  ElementsAttr value = cst_op.getValue();
  if (!value.isSplat()) return false;
  if (!value.getElementType().isF32()) return false;
  return std::abs(value.getSplatValue<float>() - val) < kTolerance;
}
bool MatchERF(Operation* op) {
  if (auto custom_call = llvm::dyn_cast_or_null<stablehlo::CustomCallOp>(op)) {
    return custom_call.getCallTargetName() == "mhlo.erf";
  }
  return llvm::isa<chlo::ErfOp>(op);
}
LogicalResult LowerGELU::matchAndRewrite(Operation* op,
                                         PatternRewriter& rewriter) const {
  if (!MatchERF(op)) return failure();
  auto* erf_user = GetUserIfOnlyOne(op);
  if (!erf_user) return failure();
  auto* erf_user_user = GetUserIfOnlyOne(erf_user);
  if (!erf_user_user) return failure();
  auto* erf_input = GetInputOpWithOneUse(op, 0);
  if (!erf_input) return failure();
  auto* erf_user_user_input = GetInputOpWithOneUse(erf_user_user, 0);
  if (!erf_user_user_input) return failure();
  if (erf_user_user_input->getOperand(0) != erf_input->getOperand(0)) {
    return failure();
  }
  auto rhs_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_input);
  if (!rhs_mul) return failure();
  auto lhs_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user_input);
  if (!lhs_mul) return failure();
  auto output_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user);
  if (!output_mul) return failure();
  auto rhs_add = llvm::dyn_cast_or_null<stablehlo::AddOp>(erf_user);
  if (!rhs_add) return failure();
  if (!HasSplatArg(rhs_add, kOne, 1)) return failure();
  if (!HasSplatArg(lhs_mul, kHalf, 1)) return failure();
  if (!HasSplatArg(rhs_mul, kOneOverRoot2, 1)) return failure();
  auto is_approx_attr = rewriter.getBoolAttr(false);
  auto gelu = rewriter.create<TFL::GeluOp>(
      output_mul.getLoc(), output_mul.getResult().getType(),
      erf_input->getOperand(0), is_approx_attr);
  rewriter.replaceAllOpUsesWith(output_mul, gelu);
  rewriter.eraseOp(output_mul);
  rewriter.eraseOp(rhs_add);
  rewriter.eraseOp(op);
  rewriter.eraseOp(lhs_mul);
  rewriter.eraseOp(rhs_mul);
  return success();
}
}  