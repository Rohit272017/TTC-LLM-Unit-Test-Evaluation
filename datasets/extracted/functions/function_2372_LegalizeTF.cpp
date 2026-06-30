#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Quant/IR/Quant.h"  
#include "mlir/Dialect/Shape/IR/Shape.h"  
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  
#include "mlir/Dialect/Tensor/IR/Tensor.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "stablehlo/dialect/ChloOps.h"  
#include "stablehlo/dialect/StablehloOps.h"  
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_with_tf2xla_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "xla/mlir_hlo/mhlo/utils/type_conversion.h"
#include "tensorflow/core/lib/monitoring/counter.h"
namespace mlir {
namespace mhlo {
namespace {
#define GEN_PASS_DEF_LEGALIZETF
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"
auto *mlir_legalization_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_count",
    "Counts the attempts of legalization of ops", "op_name");
auto *mlir_failed_legalization_count = tensorflow::monitoring::Counter<2>::New(
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count",
    "Counts the failure of legalization of ops", "op_name", "legality");
class LegalizeTF : public impl::LegalizeTFBase<LegalizeTF> {
 public:
  explicit LegalizeTF(bool legalize_chlo,
                      std::optional<StringRef> tf2xla_fallback_device_type,
                      bool prefer_tf2xla) {
    legalize_chlo_ = legalize_chlo;
    prefer_tf2xla_ = prefer_tf2xla;
    use_tf2xla_fallback_ = tf2xla_fallback_device_type.has_value();
    if (tf2xla_fallback_device_type.has_value()) {
      device_type_ = tf2xla_fallback_device_type.value().str();
    }
  }
  void runOnOperation() override;
};
#define GEN_PASS_DEF_LEGALIZETFMODULEPASS
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"
RewritePatternSet PatternsIncludeOps(RewritePatternSet &from) {
  RewritePatternSet to(from.getContext());
  for (auto &pattern : from.getNativePatterns()) {
    std::optional<OperationName> pat_op_name = pattern->getRootKind();
    bool include =
        !pat_op_name ||
        IsTypeLegalizedWithMlir(pat_op_name->getRegisteredInfo()->getTypeID());
    if (include) to.add(std::move(pattern));
  }
  to.add(std::move(from.getPDLPatterns()));
  return to;
}
std::string OperationLegalityString(Operation *op,
                                    const ConversionTarget &target) {
  auto op_name = op->getName();
  auto action = target.getOpAction(op_name);
  if (!action.has_value()) {
    return "Unknown";
  }
  switch (action.value_or(ConversionTarget::LegalizationAction::Legal)) {
    case ConversionTarget::LegalizationAction::Legal:
      return "Legal";
    case ConversionTarget::LegalizationAction::Dynamic:
      return "Dynamic";
    case ConversionTarget::LegalizationAction::Illegal:
      return "Illegal";
    default:
      return "Invalid";
  }
}
void IncrementFailedLegalizationCount(Operation *op,
                                      const ConversionTarget &target) {
  auto op_name = op->getName();
  auto name_string = op_name.getStringRef().str();
  auto op_legality = OperationLegalityString(op, target);
  mlir_failed_legalization_count->GetCell(name_string, op_legality)
      ->IncrementBy(1);
}
mlir::LogicalResult ApplyPatterns(Operation *op, RewritePatternSet &patterns,
                                  bool legalize_chlo) {
  ConversionTarget target =
      GetDefaultLegalConversionTargets(*op->getContext(), legalize_chlo);
  DenseSet<Operation *> unconverted_ops;
  ConversionConfig config;
  config.unlegalizedOps = &unconverted_ops;
  auto result = applyPartialConversion(op, target, std::move(patterns), config);
  if (failed(result)) {
    IncrementFailedLegalizationCount(op, target);
  }
  for (const auto &unconverted_op : unconverted_ops) {
    IncrementFailedLegalizationCount(unconverted_op, target);
  }
  return result;
}
LogicalResult legalizeTF(Operation *op, bool legalize_chlo,
                         std::optional<StringRef> tf2xla_fallback_device_type,
                         bool prefer_tf2xla) {
  MLIRContext *context = op->getContext();
  RewritePatternSet legalize_lower_patterns(context);
  PopulateLegalizeTfPatterns(context, &legalize_lower_patterns);
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &legalize_lower_patterns);
  if (tf2xla_fallback_device_type && prefer_tf2xla) {
    VLOG(1) << "TF to XLA legalization patterns are partitioned by op into "
               "either native MLIR legalization, or TF2XLA fallback "
               "legalzation, with a preference toward TF2XLA.";
  } else if (tf2xla_fallback_device_type) {
    VLOG(1) << "TF to XLA legalization patterns include all native patterns "
               "and TF2XLA fallback patterns.";
  } else {
    VLOG(1) << "TF to XLA legalization patterns are native patterns only.";
  }
  RewritePatternSet patterns = (tf2xla_fallback_device_type && prefer_tf2xla)
                                   ? PatternsIncludeOps(legalize_lower_patterns)
                                   : std::move(legalize_lower_patterns);
  Tf2XlaTypeConverter converter;
  if (tf2xla_fallback_device_type) {
    PopulateLegalizeTfWithTf2XlaPatterns(tf2xla_fallback_device_type.value(),
                                         patterns, context, converter,
                                         prefer_tf2xla);
  }
  stablehlo::StablehloToHloTypeConverter hlo_converter;
  if (legalize_chlo) {
    chlo::populateChloToHloPatterns(context, &hlo_converter, &patterns);
  }
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);
  return ApplyPatterns(op, patterns, legalize_chlo);
}
void LegalizeTF::runOnOperation() {
  auto op = getOperation();
  auto op_name = op->getName().getStringRef().str();
  mlir_legalization_count->GetCell(op_name)->IncrementBy(1);
  std::optional<StringRef> tf2xla_fallback_device_type = std::nullopt;
  if (use_tf2xla_fallback_) {
    tf2xla_fallback_device_type = device_type_;
  }
  if (failed(legalizeTF(op, legalize_chlo_, tf2xla_fallback_device_type,
                        prefer_tf2xla_))) {
    signalPassFailure();
  }
}
}  
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFPass(
    bool legalize_chlo, std::optional<StringRef> tf2xla_fallback_device_type,
    bool prefer_tf2xla) {
  return std::make_unique<LegalizeTF>(
      legalize_chlo, tf2xla_fallback_device_type, prefer_tf2xla);
}
}  
}  