#include <memory>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Serialization.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/serdes.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
namespace {
class HloProgramSerDes : public llvm::RTTIExtends<HloProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::XlaProgram";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const auto& program = llvm::cast<HloProgram>(serializable);
    if (program.mlir_module == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }
    mlir::OwningOpRef<mlir::ModuleOp> module(
        llvm::cast<mlir::ModuleOp>(program.mlir_module->clone()));
    TF_ASSIGN_OR_RETURN(std::string serialized,
                        xla::SerializeUsingVersionedStablehlo(
                            *module, xla::GetDefaultStablehloVersion()));
    return serialized;
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    auto context = std::make_unique<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(context.get());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::stablehlo::deserializePortableArtifact(serialized, context.get());
    if (!module) {
      const absl::Status status = diagnostic_handler.ConsumeStatus();
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to deserialize StableHLO module;\n\nDetailed "
                       "error from MLIR: ",
                       status.message()));
    }
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    if (!mlir::succeeded(pm.run(*module))) {
      const absl::Status status = diagnostic_handler.ConsumeStatus();
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to legalize StableHLO to MHLO;\n\nDetailed error from MLIR: ",
          status.message()));
    }
    return std::make_unique<HloProgram>(std::move(context), std::move(module));
  }
  static char ID;  
};
char HloProgramSerDes::ID = 0;  
bool register_xla_program_serdes = ([]() {
  RegisterSerDes<HloProgram>(std::make_unique<HloProgramSerDes>());
}(), true);
}  
}  
}  