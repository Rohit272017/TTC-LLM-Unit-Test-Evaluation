#include <memory>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
namespace {
class IfrtIRProgramSerDes
    : public llvm::RTTIExtends<IfrtIRProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::IfrtIRProgram";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const auto& program = llvm::cast<IfrtIRProgram>(serializable);
    if (program.mlir_module == nullptr) {
      return absl::InvalidArgumentError("Unable to serialize null MLIR module");
    }
    std::string serialized;
    llvm::raw_string_ostream out(serialized);
    mlir::BytecodeWriterConfig config;
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(
        program.mlir_module->getContext());
    if (mlir::failed(
            mlir::writeBytecodeToFile(program.mlir_module, out, config))) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to serialize IFRT IR module string: %s",
                          diagnostic_handler.ConsumeStatus().message()));
    }
    return serialized;
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSIGN_OR_RETURN(auto module,
                        support::ParseMlirModuleString(serialized, *context));
    return std::make_unique<IfrtIRProgram>(std::move(context),
                                           std::move(module));
  }
  static char ID;  
};
char IfrtIRProgramSerDes::ID = 0;  
bool register_ifrt_ir_program_serdes = ([]() {
  RegisterSerDes<IfrtIRProgram>(std::make_unique<IfrtIRProgramSerDes>());
}(), true);
}  
}  
}  