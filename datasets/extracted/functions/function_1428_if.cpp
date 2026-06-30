#include <memory>
#include <string>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/plugin_program.h"
#include "xla/python/ifrt/serdes.h"
namespace xla {
namespace ifrt {
namespace {
constexpr absl::string_view kSerializationPrefix =
    "__serialized_plugin_program ";
class PluginProgramSerDes
    : public llvm::RTTIExtends<PluginProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::PluginProgram";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    return absl::StrCat(kSerializationPrefix,
                        llvm::cast<PluginProgram>(serializable).data);
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    if (!absl::StartsWith(serialized, kSerializationPrefix)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Bad serialized ", type_name()));
    }
    absl::string_view data(serialized);
    data.remove_prefix(kSerializationPrefix.size());
    auto result = std::make_unique<PluginProgram>();
    result->data = data;
    return result;
  }
  static char ID;  
};
[[maybe_unused]] char PluginProgramSerDes::ID = 0;
bool register_plugin_program_serdes = ([]() {
  RegisterSerDes<PluginProgram>(
      std::make_unique<PluginProgramSerDes>());
}(), true);
class PluginCompileOptionsSerDes
    : public llvm::RTTIExtends<PluginCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::PluginCompileOptions";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    return "";
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    return std::make_unique<PluginCompileOptions>();
  }
  static char ID;  
};
[[maybe_unused]] char PluginCompileOptionsSerDes::ID = 0;
bool register_plugin_compile_options_serdes = ([]() {
  RegisterSerDes<PluginCompileOptions>(
      std::make_unique<PluginCompileOptionsSerDes>());
}(), true);
}  
}  
}  