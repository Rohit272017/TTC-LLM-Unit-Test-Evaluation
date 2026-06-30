#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/custom_call_program.h"
#include "xla/python/ifrt/custom_call_program.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/program_serdes.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/sharding.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace ifrt {
namespace {
class CustomCallProgramSerDes
    : public llvm::RTTIExtends<CustomCallProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::CustomCallProgram";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const CustomCallProgram& program =
        llvm::cast<CustomCallProgram>(serializable);
    CustomCallProgramProto proto;
    proto.set_type(program.type);
    proto.set_name(program.name);
    absl::CopyCordToString(program.serialized_program_text,
                           proto.mutable_serialized_program_text());
    *proto.mutable_devices() = program.devices->ToProto();
    for (const ArraySpec& spec : program.input_specs) {
      TF_ASSIGN_OR_RETURN(*proto.add_input_specs(), spec.ToProto());
    }
    for (const ArraySpec& spec : program.output_specs) {
      TF_ASSIGN_OR_RETURN(*proto.add_output_specs(), spec.ToProto());
    }
    return proto.SerializeAsString();
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    const auto* deserialize_program_options =
        llvm::cast<DeserializeProgramOptions>(options.get());
    CustomCallProgramProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized CustomCallProgramProto");
    }
    TF_ASSIGN_OR_RETURN(
        tsl::RCReference<DeviceList> devices,
        DeviceList::FromProto(deserialize_program_options->lookup_device,
                              proto.devices()));
    std::vector<ArraySpec> input_specs;
    input_specs.reserve(proto.input_specs_size());
    for (const ArraySpecProto& spec_proto : proto.input_specs()) {
      TF_ASSIGN_OR_RETURN(
          ArraySpec spec,
          ArraySpec::FromProto(deserialize_program_options->lookup_device,
                               spec_proto));
      input_specs.push_back(std::move(spec));
    }
    std::vector<ArraySpec> output_specs;
    output_specs.reserve(proto.output_specs_size());
    for (const ArraySpecProto& spec_proto : proto.output_specs()) {
      TF_ASSIGN_OR_RETURN(
          ArraySpec spec,
          ArraySpec::FromProto(deserialize_program_options->lookup_device,
                               spec_proto));
      output_specs.push_back(std::move(spec));
    }
    return std::make_unique<CustomCallProgram>(
        proto.type(), proto.name(),
        absl::Cord(std::move(*proto.mutable_serialized_program_text())),
        std::move(devices),
        std::move(input_specs),
        std::move(output_specs));
  }
  static char ID;  
};
class CustomCallCompileOptionsSerDes
    : public llvm::RTTIExtends<CustomCallCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::CustomCallCompileOptions";
  }
  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    return "";
  }
  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    if (!serialized.empty()) {
      return absl::InvalidArgumentError(
          "Invalid serialized CustomCallCompileOptions; a serialized "
          "CustomCallCompileOptions is expected to be an empty string");
    }
    return std::make_unique<CustomCallCompileOptions>();
  }
  static char ID;  
};
[[maybe_unused]] char CustomCallProgramSerDes::ID = 0;         
[[maybe_unused]] char CustomCallCompileOptionsSerDes::ID = 0;  
bool register_custom_call_program_serdes = ([]{
  RegisterSerDes<CustomCallProgram>(
      std::make_unique<CustomCallProgramSerDes>());
}(), true);
bool register_custom_call_compile_options_serdes = ([]{
  RegisterSerDes<CustomCallCompileOptions>(
      std::make_unique<CustomCallCompileOptionsSerDes>());
}(), true);
}  
}  
}  