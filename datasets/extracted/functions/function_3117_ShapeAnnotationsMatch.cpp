#include "tensorflow/compiler/jit/test_util.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "tensorflow/compiler/jit/shape_inference.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/public/version.h"
namespace tensorflow {
Status ShapeAnnotationsMatch(
    const Graph& graph, const GraphShapeInfo& shape_info,
    std::map<string, std::vector<PartialTensorShape>> expected_shapes) {
  for (Node* node : graph.op_nodes()) {
    auto sit = shape_info.find(node->name());
    TF_RET_CHECK(sit != shape_info.end())
        << "Missing shape information for node " << node->name();
    std::vector<PartialTensorShape> shapes;
    for (const auto& output : sit->second) shapes.push_back(output.shape);
    auto it = expected_shapes.find(node->name());
    if (it != expected_shapes.end()) {
      if (!PartialTensorShapeUtils::AreIdentical(shapes, it->second)) {
        return errors::InvalidArgument(
            "Shape mismatch for ", node->name(), ". Expected: ",
            PartialTensorShapeUtils::PartialShapeListString(it->second),
            ", actual: ",
            PartialTensorShapeUtils::PartialShapeListString(shapes));
      }
      expected_shapes.erase(it);
    }
  }
  if (!expected_shapes.empty()) {
    std::vector<string> missing;
    missing.reserve(expected_shapes.size());
    for (const auto& entry : expected_shapes) {
      missing.push_back(entry.first);
    }
    return errors::InvalidArgument("Missing shapes for nodes: ",
                                   absl::StrJoin(missing, ","));
  }
  return absl::OkStatus();
}
void DeviceSetup::AddDevicesAndSetUp(
    const std::vector<std::string>& device_names,
    const std::optional<FunctionDef>& fdef) {
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  for (const auto& device_name : device_names) {
    device_count->insert({device_name, 1});
  }
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
  OptimizerOptions opts;
  lib_def_ = std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
                                                         FunctionDefLibrary());
  if (fdef.has_value()) {
    TF_CHECK_OK(lib_def_->AddFunctionDef(*fdef));
  }
  pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), nullptr,
      TF_GRAPH_DEF_VERSION, lib_def_.get(), opts,
      nullptr, nullptr);
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
}
Device* DeviceSetup::GetDevice(const string& device_name) {
  if (device_mgr_ == nullptr) {
    return nullptr;
  }
  string full_device_name = absl::StrCat(
      "/job:localhost/replica:0/task:0/device:", device_name, ":0");
  Device* device;
  TF_CHECK_OK(device_mgr_->LookupDevice(full_device_name, &device));
  return device;
}
}  