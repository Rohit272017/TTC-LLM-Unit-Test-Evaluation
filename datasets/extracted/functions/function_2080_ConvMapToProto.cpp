#include "tensorflow/core/util/autotune_maps/autotune_serialize.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "xla/status_macros.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/autotune_maps/autotune_map.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_autotune_maps.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"
namespace tensorflow {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
using stream_executor::dnn::AlgorithmConfig;
using stream_executor::dnn::AlgorithmConfigProto;
using stream_executor::dnn::AlgorithmDesc;
using stream_executor::dnn::AlgorithmProto;
template <typename Op>
StatusOr<ConvMapProto> ConvMapToProto(
    const AutotuneMap<ConvParameters, AutotuneEntry<Op>> &autotune_map) {
  ConvMapProto proto;
  std::map<string, ConvMapProto::Entry> sorted_map;
  for (auto const &p : autotune_map.GetMap()) {
    const ConvParameters &params = p.first;
    const ConvParametersProto &params_proto = params.proto();
    VLOG(1) << "Reading: " << params.ToString();
    ConvMapProto::Entry kv;
    *kv.mutable_key() = params_proto;
    if (p.second.is_algorithm_config()) {
      *kv.mutable_value() = p.second.GetAlgorithmConfig().ToProto();
    } else {
      const auto &runners = p.second.GetOpRunners();
      *kv.mutable_value()->mutable_algorithm() =
          runners.primary->ToAlgorithmDesc().ToProto();
      if (runners.no_scratch_fallback) {
        *kv.mutable_value()->mutable_algorithm_no_scratch() =
            runners.no_scratch_fallback->ToAlgorithmDesc().ToProto();
      }
    }
    std::string serialized_params;
    TF_RET_CHECK(
        tsl::SerializeToStringDeterministic(params_proto, &serialized_params));
    sorted_map.insert(std::make_pair(std::move(serialized_params), kv));
  }
  for (auto const &p : sorted_map) {
    ConvMapProto::Entry *kv = proto.add_kv_pairs();
    *kv = p.second;
  }
  return proto;
}
template <typename Op>
Status PopulateConvMap(
    const ConvMapProto &m,
    AutotuneMap<ConvParameters, AutotuneEntry<Op>> *autotune_map) {
  if (m.kv_pairs().size() == 0) {
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()));
  std::vector<std::string> device_descs;
  for (int i = 0; i < platform->VisibleDeviceCount(); i++) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::DeviceDescription> device_desc,
                        platform->DescriptionForDevice(i));
    device_descs.push_back(device_desc->model_str());
  }
  std::set<std::string> unmatched_device_descs;
  for (const ConvMapProto::Entry &kv : m.kv_pairs()) {
    const ConvParametersProto &params_proto = kv.key();
    if (params_proto.version() != ConvParameters::kVersion) {
      VLOG(1) << "ConvParametersProto with the incompatible version:"
              << params_proto.DebugString();
      return errors::Aborted(
          "Aborted because the loaded autotune results for convolution "
          "operations have a version different "
          "from runtime's version. Expected version: ",
          ConvParameters::kVersion,
          ". Actual version: ", params_proto.version());
    }
    const AlgorithmConfigProto &algorithm_config_proto = kv.value();
    const AlgorithmDesc primary(algorithm_config_proto.algorithm());
    const absl::optional<AlgorithmDesc> fallback =
        algorithm_config_proto.has_algorithm_no_scratch()
            ? absl::optional<AlgorithmDesc>(
                  AlgorithmDesc(algorithm_config_proto.algorithm_no_scratch()))
            : absl::nullopt;
    bool devices_matched = false;
    for (int ordinal = 0; ordinal < device_descs.size(); ordinal++) {
      const std::string &desc_str = device_descs[ordinal];
      if (desc_str != params_proto.device_identifier()) {
        continue;
      }
      devices_matched = true;
      AutotuneEntry<Op> entry;
#if TENSORFLOW_USE_ROCM
      entry = AutotuneEntry<Op>(AlgorithmConfig(algorithm_config_proto));
#else
      entry = AutotuneEntry<Op>(primary, fallback);
#endif
      autotune_map->Insert(ConvParameters(ordinal, params_proto), entry);
    }
    if (!devices_matched) {
      unmatched_device_descs.insert(params_proto.device_identifier());
    }
  }
  if (!unmatched_device_descs.empty()) {
    LOG(WARNING) << "Unmatched device id's from AoT autotuning data: "
                 << str_util::Join(unmatched_device_descs, ", ")
                 << "; existing devices: "
                 << str_util::Join(device_descs, ", ");
  }
  return OkStatus();
}
}  
#endif  
Status SerializeAutotuneMaps(std::string *output) {
  AutotuneMapsProto proto;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  TF_ASSIGN_OR_RETURN(*proto.mutable_conv_map(),
                      ConvMapToProto(*ConvAutotuneMap::GetInstance()));
  TF_ASSIGN_OR_RETURN(*proto.mutable_fused_conv_map(),
                      ConvMapToProto(*FusedConvAutotuneMap::GetInstance()));
#endif  
  TF_RET_CHECK(tsl::SerializeToStringDeterministic(proto, output));
  return absl::OkStatus();
}
Status LoadSerializedAutotuneMaps(absl::string_view s) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  AutotuneMapsProto proto;
  if (!proto.ParseFromString(string(s))) {
    return errors::InvalidArgument(
        "Failed to parse the autotune maps from string.");
  }
  TF_RETURN_IF_ERROR(
      PopulateConvMap(proto.conv_map(), ConvAutotuneMap::GetInstance()));
  TF_RETURN_IF_ERROR(PopulateConvMap(proto.fused_conv_map(),
                                     FusedConvAutotuneMap::GetInstance()));
#endif  
  return absl::OkStatus();
}
void ResetAutotuneMaps() {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  ConvAutotuneMap::GetInstance()->ClearMap();
  FusedConvAutotuneMap::GetInstance()->ClearMap();
#endif  
}
}  