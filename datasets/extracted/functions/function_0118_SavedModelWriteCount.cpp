#include "tensorflow/cc/saved_model/metrics.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "json/config.h"
#include "json/json.h"
#include "json/writer.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
namespace tensorflow {
namespace metrics {
namespace {
auto* saved_model_write_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/count",
    "The number of SavedModels successfully written.", "write_version");
auto* saved_model_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/count",
    "The number of SavedModels successfully loaded.", "write_version");
auto* saved_model_write_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/write/api",
    "The API used to write the SavedModel.", "api_label");
auto* saved_model_read_api = monitoring::Counter<1>::New(
    "/tensorflow/core/saved_model/read/api",
    "The API used to load the SavedModel.", "api_label");
auto* saved_model_write_fingerprint = monitoring::Gauge<std::string, 0>::New(
    "/tensorflow/core/saved_model/write/fingerprint",
    "The fingerprint (saved_model_checksum) of the exported SavedModel.");
auto* saved_model_write_path = monitoring::Gauge<std::string, 0>::New(
    "/tensorflow/core/saved_model/write/path",
    "The path (saved_model_path) of the exported SavedModel.");
auto* saved_model_write_path_and_singleprint =
    monitoring::Gauge<std::string, 0>::New(
        "/tensorflow/core/saved_model/write/path_and_singleprint",
        "The path (saved_model_path) and singleprint (concatenation of "
        "graph_def_program_hash, signature_def_hash, saved_object_graph_hash, "
        "and checkpoint_hash) of the newly written SavedModel.");
auto* saved_model_read_fingerprint = monitoring::Gauge<std::string, 0>::New(
    "/tensorflow/core/saved_model/read/fingerprint",
    "The fingerprint (saved_model_checksum) of the loaded SavedModel.");
auto* saved_model_read_path = monitoring::Gauge<std::string, 0>::New(
    "/tensorflow/core/saved_model/read/path",
    "The path (saved_model_path) of the loaded SavedModel.");
auto* saved_model_read_path_and_singleprint =
    monitoring::Gauge<std::string, 0>::New(
        "/tensorflow/core/saved_model/read/path_and_singleprint",
        "The path (saved_model_path) and singleprint (concatenation of "
        "graph_def_program_hash, signature_def_hash, saved_object_graph_hash, "
        "and checkpoint_hash) of the loaded SavedModel.");
auto* saved_model_found_fingerprint_on_load =
    monitoring::Gauge<std::string, 0>::New(
        "/tensorflow/core/saved_model/found_fingerprint_on_load",
        "Whether or not the fingerprint.pb file was found when loading the "
        "SavedModel.");
auto* checkpoint_write_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/write/write_durations",  
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint write operation.",  
        "api_label"                     
    },
    monitoring::Buckets::Exponential(1000, 1.5, 41));
auto* checkpoint_read_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/read/read_durations",  
        "Distribution of the wall time duration in microseconds of the "
        "checkpoint read operation.",  
        "api_label"                    
    },
    monitoring::Buckets::Exponential(1000, 1.5, 41));
auto* async_checkpoint_write_durations = monitoring::Sampler<1>::New(
    {
        "/tensorflow/core/checkpoint/write/async_write_durations",  
        "Distribution of the wall time duration in microseconds of the async "
        "checkpoint write operation",  
        "api_label"                    
    },
    monitoring::Buckets::Exponential(1000, 1.5, 41));
auto* checkpoint_training_time_saved = monitoring::Counter<1>::New(
    "/tensorflow/core/checkpoint/write/training_time_saved",
    "Total time in microseconds elapsed between two consecutive write "
    "operations in a single job or between Checkpoint construction and the "
    "first write operation.",
    "api_label");
auto* checkpoint_size = monitoring::Counter<2>::New(
    "/tensorflow/core/checkpoint/write/checkpoint_size",
    "Size of checkpoint (.index and sharded data files), rounded to the "
    "nearest 100 MB.",
    "api_label", "filesize");
}  
auto* sharding_callback_duration = monitoring::Counter<0>::New(
    "/tensorflow/core/checkpoint/sharding/callback_duration",
    "Sharding callback execution duration in microseconds.");
auto* num_checkpoint_shards_written = monitoring::Counter<0>::New(
    "/tensorflow/core/checkpoint/sharding/num_checkpoint_shards_written",
    "Number of checkpoint shard files written during saving.");
auto* sharding_callback_description = monitoring::Gauge<std::string, 0>::New(
    "/tensorflow/core/checkpoint/sharding/callback_description",
    "Describes the callback used to shard the checkpoint during saving.");
monitoring::CounterCell& SavedModelWriteCount(absl::string_view write_version) {
  return *saved_model_write_counter->GetCell(std::string(write_version));
}
monitoring::CounterCell& SavedModelReadCount(absl::string_view write_version) {
  return *saved_model_read_counter->GetCell(std::string(write_version));
}
monitoring::CounterCell& SavedModelWriteApi(absl::string_view api_label) {
  return *saved_model_write_api->GetCell(std::string(api_label));
}
monitoring::CounterCell& SavedModelReadApi(absl::string_view api_label) {
  return *saved_model_read_api->GetCell(std::string(api_label));
}
monitoring::GaugeCell<std::string>& SavedModelReadFingerprint() {
  return *saved_model_read_fingerprint->GetCell();
}
monitoring::GaugeCell<std::string>& SavedModelReadPath() {
  return *saved_model_read_path->GetCell();
}
monitoring::GaugeCell<std::string>& SavedModelReadPathAndSingleprint() {
  return *saved_model_read_path_and_singleprint->GetCell();
}
monitoring::GaugeCell<std::string>& SavedModelWriteFingerprint() {
  return *saved_model_write_fingerprint->GetCell();
}
monitoring::GaugeCell<std::string>& SavedModelWritePath() {
  return *saved_model_write_path->GetCell();
}
monitoring::GaugeCell<std::string>& SavedModelWritePathAndSingleprint() {
  return *saved_model_write_path_and_singleprint->GetCell();
}
std::string MakeFingerprintJson(FingerprintDef fingerprint_def) {
  Json::Value fingerprint = Json::objectValue;
  fingerprint["saved_model_checksum"] =
      Json::UInt64(fingerprint_def.saved_model_checksum());
  fingerprint["graph_def_program_hash"] =
      Json::UInt64(fingerprint_def.graph_def_program_hash());
  fingerprint["signature_def_hash"] =
      Json::UInt64(fingerprint_def.signature_def_hash());
  fingerprint["saved_object_graph_hash"] =
      Json::UInt64(fingerprint_def.saved_object_graph_hash());
  fingerprint["checkpoint_hash"] =
      Json::UInt64(fingerprint_def.checkpoint_hash());
  Json::StreamWriterBuilder json_factory;
  return Json::writeString(json_factory, fingerprint);
}
absl::StatusOr<std::string> MakeSavedModelPathAndSingleprint(
    std::string path, std::string singleprint) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        "Invalid path_and_singleprint argument. Empty path.");
  }
  if (singleprint.empty()) {
    return absl::InvalidArgumentError(
        "Invalid path_and_singleprint argument. Empty singleprint.");
  }
  return absl::StrCat(path, ":", singleprint);
}
absl::StatusOr<std::pair<std::string, std::string>>
ParseSavedModelPathAndSingleprint(std::string path_and_singleprint) {
  size_t delimiter = path_and_singleprint.rfind(':');
  if (delimiter == std::string::npos) {
    return absl::InvalidArgumentError(
        "Invalid path_and_singleprint argument. Found no delimeter.");
  }
  std::string path = path_and_singleprint.substr(0, delimiter);
  if (path.empty()) {
    return absl::InvalidArgumentError(
        "Invalid path_and_singleprint argument. Empty path.");
  }
  std::string singleprint = path_and_singleprint.substr(delimiter + 1);
  if (singleprint.empty()) {
    return absl::InvalidArgumentError(
        "Invalid path_and_singleprint argument. Empty singleprint.");
  }
  return std::pair<std::string, std::string>(path, singleprint);
}
monitoring::GaugeCell<std::string>& SavedModelFoundFingerprintOnLoad() {
  return *saved_model_found_fingerprint_on_load->GetCell();
}
monitoring::SamplerCell& CheckpointReadDuration(absl::string_view api_label) {
  return *checkpoint_read_durations->GetCell(std::string(api_label));
}
monitoring::SamplerCell& CheckpointWriteDuration(absl::string_view api_label) {
  return *checkpoint_write_durations->GetCell(std::string(api_label));
}
monitoring::SamplerCell& AsyncCheckpointWriteDuration(
    absl::string_view api_label) {
  return *async_checkpoint_write_durations->GetCell(std::string(api_label));
}
monitoring::CounterCell& TrainingTimeSaved(absl::string_view api_label) {
  return *checkpoint_training_time_saved->GetCell(std::string(api_label));
}
monitoring::CounterCell& CheckpointSize(absl::string_view api_label,
                                        int64_t filesize) {
  return *checkpoint_size->GetCell(std::string(api_label),
                                   std::to_string(filesize));
}
monitoring::CounterCell& ShardingCallbackDuration() {
  return *sharding_callback_duration->GetCell();
}
monitoring::CounterCell& NumCheckpointShardsWritten() {
  return *num_checkpoint_shards_written->GetCell();
}
monitoring::GaugeCell<std::string>& ShardingCallbackDescription() {
  return *sharding_callback_description->GetCell();
}
}  
}  