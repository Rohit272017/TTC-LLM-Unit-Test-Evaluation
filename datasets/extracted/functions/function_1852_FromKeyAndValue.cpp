#include "xla/autotune_result_wrapper.h"
#include <vector>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
namespace xla {
 absl::StatusOr<AutotuneResultWrapper>
AutotuneResultWrapper::FromKeyAndValue(OpaqueKey key, OpaqueValue value) {
  AutotuneResults key_proto;
  if (!key_proto.ParseFromString(key)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Could not parse the provided key");
  }
  AutotuneResults::Entry value_entry;
  if (!value_entry.ParseFromString(value)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Could not parse the provided value");
  }
  AutotuneResults::Entry full_entry;
  full_entry.set_device(key_proto.results(0).device());
  full_entry.set_hlo(key_proto.results(0).hlo());
  *full_entry.mutable_result() = value_entry.result();
  return AutotuneResultWrapper(full_entry, key_proto.version());
}
AutotuneResultWrapper::OpaqueKey AutotuneResultWrapper::Key() const {
  AutotuneResults key_proto;
  key_proto.set_version(version_);
  auto entry = key_proto.add_results();
  entry->set_device(autotune_result_.device());
  entry->set_hlo(autotune_result_.hlo());
  OpaqueKey serialized;
  CHECK(tsl::SerializeToStringDeterministic(key_proto, &serialized));
  return serialized;
}
AutotuneResultWrapper::OpaqueValue AutotuneResultWrapper::Value() const {
  AutotuneResults::Entry entry;
  *entry.mutable_result() = autotune_result_.result();
  OpaqueValue serialized;
  CHECK(tsl::SerializeToStringDeterministic(entry, &serialized));
  return serialized;
}
 std::vector<AutotuneResultWrapper>
AutotuneResultWrapper::AutotuneResultsToWrappers(
    const AutotuneResults& autotune_results) {
  std::vector<AutotuneResultWrapper> wrappers;
  wrappers.reserve(autotune_results.results_size());
  for (const auto& result : autotune_results.results()) {
    wrappers.push_back(
        AutotuneResultWrapper(result, autotune_results.version()));
  }
  return wrappers;
}
 absl::StatusOr<AutotuneResults>
AutotuneResultWrapper::AutotuneResultsFromWrappers(
    const std::vector<AutotuneResultWrapper>& wrappers) {
  AutotuneResults autotune_results;
  for (const auto& wrapper : wrappers) {
    if (autotune_results.results_size() > 0 &&
        autotune_results.version() != wrapper.version_) {
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          "All wrappers must have the same version number");
    }
    *autotune_results.add_results() = wrapper.autotune_result_;
    autotune_results.set_version(wrapper.version_);
  }
  return autotune_results;
}
}  