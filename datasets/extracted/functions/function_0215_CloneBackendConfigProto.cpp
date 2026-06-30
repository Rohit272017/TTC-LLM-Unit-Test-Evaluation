#include "xla/hlo/ir/backend_config.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/protobuf.h"
namespace xla {
std::unique_ptr<tsl::protobuf::Message> CloneBackendConfigProto(
    const tsl::protobuf::Message* proto) {
  if (proto == nullptr) {
    return nullptr;
  }
  std::unique_ptr<tsl::protobuf::Message> result(proto->New());
  result->CopyFrom(*proto);
  return result;
}
absl::StatusOr<std::string> BackendConfigToRawString(
    const tsl::protobuf::Message& proto) {
  return tsl::ProtoToHumanReadableJson(proto, true);
}
const std::string& BackendConfigWrapper::GetRawStringWithoutMutex() const {
  if (proto_ && raw_string_.empty()) {
    raw_string_ = BackendConfigToRawString(*proto_).value();
  }
  static const std::string* kEmptyString = new std::string();
  return raw_string_.empty() ? *kEmptyString : raw_string_;
}
absl::Status BackendConfigWrapper::GetProto(
    tsl::protobuf::Message* output_proto) const {
  output_proto->Clear();
  absl::WriterMutexLock lock{&mutex_};
  if (proto_ != nullptr) {
    if (proto_->GetDescriptor() != output_proto->GetDescriptor()) {
      return Internal("Mismatched backend config descriptors.");
    }
    output_proto->CopyFrom(*proto_);
    return absl::OkStatus();
  }
  if (raw_string_.empty()) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(tsl::HumanReadableJsonToProto(raw_string_, output_proto));
  proto_ = CloneBackendConfigProto(output_proto);
  return absl::OkStatus();
}
BackendConfigWrapper& BackendConfigWrapper::operator=(
    BackendConfigWrapper&& other) {
  std::unique_ptr<tsl::protobuf::Message> temp_proto;
  std::string temp_string;
  {
    absl::MutexLock other_lock{&other.mutex_};
    temp_proto = std::move(other.proto_);
    temp_string = std::move(other.raw_string_);
  }
  absl::MutexLock this_lock{&mutex_};
  proto_ = std::move(temp_proto);
  raw_string_ = std::move(temp_string);
  return *this;
}
bool BackendConfigWrapper::operator==(const BackendConfigWrapper& other) const {
  tsl::protobuf::Message* this_proto = nullptr;
  {
    absl::MutexLock this_lock{&mutex_};
    this_proto = proto_.get();
  }
  const std::string* other_raw_string = nullptr;
  {
    absl::MutexLock other_lock{&other.mutex_};
    if (this_proto != nullptr && other.proto_ != nullptr) {
      using ::tsl::protobuf::util::MessageDifferencer;
      return MessageDifferencer::Equals(*this_proto, *other.proto_);
    }
    other_raw_string = &other.GetRawStringWithoutMutex();
  }
  return GetRawString() == *other_raw_string;
}
}  