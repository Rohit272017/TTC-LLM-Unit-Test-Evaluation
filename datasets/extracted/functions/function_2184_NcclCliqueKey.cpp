#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/service/global_device_id.h"
#include "tsl/platform/logging.h"
namespace xla::gpu {
NcclCliqueKey::NcclCliqueKey(
    std::vector<GlobalDeviceId> devices, NcclStreamId stream_id,
    AsyncStreamKind stream_kind,
    std::vector<std::vector<GlobalDeviceId>> participant_groups)
    : devices_(std::move(devices)),
      stream_id_(stream_id),
      stream_kind_(stream_kind),
      participant_groups_(std::move(participant_groups)) {
  for (std::vector<GlobalDeviceId>& group : participant_groups_) {
    absl::c_sort(group);
  }
  auto compare_groups = [](const std::vector<GlobalDeviceId>& lhs,
                           const std::vector<GlobalDeviceId>& rhs) {
    CHECK(!lhs.empty());
    CHECK(!rhs.empty());
    return lhs[0] < rhs[0];
  };
  absl::c_sort(participant_groups_, compare_groups);
}
absl::Span<const GlobalDeviceId> NcclCliqueKey::devices() const {
  return devices_;
}
NcclStreamId NcclCliqueKey::stream_id() const { return stream_id_; }
std::optional<int64_t> NcclCliqueKey::rank(GlobalDeviceId id) const {
  if (auto it = absl::c_find(devices_, id); it != devices_.end()) {
    return it - devices_.begin();
  }
  return std::nullopt;
}
bool NcclCliqueKey::IsSubsetOf(const NcclCliqueKey& other) const {
  return stream_id_ == other.stream_id_ &&
         absl::c_all_of(devices_, [&](GlobalDeviceId id) {
           return absl::c_linear_search(other.devices_, id);
         });
}
std::string NcclCliqueKey::ToString() const {
  std::string group_string = "";
  if (!participant_groups_.empty()) {
    std::vector<std::string> values;
    values.reserve(participant_groups_.size());
    for (const auto& group : participant_groups_) {
      values.push_back("[" + GlobalDeviceIdsToString(group) + "]");
    }
    group_string = absl::StrFormat("; groups=[%s]", absl::StrJoin(values, ","));
  }
  return absl::StrFormat("devices=[%s]; stream=%d%s",
                         GlobalDeviceIdsToString(devices_), stream_id_.value(),
                         group_string);
}
bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b) {
  return a.devices_ == b.devices_ && a.stream_id_ == b.stream_id_ &&
         a.participant_groups_ == b.participant_groups_;
}
bool operator<(const NcclCliqueKey& a, const NcclCliqueKey& b) {
  if (a.devices_.size() < b.devices_.size()) return true;
  if (b.devices_.size() < a.devices_.size()) return false;
  if (a.devices_ < b.devices_) return true;
  if (b.devices_ < a.devices_) return false;
  return a.stream_id_.value() < b.stream_id_.value();
}
bool operator>(const NcclCliqueKey& a, const NcclCliqueKey& b) {
  if (a.devices_.size() > b.devices_.size()) return true;
  if (b.devices_.size() > a.devices_.size()) return false;
  if (a.devices_ > b.devices_) return true;
  if (b.devices_ > a.devices_) return false;
  return a.stream_id_.value() < b.stream_id_.value();
}
NcclCliqueId::NcclCliqueId() { std::fill(data_.begin(), data_.end(), 0); }
NcclCliqueId::NcclCliqueId(char bytes[kSize]) {
  std::copy(bytes, bytes + kSize, data_.data());
}
absl::StatusOr<NcclCliqueId> NcclCliqueId::FromString(std::string_view str) {
  if (str.size() != kSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid NCCL clique id size: %d , expected %d bytes",
                        str.size(), kSize));
  }
  char bytes[kSize];
  std::copy(str.data(), str.data() + kSize, bytes);
  return NcclCliqueId(bytes);
}
absl::Span<const char> NcclCliqueId::data() const { return data_; }
std::string NcclCliqueId::ToString() const {
  return std::string(data_.data(), data_.size());
}
}  