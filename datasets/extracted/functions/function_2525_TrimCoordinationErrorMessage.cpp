#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"
#include <optional>
#include <string>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/regexp.h"
namespace tsl {
absl::Status TrimCoordinationErrorMessage(const absl::Status& s) {
  if (s.ok()) {
    return s;
  }
  auto status_message = std::string(s.message());
  auto additional_info_index = status_message.find("Additional GRPC");
  if (additional_info_index == std::string::npos) {
    return s;
  }
  std::optional<absl::Cord> payload =
      s.GetPayload(CoordinationErrorPayloadKey());
  if (!payload.has_value() && absl::IsUnavailable(s)) {
    auto prefix_message =
        "Failed to send RPC to coordination service. Either the leader task "
        "died/restarted unexpectedly or this task is experiencing network "
        "issues. Check earlier logs from this task and the "
        "leader (usually slice 0 process/task/worker 0)  to debug further.\n";
    status_message = absl::StrCat(
        prefix_message,
        status_message.substr(additional_info_index));
  } else {
    std::string rpc_name;
    RE2::PartialMatch(status_message,
                      "(/tensorflow.CoordinationService/(\\w+))", &rpc_name);
    status_message = status_message.substr(0, additional_info_index);
    absl::StrAppend(&status_message, "\nRPC: ", rpc_name);
  }
  auto trimmed_status = absl::Status(s.code(), status_message);
  if (payload.has_value()) {
    trimmed_status.SetPayload(CoordinationErrorPayloadKey(), *payload);
  }
#if defined(PLATFORM_GOOGLE)
  for (const auto& source_location : s.GetSourceLocations()) {
    trimmed_status.AddSourceLocation(source_location);
  }
#endif
  return trimmed_status;
}
}  