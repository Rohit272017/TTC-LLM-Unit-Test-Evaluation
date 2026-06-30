#include "xla/python/ifrt_proxy/server/version.h"
#include <algorithm>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
namespace xla {
namespace ifrt {
namespace proxy {
absl::StatusOr<int> ChooseVersion(int client_min_version,
                                  int client_max_version,
                                  int server_min_version,
                                  int server_max_version) {
  const int version = std::min(server_max_version, client_max_version);
  if (version < server_min_version || version < client_min_version) {
    return absl::InvalidArgumentError(absl::StrCat(
        "IFRT Proxy client and server failed to agree on the "
        "protocol version; supported versions: client = [",
        client_min_version, ", ", client_max_version, "], server = [",
        server_min_version, ", ", server_max_version, "]"));
  }
  return version;
}
}  
}  
}  