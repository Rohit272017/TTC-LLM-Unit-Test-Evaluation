#include "xla/tsl/lib/io/inputstream_interface.h"
#include "tsl/platform/errors.h"
namespace tsl {
namespace io {
static constexpr int64_t kMaxSkipSize = 8 * 1024 * 1024;
absl::Status InputStreamInterface::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can't skip a negative number of bytes");
  }
  tstring unused;
  while (bytes_to_skip > 0) {
    int64_t bytes_to_read = std::min<int64_t>(kMaxSkipSize, bytes_to_skip);
    TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &unused));
    bytes_to_skip -= bytes_to_read;
  }
  return absl::OkStatus();
}
}  
}  