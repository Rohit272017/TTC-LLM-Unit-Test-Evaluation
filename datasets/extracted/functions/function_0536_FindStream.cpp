#include "xla/stream_executor/stream_finder.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
namespace stream_executor {
absl::StatusOr<Stream*> FindStream(Platform* platform, void* gpu_stream) {
  int number_devices = platform->VisibleDeviceCount();
  for (int i = 0; i < number_devices; ++i) {
    auto stream_executor = platform->FindExisting(i);
    if (!stream_executor.ok()) {
      continue;
    }
    Stream* found_stream = nullptr;
    if ((found_stream = (*stream_executor)->FindAllocatedStream(gpu_stream)) !=
        nullptr) {
      return found_stream;
    }
  }
  return absl::NotFoundError("Stream not found");
}
}  