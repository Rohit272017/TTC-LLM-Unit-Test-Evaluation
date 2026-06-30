#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
int64_t GetUniqueRequestId() {
  int64_t request_id = 0;
  while (request_id == 0) {
    request_id = tsl::random::ThreadLocalNew64();
  }
  return request_id;
}
}  