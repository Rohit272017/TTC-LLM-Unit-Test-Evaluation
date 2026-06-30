#include "tensorflow/core/distributed_runtime/recent_request_ids.h"
#include <utility>
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
RecentRequestIds::RecentRequestIds(int num_tracked_request_ids, int num_shards)
    : index_buckets_(num_shards > 0 ? num_shards : 1) {
  DCHECK(num_tracked_request_ids >= num_shards);
  const int per_bucket_size = num_tracked_request_ids / index_buckets_.size();
  for (auto& bucket : index_buckets_) {
    mutex_lock l(bucket.mu);
    bucket.circular_buffer.resize(per_bucket_size);
    bucket.set.reserve(per_bucket_size);
  }
}
bool RecentRequestIds::Insert(int64_t request_id) {
  if (request_id == 0) {
    return true;
  }
  const int bucket_index = request_id % index_buckets_.size();
  auto& bucket = index_buckets_[bucket_index];
  mutex_lock l(bucket.mu);
  const bool inserted = bucket.set.insert(request_id).second;
  if (!inserted) {
    return false;
  }
  bucket.set.erase(bucket.circular_buffer[bucket.next_index]);
  bucket.circular_buffer[bucket.next_index] = request_id;
  bucket.next_index = (bucket.next_index + 1) % bucket.circular_buffer.size();
  return true;
}
Status RecentRequestIds::TrackUnique(int64_t request_id,
                                     const string& method_name,
                                     const protobuf::Message& request) {
  if (Insert(request_id)) {
    return absl::OkStatus();
  } else {
    return errors::Aborted("The same ", method_name,
                           " request was received twice. ",
                           request.ShortDebugString());
  }
}
}  