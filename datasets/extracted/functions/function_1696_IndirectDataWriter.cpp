#include "tensorstore/kvstore/ocdbt/io/indirect_data_writer.h"
#include <stddef.h>
#include <cassert>
#include <string>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
namespace internal_ocdbt {
namespace {
auto& indirect_data_writer_histogram =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/ocdbt/indirect_data_write_size",
        internal_metrics::MetricMetadata(
            "Histogram of OCDBT buffered write sizes.",
            internal_metrics::Units::kBytes));
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");
}  
class IndirectDataWriter
    : public internal::AtomicReferenceCount<IndirectDataWriter> {
 public:
  explicit IndirectDataWriter(kvstore::KvStore kvstore, std::string prefix,
                              size_t target_size)
      : kvstore_(std::move(kvstore)),
        prefix_(std::move(prefix)),
        target_size_(target_size) {}
  kvstore::KvStore kvstore_;
  std::string prefix_;
  size_t target_size_;
  absl::Mutex mutex_;
  size_t in_flight_ = 0;
  bool flush_requested_ = false;
  absl::Cord buffer_;
  Promise<void> promise_;
  DataFileId data_file_id_;
};
void intrusive_ptr_increment(IndirectDataWriter* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<IndirectDataWriter>*>(p));
}
void intrusive_ptr_decrement(IndirectDataWriter* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<IndirectDataWriter>*>(p));
}
namespace {
void MaybeFlush(IndirectDataWriter& self, UniqueWriterLock<absl::Mutex> lock) {
  bool buffer_at_target =
      self.target_size_ > 0 && self.buffer_.size() >= self.target_size_;
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "MaybeFlush: flush_requested=" << self.flush_requested_
      << ", in_flight=" << self.in_flight_
      << ", buffer_at_target=" << buffer_at_target;
  if (buffer_at_target) {
  } else if (!self.flush_requested_ || self.in_flight_ > 0) {
    return;
  }
  self.in_flight_++;
  self.flush_requested_ = false;
  Promise<void> promise = std::exchange(self.promise_, {});
  absl::Cord buffer = std::exchange(self.buffer_, {});
  DataFileId data_file_id = self.data_file_id_;
  lock.unlock();
  indirect_data_writer_histogram.Observe(buffer.size());
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Flushing " << buffer.size() << " bytes to " << data_file_id;
  auto write_future =
      kvstore::Write(self.kvstore_, data_file_id.FullPath(), std::move(buffer));
  write_future.Force();
  write_future.ExecuteWhenReady(
      [promise = std::move(promise), data_file_id = std::move(data_file_id),
       self = internal::IntrusivePtr<IndirectDataWriter>(&self)](
          ReadyFuture<TimestampedStorageGeneration> future) {
        auto& r = future.result();
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Done flushing data to " << data_file_id << ": " << r.status();
        if (!r.ok()) {
          promise.SetResult(r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          promise.SetResult(absl::UnavailableError("Non-unique file id"));
        } else {
          promise.SetResult(absl::OkStatus());
        }
        UniqueWriterLock lock{self->mutex_};
        assert(self->in_flight_ > 0);
        self->in_flight_--;
        MaybeFlush(*self, std::move(lock));
      });
}
}  
Future<const void> Write(IndirectDataWriter& self, absl::Cord data,
                         IndirectDataReference& ref) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Write indirect data: size=" << data.size();
  if (data.empty()) {
    ref.file_id = DataFileId{};
    ref.offset = 0;
    ref.length = 0;
    return absl::OkStatus();
  }
  UniqueWriterLock lock{self.mutex_};
  Future<const void> future;
  if (self.promise_.null() || (future = self.promise_.future()).null()) {
    self.data_file_id_ = GenerateDataFileId(self.prefix_);
    auto p = PromiseFuturePair<void>::Make();
    self.promise_ = std::move(p.promise);
    future = std::move(p.future);
    self.promise_.ExecuteWhenForced(
        [self = internal::IntrusivePtr<IndirectDataWriter>(&self)](
            Promise<void> promise) {
          ABSL_LOG_IF(INFO, ocdbt_logging) << "Force called";
          UniqueWriterLock lock{self->mutex_};
          if (!HaveSameSharedState(promise, self->promise_)) return;
          self->flush_requested_ = true;
          MaybeFlush(*self, std::move(lock));
        });
  }
  ref.file_id = self.data_file_id_;
  ref.offset = self.buffer_.size();
  ref.length = data.size();
  self.buffer_.Append(std::move(data));
  if (self.target_size_ > 0 && self.buffer_.size() >= self.target_size_) {
    MaybeFlush(self, std::move(lock));
  }
  return future;
}
IndirectDataWriterPtr MakeIndirectDataWriter(kvstore::KvStore kvstore,
                                             std::string prefix,
                                             size_t target_size) {
  return internal::MakeIntrusivePtr<IndirectDataWriter>(
      std::move(kvstore), std::move(prefix), target_size);
}
}  
}  