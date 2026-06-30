#include "xla/service/stream_pool.h"
#include <memory>
#include <utility>
#include "absl/strings/str_format.h"
namespace xla {
StreamPool::Ptr StreamPool::BorrowStream(se::StreamPriority priority) {
  std::unique_ptr<se::Stream> stream;
  {
    absl::MutexLock lock(&mu_);
    if (streams_with_pri_.find(priority) == streams_with_pri_.end()) {
      stream = nullptr;
    } else {
      while (!streams_with_pri_[priority].empty() && !stream) {
        stream = std::move(streams_with_pri_[priority].back());
        streams_with_pri_[priority].pop_back();
        if (stream->ok()) {
          VLOG(1) << absl::StrFormat(
              "StreamPool reusing existing stream (%p) with priority: %s",
              stream.get(), se::StreamPriorityToString(priority));
        } else {
          VLOG(1) << absl::StrFormat(
              "Stream (%p) was not ok, deleting with : %s", stream.get(),
              se::StreamPriorityToString(priority));
          stream = nullptr;
        }
      }
    }
  }
  if (!stream) {
    stream = executor_->CreateStream(priority).value();
    stream->set_name(absl::StrFormat("%s pool stream",
                                     se::StreamPriorityToString(priority)));
    VLOG(1) << absl::StrFormat("Created new stream (%p) with priority = %s",
                               stream.get(),
                               se::StreamPriorityToString(priority));
  }
  PtrDeleter deleter = {this};
  return Ptr(stream.release(), deleter);
}
void StreamPool::ReturnStream(se::Stream* stream) {
  if (stream->ok()) {
    VLOG(1) << absl::StrFormat("StreamPool returning ok stream (%p)", stream);
    absl::MutexLock lock(&mu_);
    auto priority = std::get<se::StreamPriority>(stream->priority());
    streams_with_pri_[priority].emplace_back(stream);
  } else {
    VLOG(1) << absl::StrFormat("StreamPool deleting !ok stream (%p)", stream);
    delete stream;
  }
}
}  