#ifndef QUICHE_HTTP2_CORE_PRIORITY_WRITE_SCHEDULER_H_
#define QUICHE_HTTP2_CORE_PRIORITY_WRITE_SCHEDULER_H_
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "quiche/http2/core/spdy_protocol.h"
#include "quiche/common/platform/api/quiche_bug_tracker.h"
#include "quiche/common/platform/api/quiche_export.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/quiche_circular_deque.h"
namespace http2 {
namespace test {
template <typename StreamIdType>
class PriorityWriteSchedulerPeer;
}
struct QUICHE_EXPORT SpdyPriorityToSpdyPriority {
  spdy::SpdyPriority operator()(spdy::SpdyPriority priority) {
    return priority;
  }
};
template <typename StreamIdType, typename PriorityType = spdy::SpdyPriority,
          typename PriorityTypeToInt = SpdyPriorityToSpdyPriority,
          typename IntToPriorityType = SpdyPriorityToSpdyPriority>
class QUICHE_EXPORT PriorityWriteScheduler {
 public:
  static constexpr int kHighestPriority = 0;
  static constexpr int kLowestPriority = 7;
  static_assert(spdy::kV3HighestPriority == kHighestPriority);
  static_assert(spdy::kV3LowestPriority == kLowestPriority);
  void RegisterStream(StreamIdType stream_id, PriorityType priority) {
    auto stream_info = std::make_unique<StreamInfo>(
        StreamInfo{std::move(priority), stream_id, false});
    bool inserted =
        stream_infos_.insert(std::make_pair(stream_id, std::move(stream_info)))
            .second;
    QUICHE_BUG_IF(spdy_bug_19_2, !inserted)
        << "Stream " << stream_id << " already registered";
  }
  void UnregisterStream(StreamIdType stream_id) {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_3) << "Stream " << stream_id << " not registered";
      return;
    }
    const StreamInfo* const stream_info = it->second.get();
    if (stream_info->ready) {
      bool erased =
          Erase(&priority_infos_[PriorityTypeToInt()(stream_info->priority)]
                     .ready_list,
                stream_info);
      QUICHE_DCHECK(erased);
    }
    stream_infos_.erase(it);
  }
  bool StreamRegistered(StreamIdType stream_id) const {
    return stream_infos_.find(stream_id) != stream_infos_.end();
  }
  PriorityType GetStreamPriority(StreamIdType stream_id) const {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_DVLOG(1) << "Stream " << stream_id << " not registered";
      return IntToPriorityType()(kLowestPriority);
    }
    return it->second->priority;
  }
  void UpdateStreamPriority(StreamIdType stream_id, PriorityType priority) {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_DVLOG(1) << "Stream " << stream_id << " not registered";
      return;
    }
    StreamInfo* const stream_info = it->second.get();
    if (stream_info->priority == priority) {
      return;
    }
    if (PriorityTypeToInt()(stream_info->priority) !=
            PriorityTypeToInt()(priority) &&
        stream_info->ready) {
      bool erased =
          Erase(&priority_infos_[PriorityTypeToInt()(stream_info->priority)]
                     .ready_list,
                stream_info);
      QUICHE_DCHECK(erased);
      priority_infos_[PriorityTypeToInt()(priority)].ready_list.push_back(
          stream_info);
      ++num_ready_streams_;
    }
    stream_info->priority = std::move(priority);
  }
  void RecordStreamEventTime(StreamIdType stream_id, absl::Time now) {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_4) << "Stream " << stream_id << " not registered";
      return;
    }
    PriorityInfo& priority_info =
        priority_infos_[PriorityTypeToInt()(it->second->priority)];
    priority_info.last_event_time =
        std::max(priority_info.last_event_time, absl::make_optional(now));
  }
  std::optional<absl::Time> GetLatestEventWithPriority(
      StreamIdType stream_id) const {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_5) << "Stream " << stream_id << " not registered";
      return std::nullopt;
    }
    std::optional<absl::Time> last_event_time;
    const StreamInfo* const stream_info = it->second.get();
    for (int p = kHighestPriority;
         p < PriorityTypeToInt()(stream_info->priority); ++p) {
      last_event_time =
          std::max(last_event_time, priority_infos_[p].last_event_time);
    }
    return last_event_time;
  }
  StreamIdType PopNextReadyStream() {
    return std::get<0>(PopNextReadyStreamAndPriority());
  }
  std::tuple<StreamIdType, PriorityType> PopNextReadyStreamAndPriority() {
    for (int p = kHighestPriority; p <= kLowestPriority; ++p) {
      ReadyList& ready_list = priority_infos_[p].ready_list;
      if (!ready_list.empty()) {
        StreamInfo* const info = ready_list.front();
        ready_list.pop_front();
        --num_ready_streams_;
        QUICHE_DCHECK(stream_infos_.find(info->stream_id) !=
                      stream_infos_.end());
        info->ready = false;
        return std::make_tuple(info->stream_id, info->priority);
      }
    }
    QUICHE_BUG(spdy_bug_19_6) << "No ready streams available";
    return std::make_tuple(0, IntToPriorityType()(kLowestPriority));
  }
  bool ShouldYield(StreamIdType stream_id) const {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_7) << "Stream " << stream_id << " not registered";
      return false;
    }
    const StreamInfo* const stream_info = it->second.get();
    for (int p = kHighestPriority;
         p < PriorityTypeToInt()(stream_info->priority); ++p) {
      if (!priority_infos_[p].ready_list.empty()) {
        return true;
      }
    }
    const auto& ready_list =
        priority_infos_[PriorityTypeToInt()(it->second->priority)].ready_list;
    if (ready_list.empty() || ready_list.front()->stream_id == stream_id) {
      return false;
    }
    return true;
  }
  void MarkStreamReady(StreamIdType stream_id, bool add_to_front) {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_8) << "Stream " << stream_id << " not registered";
      return;
    }
    StreamInfo* const stream_info = it->second.get();
    if (stream_info->ready) {
      return;
    }
    ReadyList& ready_list =
        priority_infos_[PriorityTypeToInt()(stream_info->priority)].ready_list;
    if (add_to_front) {
      ready_list.push_front(stream_info);
    } else {
      ready_list.push_back(stream_info);
    }
    ++num_ready_streams_;
    stream_info->ready = true;
  }
  void MarkStreamNotReady(StreamIdType stream_id) {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_BUG(spdy_bug_19_9) << "Stream " << stream_id << " not registered";
      return;
    }
    StreamInfo* const stream_info = it->second.get();
    if (!stream_info->ready) {
      return;
    }
    bool erased = Erase(
        &priority_infos_[PriorityTypeToInt()(stream_info->priority)].ready_list,
        stream_info);
    QUICHE_DCHECK(erased);
    stream_info->ready = false;
  }
  bool HasReadyStreams() const { return num_ready_streams_ > 0; }
  size_t NumReadyStreams() const { return num_ready_streams_; }
  size_t NumRegisteredStreams() const { return stream_infos_.size(); }
  std::string DebugString() const {
    return absl::StrCat(
        "PriorityWriteScheduler {num_streams=", stream_infos_.size(),
        " num_ready_streams=", NumReadyStreams(), "}");
  }
  bool IsStreamReady(StreamIdType stream_id) const {
    auto it = stream_infos_.find(stream_id);
    if (it == stream_infos_.end()) {
      QUICHE_DLOG(INFO) << "Stream " << stream_id << " not registered";
      return false;
    }
    return it->second->ready;
  }
 private:
  friend class test::PriorityWriteSchedulerPeer<StreamIdType>;
  struct QUICHE_EXPORT StreamInfo {
    PriorityType priority;
    StreamIdType stream_id;
    bool ready;
  };
  using ReadyList = quiche::QuicheCircularDeque<StreamInfo*>;
  struct QUICHE_EXPORT PriorityInfo {
    ReadyList ready_list;
    std::optional<absl::Time> last_event_time;
  };
  using StreamInfoMap =
      absl::flat_hash_map<StreamIdType, std::unique_ptr<StreamInfo>>;
  bool Erase(ReadyList* ready_list, const StreamInfo* info) {
    auto it = std::remove(ready_list->begin(), ready_list->end(), info);
    if (it == ready_list->end()) {
      return false;
    }
    ready_list->pop_back();
    --num_ready_streams_;
    return true;
  }
  size_t num_ready_streams_ = 0;
  PriorityInfo priority_infos_[kLowestPriority + 1];
  StreamInfoMap stream_infos_;
};
}  
#endif  