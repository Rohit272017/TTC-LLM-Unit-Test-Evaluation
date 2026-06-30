#ifndef QUICHE_COMMON_BTREE_SCHEDULER_H_
#define QUICHE_COMMON_BTREE_SCHEDULER_H_
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/container/btree_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "quiche/common/platform/api/quiche_bug_tracker.h"
#include "quiche/common/platform/api/quiche_export.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace quiche {
template <typename Id, typename Priority>
class QUICHE_NO_EXPORT BTreeScheduler {
 public:
  bool HasRegistered() const { return !streams_.empty(); }
  bool HasScheduled() const { return !schedule_.empty(); }
  size_t NumScheduled() const { return schedule_.size(); }
  size_t NumRegistered() const { return streams_.size(); }
  size_t NumScheduledInPriorityRange(std::optional<Priority> min,
                                     std::optional<Priority> max) const;
  absl::StatusOr<bool> ShouldYield(Id id) const;
  std::optional<Priority> GetPriorityFor(Id id) const {
    auto it = streams_.find(id);
    if (it == streams_.end()) {
      return std::nullopt;
    }
    return it->second.priority;
  }
  absl::StatusOr<Id> PopFront();
  absl::Status Register(Id stream_id, const Priority& priority);
  absl::Status Unregister(Id stream_id);
  absl::Status UpdatePriority(Id stream_id, const Priority& new_priority);
  absl::Status Schedule(Id stream_id);
  bool IsScheduled(Id stream_id) const;
 private:
  struct StreamEntry {
    ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Priority priority;
    std::optional<int> current_sequence_number = std::nullopt;
    bool scheduled() const { return current_sequence_number.has_value(); }
  };
  using FullStreamEntry = std::pair<const Id, StreamEntry>;
  struct ScheduleKey {
    ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Priority priority;
    int sequence_number;
    bool operator<(const ScheduleKey& other) const {
      return std::make_tuple(priority, sequence_number) >
             std::make_tuple(other.priority, other.sequence_number);
    }
    static ScheduleKey MinForPriority(Priority priority) {
      return ScheduleKey{priority, std::numeric_limits<int>::max()};
    }
    static ScheduleKey MaxForPriority(Priority priority) {
      return ScheduleKey{priority, std::numeric_limits<int>::min()};
    }
  };
  using FullScheduleEntry = std::pair<const ScheduleKey, FullStreamEntry*>;
  using ScheduleIterator =
      typename absl::btree_map<ScheduleKey, FullStreamEntry*>::const_iterator;
  static Id StreamId(const FullScheduleEntry& entry) {
    return entry.second->first;
  }
  absl::StatusOr<FullScheduleEntry> DescheduleStream(const StreamEntry& entry);
  absl::node_hash_map<Id, StreamEntry> streams_;
  absl::btree_map<ScheduleKey, FullStreamEntry*> schedule_;
  int current_write_sequence_number_ = 0;
};
template <typename Id, typename Priority>
size_t BTreeScheduler<Id, Priority>::NumScheduledInPriorityRange(
    std::optional<Priority> min, std::optional<Priority> max) const {
  if (min.has_value() && max.has_value()) {
    QUICHE_DCHECK(*min <= *max);
  }
  ScheduleIterator begin =
      max.has_value() ? schedule_.lower_bound(ScheduleKey::MinForPriority(*max))
                      : schedule_.begin();
  ScheduleIterator end =
      min.has_value() ? schedule_.upper_bound(ScheduleKey::MaxForPriority(*min))
                      : schedule_.end();
  return end - begin;
}
template <typename Id, typename Priority>
absl::Status BTreeScheduler<Id, Priority>::Register(Id stream_id,
                                                    const Priority& priority) {
  auto [it, success] = streams_.insert({stream_id, StreamEntry{priority}});
  if (!success) {
    return absl::AlreadyExistsError("ID already registered");
  }
  return absl::OkStatus();
}
template <typename Id, typename Priority>
auto BTreeScheduler<Id, Priority>::DescheduleStream(const StreamEntry& entry)
    -> absl::StatusOr<FullScheduleEntry> {
  QUICHE_DCHECK(entry.scheduled());
  auto it = schedule_.find(
      ScheduleKey{entry.priority, *entry.current_sequence_number});
  if (it == schedule_.end()) {
    return absl::InternalError(
        "Calling DescheduleStream() on an entry that is not in the schedule at "
        "the expected key.");
  }
  FullScheduleEntry result = *it;
  schedule_.erase(it);
  return result;
}
template <typename Id, typename Priority>
absl::Status BTreeScheduler<Id, Priority>::Unregister(Id stream_id) {
  auto it = streams_.find(stream_id);
  if (it == streams_.end()) {
    return absl::NotFoundError("Stream not registered");
  }
  const StreamEntry& stream = it->second;
  if (stream.scheduled()) {
    if (!DescheduleStream(stream).ok()) {
      QUICHE_BUG(BTreeSchedule_Unregister_NotInSchedule)
          << "UnregisterStream() called on a stream ID " << stream_id
          << ", which is marked ready, but is not in the schedule";
    }
  }
  streams_.erase(it);
  return absl::OkStatus();
}
template <typename Id, typename Priority>
absl::Status BTreeScheduler<Id, Priority>::UpdatePriority(
    Id stream_id, const Priority& new_priority) {
  auto it = streams_.find(stream_id);
  if (it == streams_.end()) {
    return absl::NotFoundError("ID not registered");
  }
  StreamEntry& stream = it->second;
  std::optional<int> sequence_number;
  if (stream.scheduled()) {
    absl::StatusOr<FullScheduleEntry> old_entry = DescheduleStream(stream);
    if (old_entry.ok()) {
      sequence_number = old_entry->first.sequence_number;
      QUICHE_DCHECK_EQ(old_entry->second, &*it);
    } else {
      QUICHE_BUG(BTreeScheduler_Update_Not_In_Schedule)
          << "UpdatePriority() called on a stream ID " << stream_id
          << ", which is marked ready, but is not in the schedule";
    }
  }
  stream.priority = new_priority;
  if (sequence_number.has_value()) {
    schedule_.insert({ScheduleKey{stream.priority, *sequence_number}, &*it});
  }
  return absl::OkStatus();
}
template <typename Id, typename Priority>
absl::StatusOr<bool> BTreeScheduler<Id, Priority>::ShouldYield(
    Id stream_id) const {
  const auto stream_it = streams_.find(stream_id);
  if (stream_it == streams_.end()) {
    return absl::NotFoundError("ID not registered");
  }
  const StreamEntry& stream = stream_it->second;
  if (schedule_.empty()) {
    return false;
  }
  const FullScheduleEntry& next = *schedule_.begin();
  if (StreamId(next) == stream_id) {
    return false;
  }
  return next.first.priority >= stream.priority;
}
template <typename Id, typename Priority>
absl::StatusOr<Id> BTreeScheduler<Id, Priority>::PopFront() {
  if (schedule_.empty()) {
    return absl::NotFoundError("No streams scheduled");
  }
  auto schedule_it = schedule_.begin();
  QUICHE_DCHECK(schedule_it->second->second.scheduled());
  schedule_it->second->second.current_sequence_number = std::nullopt;
  Id result = StreamId(*schedule_it);
  schedule_.erase(schedule_it);
  return result;
}
template <typename Id, typename Priority>
absl::Status BTreeScheduler<Id, Priority>::Schedule(Id stream_id) {
  const auto stream_it = streams_.find(stream_id);
  if (stream_it == streams_.end()) {
    return absl::NotFoundError("ID not registered");
  }
  if (stream_it->second.scheduled()) {
    return absl::OkStatus();
  }
  auto [schedule_it, success] =
      schedule_.insert({ScheduleKey{stream_it->second.priority,
                                    --current_write_sequence_number_},
                        &*stream_it});
  QUICHE_BUG_IF(WebTransportWriteBlockedList_AddStream_conflict, !success)
      << "Conflicting key in scheduler for stream " << stream_id;
  stream_it->second.current_sequence_number =
      schedule_it->first.sequence_number;
  return absl::OkStatus();
}
template <typename Id, typename Priority>
bool BTreeScheduler<Id, Priority>::IsScheduled(Id stream_id) const {
  const auto stream_it = streams_.find(stream_id);
  if (stream_it == streams_.end()) {
    return false;
  }
  return stream_it->second.scheduled();
}
}  
#endif  