#include "quiche/quic/core/qpack/qpack_blocking_manager.h"
#include <limits>
#include <utility>
namespace quic {
QpackBlockingManager::QpackBlockingManager() : known_received_count_(0) {}
bool QpackBlockingManager::OnHeaderAcknowledgement(QuicStreamId stream_id) {
  auto it = header_blocks_.find(stream_id);
  if (it == header_blocks_.end()) {
    return false;
  }
  QUICHE_DCHECK(!it->second.empty());
  const IndexSet& indices = it->second.front();
  QUICHE_DCHECK(!indices.empty());
  const uint64_t required_index_count = RequiredInsertCount(indices);
  if (known_received_count_ < required_index_count) {
    known_received_count_ = required_index_count;
  }
  DecreaseReferenceCounts(indices);
  it->second.pop_front();
  if (it->second.empty()) {
    header_blocks_.erase(it);
  }
  return true;
}
void QpackBlockingManager::OnStreamCancellation(QuicStreamId stream_id) {
  auto it = header_blocks_.find(stream_id);
  if (it == header_blocks_.end()) {
    return;
  }
  for (const IndexSet& indices : it->second) {
    DecreaseReferenceCounts(indices);
  }
  header_blocks_.erase(it);
}
bool QpackBlockingManager::OnInsertCountIncrement(uint64_t increment) {
  if (increment >
      std::numeric_limits<uint64_t>::max() - known_received_count_) {
    return false;
  }
  known_received_count_ += increment;
  return true;
}
void QpackBlockingManager::OnHeaderBlockSent(QuicStreamId stream_id,
                                             IndexSet indices) {
  QUICHE_DCHECK(!indices.empty());
  IncreaseReferenceCounts(indices);
  header_blocks_[stream_id].push_back(std::move(indices));
}
bool QpackBlockingManager::blocking_allowed_on_stream(
    QuicStreamId stream_id, uint64_t maximum_blocked_streams) const {
  if (header_blocks_.size() + 1 <= maximum_blocked_streams) {
    return true;
  }
  if (maximum_blocked_streams == 0) {
    return false;
  }
  uint64_t blocked_stream_count = 0;
  for (const auto& header_blocks_for_stream : header_blocks_) {
    for (const IndexSet& indices : header_blocks_for_stream.second) {
      if (RequiredInsertCount(indices) > known_received_count_) {
        if (header_blocks_for_stream.first == stream_id) {
          return true;
        }
        ++blocked_stream_count;
        if (blocked_stream_count + 1 > maximum_blocked_streams) {
          return false;
        }
        break;
      }
    }
  }
  return true;
}
uint64_t QpackBlockingManager::smallest_blocking_index() const {
  return entry_reference_counts_.empty()
             ? std::numeric_limits<uint64_t>::max()
             : entry_reference_counts_.begin()->first;
}
uint64_t QpackBlockingManager::RequiredInsertCount(const IndexSet& indices) {
  return *indices.rbegin() + 1;
}
void QpackBlockingManager::IncreaseReferenceCounts(const IndexSet& indices) {
  for (const uint64_t index : indices) {
    auto it = entry_reference_counts_.lower_bound(index);
    if (it != entry_reference_counts_.end() && it->first == index) {
      ++it->second;
    } else {
      entry_reference_counts_.insert(it, {index, 1});
    }
  }
}
void QpackBlockingManager::DecreaseReferenceCounts(const IndexSet& indices) {
  for (const uint64_t index : indices) {
    auto it = entry_reference_counts_.find(index);
    QUICHE_DCHECK(it != entry_reference_counts_.end());
    QUICHE_DCHECK_NE(0u, it->second);
    if (it->second == 1) {
      entry_reference_counts_.erase(it);
    } else {
      --it->second;
    }
  }
}
}  