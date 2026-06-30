#include "quiche/quic/core/quic_blocked_writer_list.h"
#include <utility>
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
namespace quic {
void QuicBlockedWriterList::Add(QuicBlockedWriterInterface& blocked_writer) {
  if (!blocked_writer.IsWriterBlocked()) {
    QUIC_BUG(quic_bug_12724_4)
        << "Tried to add writer into blocked list when it shouldn't be added";
    return;
  }
  write_blocked_list_.insert(std::make_pair(&blocked_writer, true));
}
bool QuicBlockedWriterList::Empty() const {
  return write_blocked_list_.empty();
}
bool QuicBlockedWriterList::Remove(QuicBlockedWriterInterface& blocked_writer) {
  return write_blocked_list_.erase(&blocked_writer) != 0;
}
void QuicBlockedWriterList::OnWriterUnblocked() {
  const size_t num_blocked_writers_before = write_blocked_list_.size();
  WriteBlockedList temp_list;
  temp_list.swap(write_blocked_list_);
  QUICHE_DCHECK(write_blocked_list_.empty());
  while (!temp_list.empty()) {
    QuicBlockedWriterInterface* blocked_writer = temp_list.begin()->first;
    temp_list.erase(temp_list.begin());
    blocked_writer->OnBlockedWriterCanWrite();
  }
  const size_t num_blocked_writers_after = write_blocked_list_.size();
  if (num_blocked_writers_after != 0) {
    if (num_blocked_writers_before == num_blocked_writers_after) {
      QUIC_CODE_COUNT(quic_zero_progress_on_can_write);
    } else {
      QUIC_CODE_COUNT(quic_blocked_again_on_can_write);
    }
  }
}
}  