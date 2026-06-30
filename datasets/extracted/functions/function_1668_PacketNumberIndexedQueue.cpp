#ifndef QUICHE_QUIC_CORE_PACKET_NUMBER_INDEXED_QUEUE_H_
#define QUICHE_QUIC_CORE_PACKET_NUMBER_INDEXED_QUEUE_H_
#include "quiche/quic/core/quic_constants.h"
#include "quiche/quic/core/quic_packet_number.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/common/quiche_circular_deque.h"
namespace quic {
template <typename T>
class QUICHE_NO_EXPORT PacketNumberIndexedQueue {
 public:
  PacketNumberIndexedQueue() : number_of_present_entries_(0) {}
  T* GetEntry(QuicPacketNumber packet_number);
  const T* GetEntry(QuicPacketNumber packet_number) const;
  template <typename... Args>
  bool Emplace(QuicPacketNumber packet_number, Args&&... args);
  bool Remove(QuicPacketNumber packet_number);
  template <typename Function>
  bool Remove(QuicPacketNumber packet_number, Function f);
  void RemoveUpTo(QuicPacketNumber packet_number);
  bool IsEmpty() const { return number_of_present_entries_ == 0; }
  size_t number_of_present_entries() const {
    return number_of_present_entries_;
  }
  size_t entry_slots_used() const { return entries_.size(); }
  QuicPacketNumber first_packet() const { return first_packet_; }
  QuicPacketNumber last_packet() const {
    if (IsEmpty()) {
      return QuicPacketNumber();
    }
    return first_packet_ + entries_.size() - 1;
  }
 private:
  struct QUICHE_NO_EXPORT EntryWrapper : T {
    bool present;
    EntryWrapper() : present(false) {}
    template <typename... Args>
    explicit EntryWrapper(Args&&... args)
        : T(std::forward<Args>(args)...), present(true) {}
  };
  void Cleanup();
  const EntryWrapper* GetEntryWrapper(QuicPacketNumber offset) const;
  EntryWrapper* GetEntryWrapper(QuicPacketNumber offset) {
    const auto* const_this = this;
    return const_cast<EntryWrapper*>(const_this->GetEntryWrapper(offset));
  }
  quiche::QuicheCircularDeque<EntryWrapper> entries_;
  size_t number_of_present_entries_;
  QuicPacketNumber first_packet_;
};
template <typename T>
T* PacketNumberIndexedQueue<T>::GetEntry(QuicPacketNumber packet_number) {
  EntryWrapper* entry = GetEntryWrapper(packet_number);
  if (entry == nullptr) {
    return nullptr;
  }
  return entry;
}
template <typename T>
const T* PacketNumberIndexedQueue<T>::GetEntry(
    QuicPacketNumber packet_number) const {
  const EntryWrapper* entry = GetEntryWrapper(packet_number);
  if (entry == nullptr) {
    return nullptr;
  }
  return entry;
}
template <typename T>
template <typename... Args>
bool PacketNumberIndexedQueue<T>::Emplace(QuicPacketNumber packet_number,
                                          Args&&... args) {
  if (!packet_number.IsInitialized()) {
    QUIC_BUG(quic_bug_10359_1)
        << "Try to insert an uninitialized packet number";
    return false;
  }
  if (IsEmpty()) {
    QUICHE_DCHECK(entries_.empty());
    QUICHE_DCHECK(!first_packet_.IsInitialized());
    entries_.emplace_back(std::forward<Args>(args)...);
    number_of_present_entries_ = 1;
    first_packet_ = packet_number;
    return true;
  }
  if (packet_number <= last_packet()) {
    return false;
  }
  size_t offset = packet_number - first_packet_;
  if (offset > entries_.size()) {
    entries_.resize(offset);
  }
  number_of_present_entries_++;
  entries_.emplace_back(std::forward<Args>(args)...);
  QUICHE_DCHECK_EQ(packet_number, last_packet());
  return true;
}
template <typename T>
bool PacketNumberIndexedQueue<T>::Remove(QuicPacketNumber packet_number) {
  return Remove(packet_number, [](const T&) {});
}
template <typename T>
template <typename Function>
bool PacketNumberIndexedQueue<T>::Remove(QuicPacketNumber packet_number,
                                         Function f) {
  EntryWrapper* entry = GetEntryWrapper(packet_number);
  if (entry == nullptr) {
    return false;
  }
  f(*static_cast<const T*>(entry));
  entry->present = false;
  number_of_present_entries_--;
  if (packet_number == first_packet()) {
    Cleanup();
  }
  return true;
}
template <typename T>
void PacketNumberIndexedQueue<T>::RemoveUpTo(QuicPacketNumber packet_number) {
  while (!entries_.empty() && first_packet_.IsInitialized() &&
         first_packet_ < packet_number) {
    if (entries_.front().present) {
      number_of_present_entries_--;
    }
    entries_.pop_front();
    first_packet_++;
  }
  Cleanup();
}
template <typename T>
void PacketNumberIndexedQueue<T>::Cleanup() {
  while (!entries_.empty() && !entries_.front().present) {
    entries_.pop_front();
    first_packet_++;
  }
  if (entries_.empty()) {
    first_packet_.Clear();
  }
}
template <typename T>
auto PacketNumberIndexedQueue<T>::GetEntryWrapper(
    QuicPacketNumber packet_number) const -> const EntryWrapper* {
  if (!packet_number.IsInitialized() || IsEmpty() ||
      packet_number < first_packet_) {
    return nullptr;
  }
  uint64_t offset = packet_number - first_packet_;
  if (offset >= entries_.size()) {
    return nullptr;
  }
  const EntryWrapper* entry = &entries_[offset];
  if (!entry->present) {
    return nullptr;
  }
  return entry;
}
}  
#endif  