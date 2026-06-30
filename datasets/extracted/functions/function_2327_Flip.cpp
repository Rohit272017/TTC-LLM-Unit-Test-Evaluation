#include "quiche/quic/moqt/moqt_priority.h"
#include <cstdint>
#include <limits>
#include "quiche/web_transport/web_transport.h"
namespace moqt {
namespace {
template <uint64_t NumBits>
constexpr uint64_t Flip(uint64_t number) {
  static_assert(NumBits <= 63);
  return (1ull << NumBits) - 1 - number;
}
template <uint64_t N>
constexpr uint64_t OnlyLowestNBits(uint64_t value) {
  static_assert(N <= 62);
  return value & ((1ull << (N + 1)) - 1);
}
}  
webtransport::SendOrder SendOrderForStream(MoqtPriority subscriber_priority,
                                           MoqtPriority publisher_priority,
                                           uint64_t group_id,
                                           MoqtDeliveryOrder delivery_order) {
  const int64_t track_bits = (Flip<8>(subscriber_priority) << 54) |
                             (Flip<8>(publisher_priority) << 46);
  group_id = OnlyLowestNBits<46>(group_id);
  if (delivery_order == MoqtDeliveryOrder::kAscending) {
    group_id = Flip<46>(group_id);
  }
  return track_bits | group_id;
}
webtransport::SendOrder SendOrderForStream(MoqtPriority subscriber_priority,
                                           MoqtPriority publisher_priority,
                                           uint64_t group_id,
                                           uint64_t subgroup_id,
                                           MoqtDeliveryOrder delivery_order) {
  const int64_t track_bits = (Flip<8>(subscriber_priority) << 54) |
                             (Flip<8>(publisher_priority) << 46);
  group_id = OnlyLowestNBits<26>(group_id);
  subgroup_id = OnlyLowestNBits<20>(subgroup_id);
  if (delivery_order == MoqtDeliveryOrder::kAscending) {
    group_id = Flip<26>(group_id);
  }
  subgroup_id = Flip<20>(subgroup_id);
  return track_bits | (group_id << 20) | subgroup_id;
}
webtransport::SendOrder UpdateSendOrderForSubscriberPriority(
    const webtransport::SendOrder send_order,
    MoqtPriority subscriber_priority) {
  webtransport::SendOrder new_send_order = OnlyLowestNBits<54>(send_order);
  const int64_t sub_bits = Flip<8>(subscriber_priority) << 54;
  new_send_order |= sub_bits;
  return new_send_order;
}
const webtransport::SendOrder kMoqtControlStreamSendOrder =
    std::numeric_limits<webtransport::SendOrder>::max();
}  