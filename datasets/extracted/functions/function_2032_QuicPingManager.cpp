#include "quiche/quic/core/quic_ping_manager.h"
#include <algorithm>
#include "quiche/quic/platform/api/quic_flags.h"
namespace quic {
namespace {
const int kMaxRetransmittableOnWireDelayShift = 10;
}  
QuicPingManager::QuicPingManager(Perspective perspective, Delegate* delegate,
                                 QuicAlarm* alarm)
    : perspective_(perspective), delegate_(delegate), alarm_(*alarm) {}
void QuicPingManager::SetAlarm(QuicTime now, bool should_keep_alive,
                               bool has_in_flight_packets) {
  UpdateDeadlines(now, should_keep_alive, has_in_flight_packets);
  const QuicTime earliest_deadline = GetEarliestDeadline();
  if (!earliest_deadline.IsInitialized()) {
    alarm_.Cancel();
    return;
  }
  if (earliest_deadline == keep_alive_deadline_) {
    alarm_.Update(earliest_deadline, QuicTime::Delta::FromSeconds(1));
    return;
  }
  alarm_.Update(earliest_deadline, kAlarmGranularity);
}
void QuicPingManager::OnAlarm() {
  const QuicTime earliest_deadline = GetEarliestDeadline();
  if (!earliest_deadline.IsInitialized()) {
    QUIC_BUG(quic_ping_manager_alarm_fires_unexpectedly)
        << "QuicPingManager alarm fires unexpectedly.";
    return;
  }
  if (earliest_deadline == retransmittable_on_wire_deadline_) {
    retransmittable_on_wire_deadline_ = QuicTime::Zero();
    if (GetQuicFlag(quic_max_aggressive_retransmittable_on_wire_ping_count) !=
        0) {
      ++consecutive_retransmittable_on_wire_count_;
    }
    ++retransmittable_on_wire_count_;
    delegate_->OnRetransmittableOnWireTimeout();
    return;
  }
  if (earliest_deadline == keep_alive_deadline_) {
    keep_alive_deadline_ = QuicTime::Zero();
    delegate_->OnKeepAliveTimeout();
  }
}
void QuicPingManager::Stop() {
  alarm_.PermanentCancel();
  retransmittable_on_wire_deadline_ = QuicTime::Zero();
  keep_alive_deadline_ = QuicTime::Zero();
}
void QuicPingManager::UpdateDeadlines(QuicTime now, bool should_keep_alive,
                                      bool has_in_flight_packets) {
  keep_alive_deadline_ = QuicTime::Zero();
  if (perspective_ == Perspective::IS_SERVER &&
      initial_retransmittable_on_wire_timeout_.IsInfinite()) {
    QUICHE_DCHECK(!retransmittable_on_wire_deadline_.IsInitialized());
    return;
  }
  if (!should_keep_alive) {
    retransmittable_on_wire_deadline_ = QuicTime::Zero();
    return;
  }
  if (perspective_ == Perspective::IS_CLIENT) {
    keep_alive_deadline_ = now + keep_alive_timeout_;
  }
  if (initial_retransmittable_on_wire_timeout_.IsInfinite() ||
      has_in_flight_packets ||
      retransmittable_on_wire_count_ >
          GetQuicFlag(quic_max_retransmittable_on_wire_ping_count)) {
    retransmittable_on_wire_deadline_ = QuicTime::Zero();
    return;
  }
  QUICHE_DCHECK_LT(initial_retransmittable_on_wire_timeout_,
                   keep_alive_timeout_);
  QuicTime::Delta retransmittable_on_wire_timeout =
      initial_retransmittable_on_wire_timeout_;
  const int max_aggressive_retransmittable_on_wire_count =
      GetQuicFlag(quic_max_aggressive_retransmittable_on_wire_ping_count);
  QUICHE_DCHECK_LE(0, max_aggressive_retransmittable_on_wire_count);
  if (consecutive_retransmittable_on_wire_count_ >
      max_aggressive_retransmittable_on_wire_count) {
    int shift = std::min(consecutive_retransmittable_on_wire_count_ -
                             max_aggressive_retransmittable_on_wire_count,
                         kMaxRetransmittableOnWireDelayShift);
    retransmittable_on_wire_timeout =
        initial_retransmittable_on_wire_timeout_ * (1 << shift);
  }
  if (retransmittable_on_wire_deadline_.IsInitialized() &&
      retransmittable_on_wire_deadline_ <
          now + retransmittable_on_wire_timeout) {
    return;
  }
  retransmittable_on_wire_deadline_ = now + retransmittable_on_wire_timeout;
}
QuicTime QuicPingManager::GetEarliestDeadline() const {
  QuicTime earliest_deadline = QuicTime::Zero();
  for (QuicTime t : {retransmittable_on_wire_deadline_, keep_alive_deadline_}) {
    if (!t.IsInitialized()) {
      continue;
    }
    if (!earliest_deadline.IsInitialized() || t < earliest_deadline) {
      earliest_deadline = t;
    }
  }
  return earliest_deadline;
}
}  