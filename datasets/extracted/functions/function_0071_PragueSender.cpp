#include "quiche/quic/core/congestion_control/prague_sender.h"
#include <algorithm>
#include "quiche/quic/core/congestion_control/rtt_stats.h"
#include "quiche/quic/core/congestion_control/tcp_cubic_sender_bytes.h"
#include "quiche/quic/core/quic_clock.h"
#include "quiche/quic/core/quic_connection_stats.h"
#include "quiche/quic/core/quic_time.h"
#include "quiche/quic/core/quic_types.h"
namespace quic {
PragueSender::PragueSender(const QuicClock* clock, const RttStats* rtt_stats,
                           QuicPacketCount initial_tcp_congestion_window,
                           QuicPacketCount max_congestion_window,
                           QuicConnectionStats* stats)
    : TcpCubicSenderBytes(clock, rtt_stats, false,
                          initial_tcp_congestion_window, max_congestion_window,
                          stats),
      connection_start_time_(clock->Now()),
      last_alpha_update_(connection_start_time_) {}
void PragueSender::OnCongestionEvent(bool rtt_updated,
                                     QuicByteCount prior_in_flight,
                                     QuicTime event_time,
                                     const AckedPacketVector& acked_packets,
                                     const LostPacketVector& lost_packets,
                                     QuicPacketCount num_ect,
                                     QuicPacketCount num_ce) {
  if (!ect1_enabled_) {
    TcpCubicSenderBytes::OnCongestionEvent(rtt_updated, prior_in_flight,
                                           event_time, acked_packets,
                                           lost_packets, num_ect, num_ce);
    return;
  }
  if (rtt_updated) {
    rtt_virt_ = std::max(rtt_stats()->smoothed_rtt(), kPragueRttVirtMin);
  }
  if (prague_alpha_.has_value()) {
    ect_count_ += num_ect;
    ce_count_ += num_ce;
    if (event_time - last_alpha_update_ > rtt_virt_) {
      float frac = static_cast<float>(ce_count_) /
                   static_cast<float>(ect_count_ + ce_count_);
      prague_alpha_ =
          (1 - kPragueEwmaGain) * *prague_alpha_ + kPragueEwmaGain * frac;
      last_alpha_update_ = event_time;
      ect_count_ = 0;
      ce_count_ = 0;
    }
  } else if (num_ce > 0) {
    last_alpha_update_ = event_time;
    prague_alpha_ = 1.0;
    ect_count_ = num_ect;
    ce_count_ = num_ce;
  }
  if (!lost_packets.empty() && last_congestion_response_time_.has_value() &&
      (event_time - *last_congestion_response_time_ < rtt_virt_)) {
    QuicByteCount previous_reduction = last_congestion_response_size_;
    last_congestion_response_time_.reset();
    set_congestion_window(GetCongestionWindow() + previous_reduction);
  }
  if (!reduce_rtt_dependence_) {
    reduce_rtt_dependence_ =
        !InSlowStart() && lost_packets.empty() &&
        (event_time - connection_start_time_) >
            kRoundsBeforeReducedRttDependence * rtt_stats()->smoothed_rtt();
  }
  float congestion_avoidance_deflator;
  if (reduce_rtt_dependence_) {
    congestion_avoidance_deflator =
        static_cast<float>(rtt_stats()->smoothed_rtt().ToMicroseconds()) /
        static_cast<float>(rtt_virt_.ToMicroseconds());
    congestion_avoidance_deflator *= congestion_avoidance_deflator;
  } else {
    congestion_avoidance_deflator = 1.0f;
  }
  QuicByteCount original_cwnd = GetCongestionWindow();
  if (num_ce == 0 || !lost_packets.empty()) {
    TcpCubicSenderBytes::OnCongestionEvent(rtt_updated, prior_in_flight,
                                           event_time, acked_packets,
                                           lost_packets, num_ect, num_ce);
    if (lost_packets.empty() && reduce_rtt_dependence_ &&
        original_cwnd < GetCongestionWindow()) {
      QuicByteCount cwnd_increase = GetCongestionWindow() - original_cwnd;
      set_congestion_window(original_cwnd +
                            cwnd_increase * congestion_avoidance_deflator);
    }
    return;
  }
  if (InSlowStart()) {
    ExitSlowstart();
  }
  QuicByteCount bytes_acked = 0;
  for (auto packet : acked_packets) {
    bytes_acked += packet.bytes_acked;
  }
  float ce_fraction =
      static_cast<float>(num_ce) / static_cast<float>(num_ect + num_ce);
  QuicByteCount bytes_ce = bytes_acked * ce_fraction;
  QuicPacketCount ce_packets_remaining = num_ce;
  bytes_acked -= bytes_ce;
  if (!last_congestion_response_time_.has_value() ||
      event_time - *last_congestion_response_time_ > rtt_virt_) {
    last_congestion_response_time_ = event_time;
    while (ce_packets_remaining > 0) {
      OnPacketLost(acked_packets.back().packet_number, bytes_ce,
                   prior_in_flight);
      bytes_ce = 0;
      ce_packets_remaining--;
    }
    QuicByteCount cwnd_reduction = original_cwnd - GetCongestionWindow();
    last_congestion_response_size_ = cwnd_reduction * *prague_alpha_;
    set_congestion_window(original_cwnd - last_congestion_response_size_);
    set_slowstart_threshold(GetCongestionWindow());
    ExitRecovery();
  }
  if (num_ect == 0) {
    return;
  }
  for (const AckedPacket& acked : acked_packets) {
    OnPacketAcked(
        acked.packet_number,
        acked.bytes_acked * (1 - ce_fraction) * congestion_avoidance_deflator,
        prior_in_flight, event_time);
  }
}
CongestionControlType PragueSender::GetCongestionControlType() const {
  return kPragueCubic;
}
bool PragueSender::EnableECT1() {
  ect1_enabled_ = true;
  return true;
}
}  