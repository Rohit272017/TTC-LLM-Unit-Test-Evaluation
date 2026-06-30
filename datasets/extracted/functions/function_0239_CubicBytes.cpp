#include "quiche/quic/core/congestion_control/cubic_bytes.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "quiche/quic/core/quic_constants.h"
#include "quiche/quic/core/quic_packets.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
namespace {
const int kCubeScale = 40;  
const int kCubeCongestionWindowScale = 410;
const uint64_t kCubeFactor =
    (UINT64_C(1) << kCubeScale) / kCubeCongestionWindowScale / kDefaultTCPMSS;
const float kDefaultCubicBackoffFactor = 0.7f;  
const float kBetaLastMax = 0.85f;
}  
CubicBytes::CubicBytes(const QuicClock* clock)
    : clock_(clock),
      num_connections_(kDefaultNumConnections),
      epoch_(QuicTime::Zero()) {
  ResetCubicState();
}
void CubicBytes::SetNumConnections(int num_connections) {
  num_connections_ = num_connections;
}
float CubicBytes::Alpha() const {
  const float beta = Beta();
  return 3 * num_connections_ * num_connections_ * (1 - beta) / (1 + beta);
}
float CubicBytes::Beta() const {
  return (num_connections_ - 1 + kDefaultCubicBackoffFactor) / num_connections_;
}
float CubicBytes::BetaLastMax() const {
  return (num_connections_ - 1 + kBetaLastMax) / num_connections_;
}
void CubicBytes::ResetCubicState() {
  epoch_ = QuicTime::Zero();  
  last_max_congestion_window_ = 0;
  acked_bytes_count_ = 0;
  estimated_tcp_congestion_window_ = 0;
  origin_point_congestion_window_ = 0;
  time_to_origin_point_ = 0;
  last_target_congestion_window_ = 0;
}
void CubicBytes::OnApplicationLimited() {
  epoch_ = QuicTime::Zero();
}
QuicByteCount CubicBytes::CongestionWindowAfterPacketLoss(
    QuicByteCount current_congestion_window) {
  if (current_congestion_window + kDefaultTCPMSS <
      last_max_congestion_window_) {
    last_max_congestion_window_ =
        static_cast<int>(BetaLastMax() * current_congestion_window);
  } else {
    last_max_congestion_window_ = current_congestion_window;
  }
  epoch_ = QuicTime::Zero();  
  return static_cast<int>(current_congestion_window * Beta());
}
QuicByteCount CubicBytes::CongestionWindowAfterAck(
    QuicByteCount acked_bytes, QuicByteCount current_congestion_window,
    QuicTime::Delta delay_min, QuicTime event_time) {
  acked_bytes_count_ += acked_bytes;
  if (!epoch_.IsInitialized()) {
    QUIC_DVLOG(1) << "Start of epoch";
    epoch_ = event_time;               
    acked_bytes_count_ = acked_bytes;  
    estimated_tcp_congestion_window_ = current_congestion_window;
    if (last_max_congestion_window_ <= current_congestion_window) {
      time_to_origin_point_ = 0;
      origin_point_congestion_window_ = current_congestion_window;
    } else {
      time_to_origin_point_ = static_cast<uint32_t>(
          cbrt(kCubeFactor *
               (last_max_congestion_window_ - current_congestion_window)));
      origin_point_congestion_window_ = last_max_congestion_window_;
    }
  }
  int64_t elapsed_time =
      ((event_time + delay_min - epoch_).ToMicroseconds() << 10) /
      kNumMicrosPerSecond;
  uint64_t offset = std::abs(time_to_origin_point_ - elapsed_time);
  QuicByteCount delta_congestion_window = (kCubeCongestionWindowScale * offset *
                                           offset * offset * kDefaultTCPMSS) >>
                                          kCubeScale;
  const bool add_delta = elapsed_time > time_to_origin_point_;
  QUICHE_DCHECK(add_delta ||
                (origin_point_congestion_window_ > delta_congestion_window));
  QuicByteCount target_congestion_window =
      add_delta ? origin_point_congestion_window_ + delta_congestion_window
                : origin_point_congestion_window_ - delta_congestion_window;
  target_congestion_window =
      std::min(target_congestion_window,
               current_congestion_window + acked_bytes_count_ / 2);
  QUICHE_DCHECK_LT(0u, estimated_tcp_congestion_window_);
  estimated_tcp_congestion_window_ += acked_bytes_count_ *
                                      (Alpha() * kDefaultTCPMSS) /
                                      estimated_tcp_congestion_window_;
  acked_bytes_count_ = 0;
  last_target_congestion_window_ = target_congestion_window;
  if (target_congestion_window < estimated_tcp_congestion_window_) {
    target_congestion_window = estimated_tcp_congestion_window_;
  }
  QUIC_DVLOG(1) << "Final target congestion_window: "
                << target_congestion_window;
  return target_congestion_window;
}
}  