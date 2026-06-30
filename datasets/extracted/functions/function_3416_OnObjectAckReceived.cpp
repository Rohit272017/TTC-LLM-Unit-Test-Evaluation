#include "quiche/quic/moqt/moqt_bitrate_adjuster.h"
#include <algorithm>
#include <cstdint>
#include "quiche/quic/core/quic_bandwidth.h"
#include "quiche/quic/core/quic_time.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/web_transport/web_transport.h"
namespace moqt {
namespace {
using ::quic::QuicBandwidth;
using ::quic::QuicTime;
using ::quic::QuicTimeDelta;
constexpr float kTargetBitrateMultiplier = 0.9f;
constexpr float kMinTimeBetweenAdjustmentsInRtts = 40;
constexpr QuicTimeDelta kMaxTimeBetweenAdjustments =
    QuicTimeDelta::FromSeconds(3);
}  
void MoqtBitrateAdjuster::OnObjectAckReceived(
    uint64_t , uint64_t ,
    QuicTimeDelta delta_from_deadline) {
  if (delta_from_deadline < QuicTimeDelta::Zero()) {
    AttemptAdjustingDown();
  }
}
void MoqtBitrateAdjuster::AttemptAdjustingDown() {
  webtransport::SessionStats stats = session_->GetSessionStats();
  QuicTimeDelta adjustment_delay =
      QuicTimeDelta(stats.smoothed_rtt * kMinTimeBetweenAdjustmentsInRtts);
  adjustment_delay = std::min(adjustment_delay, kMaxTimeBetweenAdjustments);
  QuicTime now = clock_->ApproximateNow();
  if (now - last_adjustment_time_ < adjustment_delay) {
    return;
  }
  QuicBandwidth target_bandwidth =
      kTargetBitrateMultiplier *
      QuicBandwidth::FromBitsPerSecond(stats.estimated_send_rate_bps);
  QuicBandwidth current_bandwidth = adjustable_->GetCurrentBitrate();
  if (current_bandwidth <= target_bandwidth) {
    return;
  }
  QUICHE_DLOG(INFO) << "Adjusting the bitrate from " << current_bandwidth
                    << " to " << target_bandwidth;
  bool success = adjustable_->AdjustBitrate(target_bandwidth);
  if (success) {
    last_adjustment_time_ = now;
  }
}
void MoqtBitrateAdjuster::OnObjectAckSupportKnown(bool supported) {
  QUICHE_DLOG_IF(WARNING, !supported)
      << "OBJECT_ACK not supported; bitrate adjustments will not work.";
}
}  