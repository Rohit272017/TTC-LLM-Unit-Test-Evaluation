#include "quiche/quic/core/congestion_control/bbr_sender.h"
#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include "absl/base/attributes.h"
#include "quiche/quic/core/congestion_control/rtt_stats.h"
#include "quiche/quic/core/crypto/crypto_protocol.h"
#include "quiche/quic/core/quic_time.h"
#include "quiche/quic/core/quic_time_accumulator.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
namespace {
const QuicByteCount kDefaultMinimumCongestionWindow = 4 * kMaxSegmentSize;
const float kDefaultHighGain = 2.885f;
const float kDerivedHighGain = 2.773f;
const float kDerivedHighCWNDGain = 2.0f;
const float kPacingGain[] = {1.25, 0.75, 1, 1, 1, 1, 1, 1};
const size_t kGainCycleLength = sizeof(kPacingGain) / sizeof(kPacingGain[0]);
const QuicRoundTripCount kBandwidthWindowSize = kGainCycleLength + 2;
const QuicTime::Delta kMinRttExpiry = QuicTime::Delta::FromSeconds(10);
const QuicTime::Delta kProbeRttTime = QuicTime::Delta::FromMilliseconds(200);
const float kStartupGrowthTarget = 1.25;
const QuicRoundTripCount kRoundTripsWithoutGrowthBeforeExitingStartup = 3;
}  
BbrSender::DebugState::DebugState(const BbrSender& sender)
    : mode(sender.mode_),
      max_bandwidth(sender.max_bandwidth_.GetBest()),
      round_trip_count(sender.round_trip_count_),
      gain_cycle_index(sender.cycle_current_offset_),
      congestion_window(sender.congestion_window_),
      is_at_full_bandwidth(sender.is_at_full_bandwidth_),
      bandwidth_at_last_round(sender.bandwidth_at_last_round_),
      rounds_without_bandwidth_gain(sender.rounds_without_bandwidth_gain_),
      min_rtt(sender.min_rtt_),
      min_rtt_timestamp(sender.min_rtt_timestamp_),
      recovery_state(sender.recovery_state_),
      recovery_window(sender.recovery_window_),
      last_sample_is_app_limited(sender.last_sample_is_app_limited_),
      end_of_app_limited_phase(sender.sampler_.end_of_app_limited_phase()) {}
BbrSender::DebugState::DebugState(const DebugState& state) = default;
BbrSender::BbrSender(QuicTime now, const RttStats* rtt_stats,
                     const QuicUnackedPacketMap* unacked_packets,
                     QuicPacketCount initial_tcp_congestion_window,
                     QuicPacketCount max_tcp_congestion_window,
                     QuicRandom* random, QuicConnectionStats* stats)
    : rtt_stats_(rtt_stats),
      unacked_packets_(unacked_packets),
      random_(random),
      stats_(stats),
      mode_(STARTUP),
      sampler_(unacked_packets, kBandwidthWindowSize),
      round_trip_count_(0),
      num_loss_events_in_round_(0),
      bytes_lost_in_round_(0),
      max_bandwidth_(kBandwidthWindowSize, QuicBandwidth::Zero(), 0),
      min_rtt_(QuicTime::Delta::Zero()),
      min_rtt_timestamp_(QuicTime::Zero()),
      congestion_window_(initial_tcp_congestion_window * kDefaultTCPMSS),
      initial_congestion_window_(initial_tcp_congestion_window *
                                 kDefaultTCPMSS),
      max_congestion_window_(max_tcp_congestion_window * kDefaultTCPMSS),
      min_congestion_window_(kDefaultMinimumCongestionWindow),
      high_gain_(kDefaultHighGain),
      high_cwnd_gain_(kDefaultHighGain),
      drain_gain_(1.f / kDefaultHighGain),
      pacing_rate_(QuicBandwidth::Zero()),
      pacing_gain_(1),
      congestion_window_gain_(1),
      congestion_window_gain_constant_(
          static_cast<float>(GetQuicFlag(quic_bbr_cwnd_gain))),
      num_startup_rtts_(kRoundTripsWithoutGrowthBeforeExitingStartup),
      cycle_current_offset_(0),
      last_cycle_start_(QuicTime::Zero()),
      is_at_full_bandwidth_(false),
      rounds_without_bandwidth_gain_(0),
      bandwidth_at_last_round_(QuicBandwidth::Zero()),
      exiting_quiescence_(false),
      exit_probe_rtt_at_(QuicTime::Zero()),
      probe_rtt_round_passed_(false),
      last_sample_is_app_limited_(false),
      has_non_app_limited_sample_(false),
      recovery_state_(NOT_IN_RECOVERY),
      recovery_window_(max_congestion_window_),
      slower_startup_(false),
      rate_based_startup_(false),
      enable_ack_aggregation_during_startup_(false),
      expire_ack_aggregation_in_startup_(false),
      drain_to_target_(false),
      detect_overshooting_(false),
      bytes_lost_while_detecting_overshooting_(0),
      bytes_lost_multiplier_while_detecting_overshooting_(2),
      cwnd_to_calculate_min_pacing_rate_(initial_congestion_window_),
      max_congestion_window_with_network_parameters_adjusted_(
          kMaxInitialCongestionWindow * kDefaultTCPMSS) {
  if (stats_) {
    stats_->slowstart_count = 0;
    stats_->slowstart_duration = QuicTimeAccumulator();
  }
  EnterStartupMode(now);
  set_high_cwnd_gain(kDerivedHighCWNDGain);
}
BbrSender::~BbrSender() {}
void BbrSender::SetInitialCongestionWindowInPackets(
    QuicPacketCount congestion_window) {
  if (mode_ == STARTUP) {
    initial_congestion_window_ = congestion_window * kDefaultTCPMSS;
    congestion_window_ = congestion_window * kDefaultTCPMSS;
    cwnd_to_calculate_min_pacing_rate_ = std::min(
        initial_congestion_window_, cwnd_to_calculate_min_pacing_rate_);
  }
}
bool BbrSender::InSlowStart() const { return mode_ == STARTUP; }
void BbrSender::OnPacketSent(QuicTime sent_time, QuicByteCount bytes_in_flight,
                             QuicPacketNumber packet_number,
                             QuicByteCount bytes,
                             HasRetransmittableData is_retransmittable) {
  if (stats_ && InSlowStart()) {
    ++stats_->slowstart_packets_sent;
    stats_->slowstart_bytes_sent += bytes;
  }
  last_sent_packet_ = packet_number;
  if (bytes_in_flight == 0 && sampler_.is_app_limited()) {
    exiting_quiescence_ = true;
  }
  sampler_.OnPacketSent(sent_time, packet_number, bytes, bytes_in_flight,
                        is_retransmittable);
}
void BbrSender::OnPacketNeutered(QuicPacketNumber packet_number) {
  sampler_.OnPacketNeutered(packet_number);
}
bool BbrSender::CanSend(QuicByteCount bytes_in_flight) {
  return bytes_in_flight < GetCongestionWindow();
}
QuicBandwidth BbrSender::PacingRate(QuicByteCount ) const {
  if (pacing_rate_.IsZero()) {
    return high_gain_ * QuicBandwidth::FromBytesAndTimeDelta(
                            initial_congestion_window_, GetMinRtt());
  }
  return pacing_rate_;
}
QuicBandwidth BbrSender::BandwidthEstimate() const {
  return max_bandwidth_.GetBest();
}
QuicByteCount BbrSender::GetCongestionWindow() const {
  if (mode_ == PROBE_RTT) {
    return ProbeRttCongestionWindow();
  }
  if (InRecovery()) {
    return std::min(congestion_window_, recovery_window_);
  }
  return congestion_window_;
}
QuicByteCount BbrSender::GetSlowStartThreshold() const { return 0; }
bool BbrSender::InRecovery() const {
  return recovery_state_ != NOT_IN_RECOVERY;
}
void BbrSender::SetFromConfig(const QuicConfig& config,
                              Perspective perspective) {
  if (config.HasClientRequestedIndependentOption(k1RTT, perspective)) {
    num_startup_rtts_ = 1;
  }
  if (config.HasClientRequestedIndependentOption(k2RTT, perspective)) {
    num_startup_rtts_ = 2;
  }
  if (config.HasClientRequestedIndependentOption(kBBR3, perspective)) {
    drain_to_target_ = true;
  }
  if (config.HasClientRequestedIndependentOption(kBWM3, perspective)) {
    bytes_lost_multiplier_while_detecting_overshooting_ = 3;
  }
  if (config.HasClientRequestedIndependentOption(kBWM4, perspective)) {
    bytes_lost_multiplier_while_detecting_overshooting_ = 4;
  }
  if (config.HasClientRequestedIndependentOption(kBBR4, perspective)) {
    sampler_.SetMaxAckHeightTrackerWindowLength(2 * kBandwidthWindowSize);
  }
  if (config.HasClientRequestedIndependentOption(kBBR5, perspective)) {
    sampler_.SetMaxAckHeightTrackerWindowLength(4 * kBandwidthWindowSize);
  }
  if (config.HasClientRequestedIndependentOption(kBBQ1, perspective)) {
    set_high_gain(kDerivedHighGain);
    set_high_cwnd_gain(kDerivedHighGain);
    set_drain_gain(1.0 / kDerivedHighCWNDGain);
  }
  if (config.HasClientRequestedIndependentOption(kBBQ3, perspective)) {
    enable_ack_aggregation_during_startup_ = true;
  }
  if (config.HasClientRequestedIndependentOption(kBBQ5, perspective)) {
    expire_ack_aggregation_in_startup_ = true;
  }
  if (config.HasClientRequestedIndependentOption(kMIN1, perspective)) {
    min_congestion_window_ = kMaxSegmentSize;
  }
  if (config.HasClientRequestedIndependentOption(kICW1, perspective)) {
    max_congestion_window_with_network_parameters_adjusted_ =
        100 * kDefaultTCPMSS;
  }
  if (config.HasClientRequestedIndependentOption(kDTOS, perspective)) {
    detect_overshooting_ = true;
    cwnd_to_calculate_min_pacing_rate_ =
        std::min(initial_congestion_window_, 10 * kDefaultTCPMSS);
  }
  ApplyConnectionOptions(config.ClientRequestedIndependentOptions(perspective));
}
void BbrSender::ApplyConnectionOptions(
    const QuicTagVector& connection_options) {
  if (ContainsQuicTag(connection_options, kBSAO)) {
    sampler_.EnableOverestimateAvoidance();
  }
  if (ContainsQuicTag(connection_options, kBBRA)) {
    sampler_.SetStartNewAggregationEpochAfterFullRound(true);
  }
  if (ContainsQuicTag(connection_options, kBBRB)) {
    sampler_.SetLimitMaxAckHeightTrackerBySendRate(true);
  }
}
void BbrSender::AdjustNetworkParameters(const NetworkParams& params) {
  const QuicBandwidth& bandwidth = params.bandwidth;
  const QuicTime::Delta& rtt = params.rtt;
  if (!rtt.IsZero() && (min_rtt_ > rtt || min_rtt_.IsZero())) {
    min_rtt_ = rtt;
  }
  if (mode_ == STARTUP) {
    if (bandwidth.IsZero()) {
      return;
    }
    auto cwnd_bootstrapping_rtt = GetMinRtt();
    if (params.max_initial_congestion_window > 0) {
      max_congestion_window_with_network_parameters_adjusted_ =
          params.max_initial_congestion_window * kDefaultTCPMSS;
    }
    const QuicByteCount new_cwnd = std::max(
        kMinInitialCongestionWindow * kDefaultTCPMSS,
        std::min(max_congestion_window_with_network_parameters_adjusted_,
                 bandwidth * cwnd_bootstrapping_rtt));
    stats_->cwnd_bootstrapping_rtt_us = cwnd_bootstrapping_rtt.ToMicroseconds();
    if (!rtt_stats_->smoothed_rtt().IsZero()) {
      QUIC_CODE_COUNT(quic_smoothed_rtt_available);
    } else if (rtt_stats_->initial_rtt() !=
               QuicTime::Delta::FromMilliseconds(kInitialRttMs)) {
      QUIC_CODE_COUNT(quic_client_initial_rtt_available);
    } else {
      QUIC_CODE_COUNT(quic_default_initial_rtt);
    }
    if (new_cwnd < congestion_window_ && !params.allow_cwnd_to_decrease) {
      return;
    }
    if (GetQuicReloadableFlag(quic_conservative_cwnd_and_pacing_gains)) {
      QUIC_RELOADABLE_FLAG_COUNT(quic_conservative_cwnd_and_pacing_gains);
      set_high_gain(kDerivedHighCWNDGain);
      set_high_cwnd_gain(kDerivedHighCWNDGain);
    }
    congestion_window_ = new_cwnd;
    QuicBandwidth new_pacing_rate =
        QuicBandwidth::FromBytesAndTimeDelta(congestion_window_, GetMinRtt());
    pacing_rate_ = std::max(pacing_rate_, new_pacing_rate);
    detect_overshooting_ = true;
  }
}
void BbrSender::OnCongestionEvent(bool ,
                                  QuicByteCount prior_in_flight,
                                  QuicTime event_time,
                                  const AckedPacketVector& acked_packets,
                                  const LostPacketVector& lost_packets,
                                  QuicPacketCount ,
                                  QuicPacketCount ) {
  const QuicByteCount total_bytes_acked_before = sampler_.total_bytes_acked();
  const QuicByteCount total_bytes_lost_before = sampler_.total_bytes_lost();
  bool is_round_start = false;
  bool min_rtt_expired = false;
  QuicByteCount excess_acked = 0;
  QuicByteCount bytes_lost = 0;
  SendTimeState last_packet_send_state;
  if (!acked_packets.empty()) {
    QuicPacketNumber last_acked_packet = acked_packets.rbegin()->packet_number;
    is_round_start = UpdateRoundTripCounter(last_acked_packet);
    UpdateRecoveryState(last_acked_packet, !lost_packets.empty(),
                        is_round_start);
  }
  BandwidthSamplerInterface::CongestionEventSample sample =
      sampler_.OnCongestionEvent(event_time, acked_packets, lost_packets,
                                 max_bandwidth_.GetBest(),
                                 QuicBandwidth::Infinite(), round_trip_count_);
  if (sample.last_packet_send_state.is_valid) {
    last_sample_is_app_limited_ = sample.last_packet_send_state.is_app_limited;
    has_non_app_limited_sample_ |= !last_sample_is_app_limited_;
    if (stats_) {
      stats_->has_non_app_limited_sample = has_non_app_limited_sample_;
    }
  }
  if (total_bytes_acked_before != sampler_.total_bytes_acked()) {
    QUIC_LOG_IF(WARNING, sample.sample_max_bandwidth.IsZero())
        << sampler_.total_bytes_acked() - total_bytes_acked_before
        << " bytes from " << acked_packets.size()
        << " packets have been acked, but sample_max_bandwidth is zero.";
    if (!sample.sample_is_app_limited ||
        sample.sample_max_bandwidth > max_bandwidth_.GetBest()) {
      max_bandwidth_.Update(sample.sample_max_bandwidth, round_trip_count_);
    }
  }
  if (!sample.sample_rtt.IsInfinite()) {
    min_rtt_expired = MaybeUpdateMinRtt(event_time, sample.sample_rtt);
  }
  bytes_lost = sampler_.total_bytes_lost() - total_bytes_lost_before;
  if (mode_ == STARTUP) {
    if (stats_) {
      stats_->slowstart_packets_lost += lost_packets.size();
      stats_->slowstart_bytes_lost += bytes_lost;
    }
  }
  excess_acked = sample.extra_acked;
  last_packet_send_state = sample.last_packet_send_state;
  if (!lost_packets.empty()) {
    ++num_loss_events_in_round_;
    bytes_lost_in_round_ += bytes_lost;
  }
  if (mode_ == PROBE_BW) {
    UpdateGainCyclePhase(event_time, prior_in_flight, !lost_packets.empty());
  }
  if (is_round_start && !is_at_full_bandwidth_) {
    CheckIfFullBandwidthReached(last_packet_send_state);
  }
  MaybeExitStartupOrDrain(event_time);
  MaybeEnterOrExitProbeRtt(event_time, is_round_start, min_rtt_expired);
  QuicByteCount bytes_acked =
      sampler_.total_bytes_acked() - total_bytes_acked_before;
  CalculatePacingRate(bytes_lost);
  CalculateCongestionWindow(bytes_acked, excess_acked);
  CalculateRecoveryWindow(bytes_acked, bytes_lost);
  sampler_.RemoveObsoletePackets(unacked_packets_->GetLeastUnacked());
  if (is_round_start) {
    num_loss_events_in_round_ = 0;
    bytes_lost_in_round_ = 0;
  }
}
CongestionControlType BbrSender::GetCongestionControlType() const {
  return kBBR;
}
QuicTime::Delta BbrSender::GetMinRtt() const {
  if (!min_rtt_.IsZero()) {
    return min_rtt_;
  }
  return rtt_stats_->MinOrInitialRtt();
}
QuicByteCount BbrSender::GetTargetCongestionWindow(float gain) const {
  QuicByteCount bdp = GetMinRtt() * BandwidthEstimate();
  QuicByteCount congestion_window = gain * bdp;
  if (congestion_window == 0) {
    congestion_window = gain * initial_congestion_window_;
  }
  return std::max(congestion_window, min_congestion_window_);
}
QuicByteCount BbrSender::ProbeRttCongestionWindow() const {
  return min_congestion_window_;
}
void BbrSender::EnterStartupMode(QuicTime now) {
  if (stats_) {
    ++stats_->slowstart_count;
    stats_->slowstart_duration.Start(now);
  }
  mode_ = STARTUP;
  pacing_gain_ = high_gain_;
  congestion_window_gain_ = high_cwnd_gain_;
}
void BbrSender::EnterProbeBandwidthMode(QuicTime now) {
  mode_ = PROBE_BW;
  congestion_window_gain_ = congestion_window_gain_constant_;
  cycle_current_offset_ = random_->RandUint64() % (kGainCycleLength - 1);
  if (cycle_current_offset_ >= 1) {
    cycle_current_offset_ += 1;
  }
  last_cycle_start_ = now;
  pacing_gain_ = kPacingGain[cycle_current_offset_];
}
bool BbrSender::UpdateRoundTripCounter(QuicPacketNumber last_acked_packet) {
  if (!current_round_trip_end_.IsInitialized() ||
      last_acked_packet > current_round_trip_end_) {
    round_trip_count_++;
    current_round_trip_end_ = last_sent_packet_;
    if (stats_ && InSlowStart()) {
      ++stats_->slowstart_num_rtts;
    }
    return true;
  }
  return false;
}
bool BbrSender::MaybeUpdateMinRtt(QuicTime now,
                                  QuicTime::Delta sample_min_rtt) {
  bool min_rtt_expired =
      !min_rtt_.IsZero() && (now > (min_rtt_timestamp_ + kMinRttExpiry));
  if (min_rtt_expired || sample_min_rtt < min_rtt_ || min_rtt_.IsZero()) {
    QUIC_DVLOG(2) << "Min RTT updated, old value: " << min_rtt_
                  << ", new value: " << sample_min_rtt
                  << ", current time: " << now.ToDebuggingValue();
    min_rtt_ = sample_min_rtt;
    min_rtt_timestamp_ = now;
  }
  QUICHE_DCHECK(!min_rtt_.IsZero());
  return min_rtt_expired;
}
void BbrSender::UpdateGainCyclePhase(QuicTime now,
                                     QuicByteCount prior_in_flight,
                                     bool has_losses) {
  const QuicByteCount bytes_in_flight = unacked_packets_->bytes_in_flight();
  bool should_advance_gain_cycling = now - last_cycle_start_ > GetMinRtt();
  if (pacing_gain_ > 1.0 && !has_losses &&
      prior_in_flight < GetTargetCongestionWindow(pacing_gain_)) {
    should_advance_gain_cycling = false;
  }
  if (pacing_gain_ < 1.0 && bytes_in_flight <= GetTargetCongestionWindow(1)) {
    should_advance_gain_cycling = true;
  }
  if (should_advance_gain_cycling) {
    cycle_current_offset_ = (cycle_current_offset_ + 1) % kGainCycleLength;
    if (cycle_current_offset_ == 0) {
      ++stats_->bbr_num_cycles;
    }
    last_cycle_start_ = now;
    if (drain_to_target_ && pacing_gain_ < 1 &&
        kPacingGain[cycle_current_offset_] == 1 &&
        bytes_in_flight > GetTargetCongestionWindow(1)) {
      return;
    }
    pacing_gain_ = kPacingGain[cycle_current_offset_];
  }
}
void BbrSender::CheckIfFullBandwidthReached(
    const SendTimeState& last_packet_send_state) {
  if (last_sample_is_app_limited_) {
    return;
  }
  QuicBandwidth target = bandwidth_at_last_round_ * kStartupGrowthTarget;
  if (BandwidthEstimate() >= target) {
    bandwidth_at_last_round_ = BandwidthEstimate();
    rounds_without_bandwidth_gain_ = 0;
    if (expire_ack_aggregation_in_startup_) {
      sampler_.ResetMaxAckHeightTracker(0, round_trip_count_);
    }
    return;
  }
  rounds_without_bandwidth_gain_++;
  if ((rounds_without_bandwidth_gain_ >= num_startup_rtts_) ||
      ShouldExitStartupDueToLoss(last_packet_send_state)) {
    QUICHE_DCHECK(has_non_app_limited_sample_);
    is_at_full_bandwidth_ = true;
  }
}
void BbrSender::MaybeExitStartupOrDrain(QuicTime now) {
  if (mode_ == STARTUP && is_at_full_bandwidth_) {
    OnExitStartup(now);
    mode_ = DRAIN;
    pacing_gain_ = drain_gain_;
    congestion_window_gain_ = high_cwnd_gain_;
  }
  if (mode_ == DRAIN &&
      unacked_packets_->bytes_in_flight() <= GetTargetCongestionWindow(1)) {
    EnterProbeBandwidthMode(now);
  }
}
void BbrSender::OnExitStartup(QuicTime now) {
  QUICHE_DCHECK_EQ(mode_, STARTUP);
  if (stats_) {
    stats_->slowstart_duration.Stop(now);
  }
}
bool BbrSender::ShouldExitStartupDueToLoss(
    const SendTimeState& last_packet_send_state) const {
  if (num_loss_events_in_round_ <
          GetQuicFlag(quic_bbr2_default_startup_full_loss_count) ||
      !last_packet_send_state.is_valid) {
    return false;
  }
  const QuicByteCount inflight_at_send = last_packet_send_state.bytes_in_flight;
  if (inflight_at_send > 0 && bytes_lost_in_round_ > 0) {
    if (bytes_lost_in_round_ >
        inflight_at_send * GetQuicFlag(quic_bbr2_default_loss_threshold)) {
      stats_->bbr_exit_startup_due_to_loss = true;
      return true;
    }
    return false;
  }
  return false;
}
void BbrSender::MaybeEnterOrExitProbeRtt(QuicTime now, bool is_round_start,
                                         bool min_rtt_expired) {
  if (min_rtt_expired && !exiting_quiescence_ && mode_ != PROBE_RTT) {
    if (InSlowStart()) {
      OnExitStartup(now);
    }
    mode_ = PROBE_RTT;
    pacing_gain_ = 1;
    exit_probe_rtt_at_ = QuicTime::Zero();
  }
  if (mode_ == PROBE_RTT) {
    sampler_.OnAppLimited();
    if (exit_probe_rtt_at_ == QuicTime::Zero()) {
      if (unacked_packets_->bytes_in_flight() <
          ProbeRttCongestionWindow() + kMaxOutgoingPacketSize) {
        exit_probe_rtt_at_ = now + kProbeRttTime;
        probe_rtt_round_passed_ = false;
      }
    } else {
      if (is_round_start) {
        probe_rtt_round_passed_ = true;
      }
      if (now >= exit_probe_rtt_at_ && probe_rtt_round_passed_) {
        min_rtt_timestamp_ = now;
        if (!is_at_full_bandwidth_) {
          EnterStartupMode(now);
        } else {
          EnterProbeBandwidthMode(now);
        }
      }
    }
  }
  exiting_quiescence_ = false;
}
void BbrSender::UpdateRecoveryState(QuicPacketNumber last_acked_packet,
                                    bool has_losses, bool is_round_start) {
  if (!is_at_full_bandwidth_) {
    return;
  }
  if (has_losses) {
    end_recovery_at_ = last_sent_packet_;
  }
  switch (recovery_state_) {
    case NOT_IN_RECOVERY:
      if (has_losses) {
        recovery_state_ = CONSERVATION;
        recovery_window_ = 0;
        current_round_trip_end_ = last_sent_packet_;
      }
      break;
    case CONSERVATION:
      if (is_round_start) {
        recovery_state_ = GROWTH;
      }
      ABSL_FALLTHROUGH_INTENDED;
    case GROWTH:
      if (!has_losses && last_acked_packet > end_recovery_at_) {
        recovery_state_ = NOT_IN_RECOVERY;
      }
      break;
  }
}
void BbrSender::CalculatePacingRate(QuicByteCount bytes_lost) {
  if (BandwidthEstimate().IsZero()) {
    return;
  }
  QuicBandwidth target_rate = pacing_gain_ * BandwidthEstimate();
  if (is_at_full_bandwidth_) {
    pacing_rate_ = target_rate;
    return;
  }
  if (pacing_rate_.IsZero() && !rtt_stats_->min_rtt().IsZero()) {
    pacing_rate_ = QuicBandwidth::FromBytesAndTimeDelta(
        initial_congestion_window_, rtt_stats_->min_rtt());
    return;
  }
  if (detect_overshooting_) {
    bytes_lost_while_detecting_overshooting_ += bytes_lost;
    if (pacing_rate_ > target_rate &&
        bytes_lost_while_detecting_overshooting_ > 0) {
      if (has_non_app_limited_sample_ ||
          bytes_lost_while_detecting_overshooting_ *
                  bytes_lost_multiplier_while_detecting_overshooting_ >
              initial_congestion_window_) {
        pacing_rate_ = std::max(
            target_rate, QuicBandwidth::FromBytesAndTimeDelta(
                             cwnd_to_calculate_min_pacing_rate_, GetMinRtt()));
        if (stats_) {
          stats_->overshooting_detected_with_network_parameters_adjusted = true;
        }
        bytes_lost_while_detecting_overshooting_ = 0;
        detect_overshooting_ = false;
      }
    }
  }
  pacing_rate_ = std::max(pacing_rate_, target_rate);
}
void BbrSender::CalculateCongestionWindow(QuicByteCount bytes_acked,
                                          QuicByteCount excess_acked) {
  if (mode_ == PROBE_RTT) {
    return;
  }
  QuicByteCount target_window =
      GetTargetCongestionWindow(congestion_window_gain_);
  if (is_at_full_bandwidth_) {
    target_window += sampler_.max_ack_height();
  } else if (enable_ack_aggregation_during_startup_) {
    target_window += excess_acked;
  }
  if (is_at_full_bandwidth_) {
    congestion_window_ =
        std::min(target_window, congestion_window_ + bytes_acked);
  } else if (congestion_window_ < target_window ||
             sampler_.total_bytes_acked() < initial_congestion_window_) {
    congestion_window_ = congestion_window_ + bytes_acked;
  }
  congestion_window_ = std::max(congestion_window_, min_congestion_window_);
  congestion_window_ = std::min(congestion_window_, max_congestion_window_);
}
void BbrSender::CalculateRecoveryWindow(QuicByteCount bytes_acked,
                                        QuicByteCount bytes_lost) {
  if (recovery_state_ == NOT_IN_RECOVERY) {
    return;
  }
  if (recovery_window_ == 0) {
    recovery_window_ = unacked_packets_->bytes_in_flight() + bytes_acked;
    recovery_window_ = std::max(min_congestion_window_, recovery_window_);
    return;
  }
  recovery_window_ = recovery_window_ >= bytes_lost
                         ? recovery_window_ - bytes_lost
                         : kMaxSegmentSize;
  if (recovery_state_ == GROWTH) {
    recovery_window_ += bytes_acked;
  }
  recovery_window_ = std::max(
      recovery_window_, unacked_packets_->bytes_in_flight() + bytes_acked);
  recovery_window_ = std::max(min_congestion_window_, recovery_window_);
}
std::string BbrSender::GetDebugState() const {
  std::ostringstream stream;
  stream << ExportDebugState();
  return stream.str();
}
void BbrSender::OnApplicationLimited(QuicByteCount bytes_in_flight) {
  if (bytes_in_flight >= GetCongestionWindow()) {
    return;
  }
  sampler_.OnAppLimited();
  QUIC_DVLOG(2) << "Becoming application limited. Last sent packet: "
                << last_sent_packet_ << ", CWND: " << GetCongestionWindow();
}
void BbrSender::PopulateConnectionStats(QuicConnectionStats* stats) const {
  stats->num_ack_aggregation_epochs = sampler_.num_ack_aggregation_epochs();
}
BbrSender::DebugState BbrSender::ExportDebugState() const {
  return DebugState(*this);
}
static std::string ModeToString(BbrSender::Mode mode) {
  switch (mode) {
    case BbrSender::STARTUP:
      return "STARTUP";
    case BbrSender::DRAIN:
      return "DRAIN";
    case BbrSender::PROBE_BW:
      return "PROBE_BW";
    case BbrSender::PROBE_RTT:
      return "PROBE_RTT";
  }
  return "???";
}
std::ostream& operator<<(std::ostream& os, const BbrSender::Mode& mode) {
  os << ModeToString(mode);
  return os;
}
std::ostream& operator<<(std::ostream& os, const BbrSender::DebugState& state) {
  os << "Mode: " << ModeToString(state.mode) << std::endl;
  os << "Maximum bandwidth: " << state.max_bandwidth << std::endl;
  os << "Round trip counter: " << state.round_trip_count << std::endl;
  os << "Gain cycle index: " << static_cast<int>(state.gain_cycle_index)
     << std::endl;
  os << "Congestion window: " << state.congestion_window << " bytes"
     << std::endl;
  if (state.mode == BbrSender::STARTUP) {
    os << "(startup) Bandwidth at last round: " << state.bandwidth_at_last_round
       << std::endl;
    os << "(startup) Rounds without gain: "
       << state.rounds_without_bandwidth_gain << std::endl;
  }
  os << "Minimum RTT: " << state.min_rtt << std::endl;
  os << "Minimum RTT timestamp: " << state.min_rtt_timestamp.ToDebuggingValue()
     << std::endl;
  os << "Last sample is app-limited: "
     << (state.last_sample_is_app_limited ? "yes" : "no");
  return os;
}
}  