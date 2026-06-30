#include "quiche/quic/core/quic_stream_id_manager.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include "absl/strings/str_cat.h"
#include "quiche/quic/core/quic_connection.h"
#include "quiche/quic/core/quic_constants.h"
#include "quiche/quic/core/quic_session.h"
#include "quiche/quic/core/quic_utils.h"
#include "quiche/quic/core/quic_versions.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
#define ENDPOINT \
  (perspective_ == Perspective::IS_SERVER ? " Server: " : " Client: ")
QuicStreamIdManager::QuicStreamIdManager(
    DelegateInterface* delegate, bool unidirectional, Perspective perspective,
    ParsedQuicVersion version, QuicStreamCount max_allowed_outgoing_streams,
    QuicStreamCount max_allowed_incoming_streams)
    : delegate_(delegate),
      unidirectional_(unidirectional),
      perspective_(perspective),
      version_(version),
      outgoing_max_streams_(max_allowed_outgoing_streams),
      next_outgoing_stream_id_(GetFirstOutgoingStreamId()),
      outgoing_stream_count_(0),
      incoming_actual_max_streams_(max_allowed_incoming_streams),
      incoming_advertised_max_streams_(max_allowed_incoming_streams),
      incoming_initial_max_open_streams_(max_allowed_incoming_streams),
      incoming_stream_count_(0),
      largest_peer_created_stream_id_(
          QuicUtils::GetInvalidStreamId(version.transport_version)),
      stop_increasing_incoming_max_streams_(false) {}
QuicStreamIdManager::~QuicStreamIdManager() {}
bool QuicStreamIdManager::OnStreamsBlockedFrame(
    const QuicStreamsBlockedFrame& frame, std::string* error_details) {
  QUICHE_DCHECK_EQ(frame.unidirectional, unidirectional_);
  if (frame.stream_count > incoming_advertised_max_streams_) {
    *error_details = absl::StrCat(
        "StreamsBlockedFrame's stream count ", frame.stream_count,
        " exceeds incoming max stream ", incoming_advertised_max_streams_);
    return false;
  }
  QUICHE_DCHECK_LE(incoming_advertised_max_streams_,
                   incoming_actual_max_streams_);
  if (incoming_advertised_max_streams_ == incoming_actual_max_streams_) {
    return true;
  }
  if (frame.stream_count < incoming_actual_max_streams_ &&
      delegate_->CanSendMaxStreams()) {
    SendMaxStreamsFrame();
  }
  return true;
}
bool QuicStreamIdManager::MaybeAllowNewOutgoingStreams(
    QuicStreamCount max_open_streams) {
  if (max_open_streams <= outgoing_max_streams_) {
    return false;
  }
  outgoing_max_streams_ =
      std::min(max_open_streams, QuicUtils::GetMaxStreamCount());
  return true;
}
void QuicStreamIdManager::SetMaxOpenIncomingStreams(
    QuicStreamCount max_open_streams) {
  QUIC_BUG_IF(quic_bug_12413_1, incoming_stream_count_ > 0)
      << "non-zero incoming stream count " << incoming_stream_count_
      << " when setting max incoming stream to " << max_open_streams;
  QUIC_DLOG_IF(WARNING, incoming_initial_max_open_streams_ != max_open_streams)
      << absl::StrCat(unidirectional_ ? "unidirectional " : "bidirectional: ",
                      "incoming stream limit changed from ",
                      incoming_initial_max_open_streams_, " to ",
                      max_open_streams);
  incoming_actual_max_streams_ = max_open_streams;
  incoming_advertised_max_streams_ = max_open_streams;
  incoming_initial_max_open_streams_ = max_open_streams;
}
void QuicStreamIdManager::MaybeSendMaxStreamsFrame() {
  int divisor = GetQuicFlag(quic_max_streams_window_divisor);
  if (divisor > 0) {
    if ((incoming_advertised_max_streams_ - incoming_stream_count_) >
        (incoming_initial_max_open_streams_ / divisor)) {
      return;
    }
  }
  if (delegate_->CanSendMaxStreams() &&
      incoming_advertised_max_streams_ < incoming_actual_max_streams_) {
    SendMaxStreamsFrame();
  }
}
void QuicStreamIdManager::SendMaxStreamsFrame() {
  QUIC_BUG_IF(quic_bug_12413_2,
              incoming_advertised_max_streams_ >= incoming_actual_max_streams_);
  incoming_advertised_max_streams_ = incoming_actual_max_streams_;
  delegate_->SendMaxStreams(incoming_advertised_max_streams_, unidirectional_);
}
void QuicStreamIdManager::OnStreamClosed(QuicStreamId stream_id) {
  QUICHE_DCHECK_NE(QuicUtils::IsBidirectionalStreamId(stream_id, version_),
                   unidirectional_);
  if (QuicUtils::IsOutgoingStreamId(version_, stream_id, perspective_)) {
    return;
  }
  if (incoming_actual_max_streams_ == QuicUtils::GetMaxStreamCount()) {
    return;
  }
  if (!stop_increasing_incoming_max_streams_) {
    incoming_actual_max_streams_++;
    MaybeSendMaxStreamsFrame();
  }
}
QuicStreamId QuicStreamIdManager::GetNextOutgoingStreamId() {
  QUIC_BUG_IF(quic_bug_12413_3, outgoing_stream_count_ >= outgoing_max_streams_)
      << "Attempt to allocate a new outgoing stream that would exceed the "
         "limit ("
      << outgoing_max_streams_ << ")";
  QuicStreamId id = next_outgoing_stream_id_;
  next_outgoing_stream_id_ +=
      QuicUtils::StreamIdDelta(version_.transport_version);
  outgoing_stream_count_++;
  return id;
}
bool QuicStreamIdManager::CanOpenNextOutgoingStream() const {
  QUICHE_DCHECK(VersionHasIetfQuicFrames(version_.transport_version));
  return outgoing_stream_count_ < outgoing_max_streams_;
}
bool QuicStreamIdManager::MaybeIncreaseLargestPeerStreamId(
    const QuicStreamId stream_id, std::string* error_details) {
  QUICHE_DCHECK_NE(QuicUtils::IsBidirectionalStreamId(stream_id, version_),
                   unidirectional_);
  QUICHE_DCHECK_NE(QuicUtils::IsServerInitiatedStreamId(
                       version_.transport_version, stream_id),
                   perspective_ == Perspective::IS_SERVER);
  if (available_streams_.erase(stream_id) == 1) {
    return true;
  }
  if (largest_peer_created_stream_id_ !=
      QuicUtils::GetInvalidStreamId(version_.transport_version)) {
    QUICHE_DCHECK_GT(stream_id, largest_peer_created_stream_id_);
  }
  const QuicStreamCount delta =
      QuicUtils::StreamIdDelta(version_.transport_version);
  const QuicStreamId least_new_stream_id =
      largest_peer_created_stream_id_ ==
              QuicUtils::GetInvalidStreamId(version_.transport_version)
          ? GetFirstIncomingStreamId()
          : largest_peer_created_stream_id_ + delta;
  const QuicStreamCount stream_count_increment =
      (stream_id - least_new_stream_id) / delta + 1;
  if (incoming_stream_count_ + stream_count_increment >
      incoming_advertised_max_streams_) {
    QUIC_DLOG(INFO) << ENDPOINT
                    << "Failed to create a new incoming stream with id:"
                    << stream_id << ", reaching MAX_STREAMS limit: "
                    << incoming_advertised_max_streams_ << ".";
    *error_details = absl::StrCat("Stream id ", stream_id,
                                  " would exceed stream count limit ",
                                  incoming_advertised_max_streams_);
    return false;
  }
  for (QuicStreamId id = least_new_stream_id; id < stream_id; id += delta) {
    available_streams_.insert(id);
  }
  incoming_stream_count_ += stream_count_increment;
  largest_peer_created_stream_id_ = stream_id;
  return true;
}
bool QuicStreamIdManager::IsAvailableStream(QuicStreamId id) const {
  QUICHE_DCHECK_NE(QuicUtils::IsBidirectionalStreamId(id, version_),
                   unidirectional_);
  if (QuicUtils::IsOutgoingStreamId(version_, id, perspective_)) {
    return id >= next_outgoing_stream_id_;
  }
  return largest_peer_created_stream_id_ ==
             QuicUtils::GetInvalidStreamId(version_.transport_version) ||
         id > largest_peer_created_stream_id_ ||
         available_streams_.contains(id);
}
QuicStreamId QuicStreamIdManager::GetFirstOutgoingStreamId() const {
  return (unidirectional_) ? QuicUtils::GetFirstUnidirectionalStreamId(
                                 version_.transport_version, perspective_)
                           : QuicUtils::GetFirstBidirectionalStreamId(
                                 version_.transport_version, perspective_);
}
QuicStreamId QuicStreamIdManager::GetFirstIncomingStreamId() const {
  return (unidirectional_) ? QuicUtils::GetFirstUnidirectionalStreamId(
                                 version_.transport_version,
                                 QuicUtils::InvertPerspective(perspective_))
                           : QuicUtils::GetFirstBidirectionalStreamId(
                                 version_.transport_version,
                                 QuicUtils::InvertPerspective(perspective_));
}
QuicStreamCount QuicStreamIdManager::available_incoming_streams() const {
  return incoming_advertised_max_streams_ - incoming_stream_count_;
}
}  