#include "quiche/quic/core/chlo_extractor.h"
#include <memory>
#include <optional>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "quiche/quic/core/crypto/crypto_framer.h"
#include "quiche/quic/core/crypto/crypto_handshake.h"
#include "quiche/quic/core/crypto/crypto_handshake_message.h"
#include "quiche/quic/core/crypto/quic_decrypter.h"
#include "quiche/quic/core/crypto/quic_encrypter.h"
#include "quiche/quic/core/frames/quic_ack_frequency_frame.h"
#include "quiche/quic/core/frames/quic_reset_stream_at_frame.h"
#include "quiche/quic/core/quic_framer.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/core/quic_utils.h"
namespace quic {
namespace {
class ChloFramerVisitor : public QuicFramerVisitorInterface,
                          public CryptoFramerVisitorInterface {
 public:
  ChloFramerVisitor(QuicFramer* framer,
                    const QuicTagVector& create_session_tag_indicators,
                    ChloExtractor::Delegate* delegate);
  ~ChloFramerVisitor() override = default;
  void OnError(QuicFramer* ) override {}
  bool OnProtocolVersionMismatch(ParsedQuicVersion version) override;
  void OnPacket() override {}
  void OnVersionNegotiationPacket(
      const QuicVersionNegotiationPacket& ) override {}
  void OnRetryPacket(QuicConnectionId ,
                     QuicConnectionId ,
                     absl::string_view ,
                     absl::string_view ,
                     absl::string_view ) override {}
  bool OnUnauthenticatedPublicHeader(const QuicPacketHeader& header) override;
  bool OnUnauthenticatedHeader(const QuicPacketHeader& header) override;
  void OnDecryptedPacket(size_t ,
                         EncryptionLevel ) override {}
  bool OnPacketHeader(const QuicPacketHeader& header) override;
  void OnCoalescedPacket(const QuicEncryptedPacket& packet) override;
  void OnUndecryptablePacket(const QuicEncryptedPacket& packet,
                             EncryptionLevel decryption_level,
                             bool has_decryption_key) override;
  bool OnStreamFrame(const QuicStreamFrame& frame) override;
  bool OnCryptoFrame(const QuicCryptoFrame& frame) override;
  bool OnAckFrameStart(QuicPacketNumber largest_acked,
                       QuicTime::Delta ack_delay_time) override;
  bool OnAckRange(QuicPacketNumber start, QuicPacketNumber end) override;
  bool OnAckTimestamp(QuicPacketNumber packet_number,
                      QuicTime timestamp) override;
  bool OnAckFrameEnd(QuicPacketNumber start,
                     const std::optional<QuicEcnCounts>& ecn_counts) override;
  bool OnStopWaitingFrame(const QuicStopWaitingFrame& frame) override;
  bool OnPingFrame(const QuicPingFrame& frame) override;
  bool OnRstStreamFrame(const QuicRstStreamFrame& frame) override;
  bool OnConnectionCloseFrame(const QuicConnectionCloseFrame& frame) override;
  bool OnNewConnectionIdFrame(const QuicNewConnectionIdFrame& frame) override;
  bool OnRetireConnectionIdFrame(
      const QuicRetireConnectionIdFrame& frame) override;
  bool OnNewTokenFrame(const QuicNewTokenFrame& frame) override;
  bool OnStopSendingFrame(const QuicStopSendingFrame& frame) override;
  bool OnPathChallengeFrame(const QuicPathChallengeFrame& frame) override;
  bool OnPathResponseFrame(const QuicPathResponseFrame& frame) override;
  bool OnGoAwayFrame(const QuicGoAwayFrame& frame) override;
  bool OnMaxStreamsFrame(const QuicMaxStreamsFrame& frame) override;
  bool OnStreamsBlockedFrame(const QuicStreamsBlockedFrame& frame) override;
  bool OnWindowUpdateFrame(const QuicWindowUpdateFrame& frame) override;
  bool OnBlockedFrame(const QuicBlockedFrame& frame) override;
  bool OnPaddingFrame(const QuicPaddingFrame& frame) override;
  bool OnMessageFrame(const QuicMessageFrame& frame) override;
  bool OnHandshakeDoneFrame(const QuicHandshakeDoneFrame& frame) override;
  bool OnAckFrequencyFrame(const QuicAckFrequencyFrame& farme) override;
  bool OnResetStreamAtFrame(const QuicResetStreamAtFrame& frame) override;
  void OnPacketComplete() override {}
  bool IsValidStatelessResetToken(
      const StatelessResetToken& token) const override;
  void OnAuthenticatedIetfStatelessResetPacket(
      const QuicIetfStatelessResetPacket& ) override {}
  void OnKeyUpdate(KeyUpdateReason ) override;
  void OnDecryptedFirstPacketInKeyPhase() override;
  std::unique_ptr<QuicDecrypter> AdvanceKeysAndCreateCurrentOneRttDecrypter()
      override;
  std::unique_ptr<QuicEncrypter> CreateCurrentOneRttEncrypter() override;
  void OnError(CryptoFramer* framer) override;
  void OnHandshakeMessage(const CryptoHandshakeMessage& message) override;
  bool OnHandshakeData(absl::string_view data);
  bool found_chlo() { return found_chlo_; }
  bool chlo_contains_tags() { return chlo_contains_tags_; }
 private:
  QuicFramer* framer_;
  const QuicTagVector& create_session_tag_indicators_;
  ChloExtractor::Delegate* delegate_;
  bool found_chlo_;
  bool chlo_contains_tags_;
  QuicConnectionId connection_id_;
};
ChloFramerVisitor::ChloFramerVisitor(
    QuicFramer* framer, const QuicTagVector& create_session_tag_indicators,
    ChloExtractor::Delegate* delegate)
    : framer_(framer),
      create_session_tag_indicators_(create_session_tag_indicators),
      delegate_(delegate),
      found_chlo_(false),
      chlo_contains_tags_(false),
      connection_id_(EmptyQuicConnectionId()) {}
bool ChloFramerVisitor::OnProtocolVersionMismatch(ParsedQuicVersion version) {
  if (!framer_->IsSupportedVersion(version)) {
    return false;
  }
  framer_->set_version(version);
  return true;
}
bool ChloFramerVisitor::OnUnauthenticatedPublicHeader(
    const QuicPacketHeader& header) {
  connection_id_ = header.destination_connection_id;
  framer_->SetInitialObfuscators(header.destination_connection_id);
  return true;
}
bool ChloFramerVisitor::OnUnauthenticatedHeader(
    const QuicPacketHeader& ) {
  return true;
}
bool ChloFramerVisitor::OnPacketHeader(const QuicPacketHeader& ) {
  return true;
}
void ChloFramerVisitor::OnCoalescedPacket(
    const QuicEncryptedPacket& ) {}
void ChloFramerVisitor::OnUndecryptablePacket(
    const QuicEncryptedPacket& , EncryptionLevel ,
    bool ) {}
bool ChloFramerVisitor::OnStreamFrame(const QuicStreamFrame& frame) {
  if (QuicVersionUsesCryptoFrames(framer_->transport_version())) {
    return false;
  }
  absl::string_view data(frame.data_buffer, frame.data_length);
  if (QuicUtils::IsCryptoStreamId(framer_->transport_version(),
                                  frame.stream_id) &&
      frame.offset == 0 && absl::StartsWith(data, "CHLO")) {
    return OnHandshakeData(data);
  }
  return true;
}
bool ChloFramerVisitor::OnCryptoFrame(const QuicCryptoFrame& frame) {
  if (!QuicVersionUsesCryptoFrames(framer_->transport_version())) {
    return false;
  }
  absl::string_view data(frame.data_buffer, frame.data_length);
  if (frame.offset == 0 && absl::StartsWith(data, "CHLO")) {
    return OnHandshakeData(data);
  }
  return true;
}
bool ChloFramerVisitor::OnHandshakeData(absl::string_view data) {
  CryptoFramer crypto_framer;
  crypto_framer.set_visitor(this);
  if (!crypto_framer.ProcessInput(data)) {
    return false;
  }
  for (const QuicTag tag : create_session_tag_indicators_) {
    if (crypto_framer.HasTag(tag)) {
      chlo_contains_tags_ = true;
    }
  }
  if (chlo_contains_tags_ && delegate_) {
    crypto_framer.ForceHandshake();
  }
  return true;
}
bool ChloFramerVisitor::OnAckFrameStart(QuicPacketNumber ,
                                        QuicTime::Delta ) {
  return true;
}
bool ChloFramerVisitor::OnResetStreamAtFrame(
    const QuicResetStreamAtFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnAckRange(QuicPacketNumber ,
                                   QuicPacketNumber ) {
  return true;
}
bool ChloFramerVisitor::OnAckTimestamp(QuicPacketNumber ,
                                       QuicTime ) {
  return true;
}
bool ChloFramerVisitor::OnAckFrameEnd(
    QuicPacketNumber ,
    const std::optional<QuicEcnCounts>& ) {
  return true;
}
bool ChloFramerVisitor::OnStopWaitingFrame(
    const QuicStopWaitingFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnPingFrame(const QuicPingFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnRstStreamFrame(const QuicRstStreamFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnConnectionCloseFrame(
    const QuicConnectionCloseFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnStopSendingFrame(
    const QuicStopSendingFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnPathChallengeFrame(
    const QuicPathChallengeFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnPathResponseFrame(
    const QuicPathResponseFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnGoAwayFrame(const QuicGoAwayFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnWindowUpdateFrame(
    const QuicWindowUpdateFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnBlockedFrame(const QuicBlockedFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnNewConnectionIdFrame(
    const QuicNewConnectionIdFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnRetireConnectionIdFrame(
    const QuicRetireConnectionIdFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnNewTokenFrame(const QuicNewTokenFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnPaddingFrame(const QuicPaddingFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnMessageFrame(const QuicMessageFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnHandshakeDoneFrame(
    const QuicHandshakeDoneFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnAckFrequencyFrame(
    const QuicAckFrequencyFrame& ) {
  return true;
}
bool ChloFramerVisitor::IsValidStatelessResetToken(
    const StatelessResetToken& ) const {
  return false;
}
bool ChloFramerVisitor::OnMaxStreamsFrame(
    const QuicMaxStreamsFrame& ) {
  return true;
}
bool ChloFramerVisitor::OnStreamsBlockedFrame(
    const QuicStreamsBlockedFrame& ) {
  return true;
}
void ChloFramerVisitor::OnKeyUpdate(KeyUpdateReason ) {}
void ChloFramerVisitor::OnDecryptedFirstPacketInKeyPhase() {}
std::unique_ptr<QuicDecrypter>
ChloFramerVisitor::AdvanceKeysAndCreateCurrentOneRttDecrypter() {
  return nullptr;
}
std::unique_ptr<QuicEncrypter>
ChloFramerVisitor::CreateCurrentOneRttEncrypter() {
  return nullptr;
}
void ChloFramerVisitor::OnError(CryptoFramer* ) {}
void ChloFramerVisitor::OnHandshakeMessage(
    const CryptoHandshakeMessage& message) {
  if (delegate_ != nullptr) {
    delegate_->OnChlo(framer_->transport_version(), connection_id_, message);
  }
  found_chlo_ = true;
}
}  
bool ChloExtractor::Extract(const QuicEncryptedPacket& packet,
                            ParsedQuicVersion version,
                            const QuicTagVector& create_session_tag_indicators,
                            Delegate* delegate, uint8_t connection_id_length) {
  QUIC_DVLOG(1) << "Extracting CHLO using version " << version;
  QuicFramer framer({version}, QuicTime::Zero(), Perspective::IS_SERVER,
                    connection_id_length);
  ChloFramerVisitor visitor(&framer, create_session_tag_indicators, delegate);
  framer.set_visitor(&visitor);
  if (!framer.ProcessPacket(packet)) {
    return false;
  }
  return visitor.found_chlo() || visitor.chlo_contains_tags();
}
}  