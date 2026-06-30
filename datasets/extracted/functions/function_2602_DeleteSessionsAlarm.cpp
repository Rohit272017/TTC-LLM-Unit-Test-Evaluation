#include "quiche/quic/core/quic_dispatcher.h"
#include <openssl/ssl.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "quiche/quic/core/chlo_extractor.h"
#include "quiche/quic/core/connection_id_generator.h"
#include "quiche/quic/core/crypto/crypto_handshake_message.h"
#include "quiche/quic/core/crypto/crypto_protocol.h"
#include "quiche/quic/core/crypto/quic_compressed_certs_cache.h"
#include "quiche/quic/core/frames/quic_connection_close_frame.h"
#include "quiche/quic/core/frames/quic_frame.h"
#include "quiche/quic/core/frames/quic_rst_stream_frame.h"
#include "quiche/quic/core/frames/quic_stop_sending_frame.h"
#include "quiche/quic/core/quic_alarm.h"
#include "quiche/quic/core/quic_alarm_factory.h"
#include "quiche/quic/core/quic_blocked_writer_interface.h"
#include "quiche/quic/core/quic_buffered_packet_store.h"
#include "quiche/quic/core/quic_connection.h"
#include "quiche/quic/core/quic_connection_id.h"
#include "quiche/quic/core/quic_constants.h"
#include "quiche/quic/core/quic_crypto_server_stream_base.h"
#include "quiche/quic/core/quic_data_writer.h"
#include "quiche/quic/core/quic_error_codes.h"
#include "quiche/quic/core/quic_framer.h"
#include "quiche/quic/core/quic_packet_creator.h"
#include "quiche/quic/core/quic_packet_number.h"
#include "quiche/quic/core/quic_packet_writer.h"
#include "quiche/quic/core/quic_packets.h"
#include "quiche/quic/core/quic_session.h"
#include "quiche/quic/core/quic_stream_frame_data_producer.h"
#include "quiche/quic/core/quic_stream_send_buffer.h"
#include "quiche/quic/core/quic_time.h"
#include "quiche/quic/core/quic_time_wait_list_manager.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/core/quic_utils.h"
#include "quiche/quic/core/quic_version_manager.h"
#include "quiche/quic/core/quic_versions.h"
#include "quiche/quic/core/tls_chlo_extractor.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/quic/platform/api/quic_socket_address.h"
#include "quiche/quic/platform/api/quic_stack_trace.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/print_elements.h"
#include "quiche/common/quiche_buffer_allocator.h"
#include "quiche/common/quiche_callbacks.h"
#include "quiche/common/quiche_text_utils.h"
namespace quic {
using BufferedPacket = QuicBufferedPacketStore::BufferedPacket;
using BufferedPacketList = QuicBufferedPacketStore::BufferedPacketList;
using EnqueuePacketResult = QuicBufferedPacketStore::EnqueuePacketResult;
namespace {
const QuicPacketLength kMinClientInitialPacketLength = 1200;
class DeleteSessionsAlarm : public QuicAlarm::DelegateWithoutContext {
 public:
  explicit DeleteSessionsAlarm(QuicDispatcher* dispatcher)
      : dispatcher_(dispatcher) {}
  DeleteSessionsAlarm(const DeleteSessionsAlarm&) = delete;
  DeleteSessionsAlarm& operator=(const DeleteSessionsAlarm&) = delete;
  void OnAlarm() override { dispatcher_->DeleteSessions(); }
 private:
  QuicDispatcher* dispatcher_;
};
class ClearStatelessResetAddressesAlarm
    : public QuicAlarm::DelegateWithoutContext {
 public:
  explicit ClearStatelessResetAddressesAlarm(QuicDispatcher* dispatcher)
      : dispatcher_(dispatcher) {}
  ClearStatelessResetAddressesAlarm(const DeleteSessionsAlarm&) = delete;
  ClearStatelessResetAddressesAlarm& operator=(const DeleteSessionsAlarm&) =
      delete;
  void OnAlarm() override { dispatcher_->ClearStatelessResetAddresses(); }
 private:
  QuicDispatcher* dispatcher_;
};
class StatelessConnectionTerminator {
 public:
  StatelessConnectionTerminator(QuicConnectionId server_connection_id,
                                QuicConnectionId original_server_connection_id,
                                const ParsedQuicVersion version,
                                QuicPacketNumber last_sent_packet_number,
                                QuicConnectionHelperInterface* helper,
                                QuicTimeWaitListManager* time_wait_list_manager)
      : server_connection_id_(server_connection_id),
        framer_(ParsedQuicVersionVector{version},
                 QuicTime::Zero(), Perspective::IS_SERVER,
                 kQuicDefaultConnectionIdLength),
        collector_(helper->GetStreamSendBufferAllocator()),
        creator_(server_connection_id, &framer_, &collector_),
        time_wait_list_manager_(time_wait_list_manager) {
    framer_.set_data_producer(&collector_);
    framer_.SetInitialObfuscators(original_server_connection_id);
    if (last_sent_packet_number.IsInitialized()) {
      QUICHE_DCHECK(
          GetQuicRestartFlag(quic_dispatcher_ack_buffered_initial_packets));
      QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets, 3,
                                8);
      creator_.set_packet_number(last_sent_packet_number);
    }
  }
  ~StatelessConnectionTerminator() {
    framer_.set_data_producer(nullptr);
  }
  void CloseConnection(QuicErrorCode error_code,
                       const std::string& error_details, bool ietf_quic,
                       std::vector<QuicConnectionId> active_connection_ids) {
    SerializeConnectionClosePacket(error_code, error_details);
    time_wait_list_manager_->AddConnectionIdToTimeWait(
        QuicTimeWaitListManager::SEND_TERMINATION_PACKETS,
        TimeWaitConnectionInfo(ietf_quic, collector_.packets(),
                               std::move(active_connection_ids),
                               QuicTime::Delta::Zero()));
  }
 private:
  void SerializeConnectionClosePacket(QuicErrorCode error_code,
                                      const std::string& error_details) {
    QuicConnectionCloseFrame* frame =
        new QuicConnectionCloseFrame(framer_.transport_version(), error_code,
                                     NO_IETF_QUIC_ERROR, error_details,
                                     0);
    if (!creator_.AddFrame(QuicFrame(frame), NOT_RETRANSMISSION)) {
      QUIC_BUG(quic_bug_10287_1) << "Unable to add frame to an empty packet";
      delete frame;
      return;
    }
    creator_.FlushCurrentPacket();
    QUICHE_DCHECK_EQ(1u, collector_.packets()->size());
  }
  QuicConnectionId server_connection_id_;
  QuicFramer framer_;
  PacketCollector collector_;
  QuicPacketCreator creator_;
  QuicTimeWaitListManager* time_wait_list_manager_;
};
class ChloAlpnSniExtractor : public ChloExtractor::Delegate {
 public:
  void OnChlo(QuicTransportVersion ,
              QuicConnectionId ,
              const CryptoHandshakeMessage& chlo) override {
    absl::string_view alpn_value;
    if (chlo.GetStringPiece(kALPN, &alpn_value)) {
      alpn_ = std::string(alpn_value);
    }
    absl::string_view sni;
    if (chlo.GetStringPiece(quic::kSNI, &sni)) {
      sni_ = std::string(sni);
    }
    absl::string_view uaid_value;
    if (chlo.GetStringPiece(quic::kUAID, &uaid_value)) {
      uaid_ = std::string(uaid_value);
    }
  }
  std::string&& ConsumeAlpn() { return std::move(alpn_); }
  std::string&& ConsumeSni() { return std::move(sni_); }
  std::string&& ConsumeUaid() { return std::move(uaid_); }
 private:
  std::string alpn_;
  std::string sni_;
  std::string uaid_;
};
}  
QuicDispatcher::QuicDispatcher(
    const QuicConfig* config, const QuicCryptoServerConfig* crypto_config,
    QuicVersionManager* version_manager,
    std::unique_ptr<QuicConnectionHelperInterface> helper,
    std::unique_ptr<QuicCryptoServerStreamBase::Helper> session_helper,
    std::unique_ptr<QuicAlarmFactory> alarm_factory,
    uint8_t expected_server_connection_id_length,
    ConnectionIdGeneratorInterface& connection_id_generator)
    : config_(config),
      crypto_config_(crypto_config),
      compressed_certs_cache_(
          QuicCompressedCertsCache::kQuicCompressedCertsCacheSize),
      helper_(std::move(helper)),
      session_helper_(std::move(session_helper)),
      alarm_factory_(std::move(alarm_factory)),
      delete_sessions_alarm_(
          alarm_factory_->CreateAlarm(new DeleteSessionsAlarm(this))),
      buffered_packets_(this, helper_->GetClock(), alarm_factory_.get(),
                        stats_),
      version_manager_(version_manager),
      last_error_(QUIC_NO_ERROR),
      new_sessions_allowed_per_event_loop_(0u),
      accept_new_connections_(true),
      expected_server_connection_id_length_(
          expected_server_connection_id_length),
      clear_stateless_reset_addresses_alarm_(alarm_factory_->CreateAlarm(
          new ClearStatelessResetAddressesAlarm(this))),
      connection_id_generator_(connection_id_generator) {
  QUIC_DLOG(INFO) << "Created QuicDispatcher with versions: "
                  << ParsedQuicVersionVectorToString(GetSupportedVersions());
}
QuicDispatcher::~QuicDispatcher() {
  if (delete_sessions_alarm_ != nullptr) {
    delete_sessions_alarm_->PermanentCancel();
  }
  if (clear_stateless_reset_addresses_alarm_ != nullptr) {
    clear_stateless_reset_addresses_alarm_->PermanentCancel();
  }
  reference_counted_session_map_.clear();
  closed_session_list_.clear();
  num_sessions_in_session_map_ = 0;
}
void QuicDispatcher::InitializeWithWriter(QuicPacketWriter* writer) {
  QUICHE_DCHECK(writer_ == nullptr);
  writer_.reset(writer);
  buffered_packets_.set_writer(writer);
  time_wait_list_manager_.reset(CreateQuicTimeWaitListManager());
}
void QuicDispatcher::ProcessPacket(const QuicSocketAddress& self_address,
                                   const QuicSocketAddress& peer_address,
                                   const QuicReceivedPacket& packet) {
  QUIC_DVLOG(2) << "Dispatcher received encrypted " << packet.length()
                << " bytes:" << std::endl
                << quiche::QuicheTextUtils::HexDump(
                       absl::string_view(packet.data(), packet.length()));
  ++stats_.packets_processed;
  ReceivedPacketInfo packet_info(self_address, peer_address, packet);
  std::string detailed_error;
  QuicErrorCode error;
  error = QuicFramer::ParsePublicHeaderDispatcherShortHeaderLengthUnknown(
      packet, &packet_info.form, &packet_info.long_packet_type,
      &packet_info.version_flag, &packet_info.use_length_prefix,
      &packet_info.version_label, &packet_info.version,
      &packet_info.destination_connection_id, &packet_info.source_connection_id,
      &packet_info.retry_token, &detailed_error, connection_id_generator_);
  if (error != QUIC_NO_ERROR) {
    SetLastError(error);
    QUIC_DLOG(ERROR) << detailed_error;
    return;
  }
  if (packet_info.destination_connection_id.length() !=
          expected_server_connection_id_length_ &&
      packet_info.version.IsKnown() &&
      !packet_info.version.AllowsVariableLengthConnectionIds()) {
    SetLastError(QUIC_INVALID_PACKET_HEADER);
    QUIC_DLOG(ERROR) << "Invalid Connection Id Length";
    return;
  }
  if (packet_info.version_flag && IsSupportedVersion(packet_info.version)) {
    if (!QuicUtils::IsConnectionIdValidForVersion(
            packet_info.destination_connection_id,
            packet_info.version.transport_version)) {
      SetLastError(QUIC_INVALID_PACKET_HEADER);
      QUIC_DLOG(ERROR)
          << "Invalid destination connection ID length for version";
      return;
    }
    if (packet_info.version.SupportsClientConnectionIds() &&
        !QuicUtils::IsConnectionIdValidForVersion(
            packet_info.source_connection_id,
            packet_info.version.transport_version)) {
      SetLastError(QUIC_INVALID_PACKET_HEADER);
      QUIC_DLOG(ERROR) << "Invalid source connection ID length for version";
      return;
    }
  }
#ifndef NDEBUG
  if (ack_buffered_initial_packets()) {
    const BufferedPacketList* packet_list =
        buffered_packets_.GetPacketList(packet_info.destination_connection_id);
    if (packet_list != nullptr &&
        packet_list->replaced_connection_id.has_value() &&
        *packet_list->replaced_connection_id ==
            packet_info.destination_connection_id) {
      QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets, 4,
                                8);
      ++stats_.packets_processed_with_replaced_cid_in_store;
    }
  }
#endif
  if (MaybeDispatchPacket(packet_info)) {
    return;
  }
  if (!packet_info.version_flag &&
      IsSupportedVersion(ParsedQuicVersion::Q046())) {
    ReceivedPacketInfo gquic_packet_info(self_address, peer_address, packet);
    const QuicErrorCode gquic_error = QuicFramer::ParsePublicHeaderDispatcher(
        packet, expected_server_connection_id_length_, &gquic_packet_info.form,
        &gquic_packet_info.long_packet_type, &gquic_packet_info.version_flag,
        &gquic_packet_info.use_length_prefix, &gquic_packet_info.version_label,
        &gquic_packet_info.version,
        &gquic_packet_info.destination_connection_id,
        &gquic_packet_info.source_connection_id, &gquic_packet_info.retry_token,
        &detailed_error);
    if (gquic_error == QUIC_NO_ERROR) {
      if (MaybeDispatchPacket(gquic_packet_info)) {
        return;
      }
    } else {
      QUICHE_VLOG(1) << "Tried to parse short header as gQUIC packet: "
                     << detailed_error;
    }
  }
  ProcessHeader(&packet_info);
}
namespace {
constexpr bool IsSourceUdpPortBlocked(uint16_t port) {
  constexpr uint16_t blocked_ports[] = {
      0,      
      17,     
      19,     
      53,     
      111,    
      123,    
      137,    
      138,    
      161,    
      389,    
      500,    
      1900,   
      3702,   
      5353,   
      5355,   
      11211,  
  };
  constexpr size_t num_blocked_ports = ABSL_ARRAYSIZE(blocked_ports);
  constexpr uint16_t highest_blocked_port =
      blocked_ports[num_blocked_ports - 1];
  if (ABSL_PREDICT_TRUE(port > highest_blocked_port)) {
    return false;
  }
  for (size_t i = 0; i < num_blocked_ports; i++) {
    if (port == blocked_ports[i]) {
      return true;
    }
  }
  return false;
}
}  
bool QuicDispatcher::MaybeDispatchPacket(
    const ReceivedPacketInfo& packet_info) {
  if (IsSourceUdpPortBlocked(packet_info.peer_address.port())) {
    QUIC_CODE_COUNT(quic_dropped_blocked_port);
    return true;
  }
  const QuicConnectionId server_connection_id =
      packet_info.destination_connection_id;
  if (packet_info.version_flag && packet_info.version.IsKnown() &&
      IsServerConnectionIdTooShort(server_connection_id)) {
    QUICHE_DCHECK(packet_info.version_flag);
    QUICHE_DCHECK(packet_info.version.AllowsVariableLengthConnectionIds());
    QUIC_DLOG(INFO) << "Packet with short destination connection ID "
                    << server_connection_id << " expected "
                    << static_cast<int>(expected_server_connection_id_length_);
    QUIC_CODE_COUNT(quic_dropped_invalid_small_initial_connection_id);
    return true;
  }
  if (packet_info.version_flag && packet_info.version.IsKnown() &&
      !QuicUtils::IsConnectionIdLengthValidForVersion(
          server_connection_id.length(),
          packet_info.version.transport_version)) {
    QUIC_DLOG(INFO) << "Packet with destination connection ID "
                    << server_connection_id << " is invalid with version "
                    << packet_info.version;
    QUIC_CODE_COUNT(quic_dropped_invalid_initial_connection_id);
    return true;
  }
  auto it = reference_counted_session_map_.find(server_connection_id);
  if (it != reference_counted_session_map_.end()) {
    QUICHE_DCHECK(!buffered_packets_.HasBufferedPackets(server_connection_id));
    it->second->ProcessUdpPacket(packet_info.self_address,
                                 packet_info.peer_address, packet_info.packet);
    return true;
  }
  if (buffered_packets_.HasChloForConnection(server_connection_id)) {
    EnqueuePacketResult rs = buffered_packets_.EnqueuePacket(
        packet_info,
        std::nullopt, ConnectionIdGenerator());
    switch (rs) {
      case EnqueuePacketResult::SUCCESS:
        break;
      case EnqueuePacketResult::CID_COLLISION:
        QUICHE_DCHECK(false) << "Connection " << server_connection_id
                             << " already has a CHLO buffered, but "
                                "EnqueuePacket returned CID_COLLISION.";
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_PACKETS:
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_CONNECTIONS:
        OnBufferPacketFailure(rs, packet_info.destination_connection_id);
        break;
    }
    return true;
  }
  if (OnFailedToDispatchPacket(packet_info)) {
    return true;
  }
  if (time_wait_list_manager_->IsConnectionIdInTimeWait(server_connection_id)) {
    time_wait_list_manager_->ProcessPacket(
        packet_info.self_address, packet_info.peer_address,
        packet_info.destination_connection_id, packet_info.form,
        packet_info.packet.length(), GetPerPacketContext());
    return true;
  }
  if (!accept_new_connections_ && packet_info.version_flag) {
    StatelesslyTerminateConnection(
        packet_info.self_address, packet_info.peer_address,
        packet_info.destination_connection_id, packet_info.form,
        packet_info.version_flag, packet_info.use_length_prefix,
        packet_info.version, QUIC_HANDSHAKE_FAILED,
        "Stop accepting new connections",
        quic::QuicTimeWaitListManager::SEND_STATELESS_RESET);
    time_wait_list_manager()->ProcessPacket(
        packet_info.self_address, packet_info.peer_address,
        packet_info.destination_connection_id, packet_info.form,
        packet_info.packet.length(), GetPerPacketContext());
    OnNewConnectionRejected();
    return true;
  }
  if (packet_info.version_flag) {
    if (!IsSupportedVersion(packet_info.version)) {
      if (ShouldCreateSessionForUnknownVersion(packet_info)) {
        return false;
      }
      MaybeSendVersionNegotiationPacket(packet_info);
      return true;
    }
    if (crypto_config()->validate_chlo_size() &&
        packet_info.form == IETF_QUIC_LONG_HEADER_PACKET &&
        packet_info.long_packet_type == INITIAL &&
        packet_info.packet.length() < kMinClientInitialPacketLength) {
      QUIC_DVLOG(1) << "Dropping initial packet which is too short, length: "
                    << packet_info.packet.length();
      QUIC_CODE_COUNT(quic_drop_small_initial_packets);
      return true;
    }
  }
  return false;
}
void QuicDispatcher::ProcessHeader(ReceivedPacketInfo* packet_info) {
  ++stats_.packets_processed_with_unknown_cid;
  QuicConnectionId server_connection_id =
      packet_info->destination_connection_id;
  QuicPacketFate fate = ValidityChecks(*packet_info);
  QuicErrorCode connection_close_error_code = QUIC_HANDSHAKE_FAILED;
  std::string tls_alert_error_detail;
  if (fate == kFateProcess) {
    ExtractChloResult extract_chlo_result =
        TryExtractChloOrBufferEarlyPacket(*packet_info);
    auto& parsed_chlo = extract_chlo_result.parsed_chlo;
    if (extract_chlo_result.tls_alert.has_value()) {
      QUIC_BUG_IF(quic_dispatcher_parsed_chlo_and_tls_alert_coexist_1,
                  parsed_chlo.has_value())
          << "parsed_chlo and tls_alert should not be set at the same time.";
      fate = kFateTimeWait;
      uint8_t tls_alert = *extract_chlo_result.tls_alert;
      connection_close_error_code = TlsAlertToQuicErrorCode(tls_alert);
      tls_alert_error_detail =
          absl::StrCat("TLS handshake failure from dispatcher (",
                       EncryptionLevelToString(ENCRYPTION_INITIAL), ") ",
                       static_cast<int>(tls_alert), ": ",
                       SSL_alert_desc_string_long(tls_alert));
    } else if (!parsed_chlo.has_value()) {
      return;
    } else {
      fate = ValidityChecksOnFullChlo(*packet_info, *parsed_chlo);
      if (fate == kFateProcess) {
        ProcessChlo(*std::move(parsed_chlo), packet_info);
        return;
      }
    }
  }
  switch (fate) {
    case kFateProcess:
      QUIC_BUG(quic_dispatcher_bad_packet_fate) << fate;
      break;
    case kFateTimeWait: {
      QUIC_DLOG(INFO) << "Adding connection ID " << server_connection_id
                      << " to time-wait list.";
      QUIC_CODE_COUNT(quic_reject_fate_time_wait);
      const std::string& connection_close_error_detail =
          tls_alert_error_detail.empty() ? "Reject connection"
                                         : tls_alert_error_detail;
      StatelesslyTerminateConnection(
          packet_info->self_address, packet_info->peer_address,
          server_connection_id, packet_info->form, packet_info->version_flag,
          packet_info->use_length_prefix, packet_info->version,
          connection_close_error_code, connection_close_error_detail,
          quic::QuicTimeWaitListManager::SEND_STATELESS_RESET);
      QUICHE_DCHECK(time_wait_list_manager_->IsConnectionIdInTimeWait(
          server_connection_id));
      time_wait_list_manager_->ProcessPacket(
          packet_info->self_address, packet_info->peer_address,
          server_connection_id, packet_info->form, packet_info->packet.length(),
          GetPerPacketContext());
      buffered_packets_.DiscardPackets(server_connection_id);
    } break;
    case kFateDrop:
      break;
  }
}
QuicDispatcher::ExtractChloResult
QuicDispatcher::TryExtractChloOrBufferEarlyPacket(
    const ReceivedPacketInfo& packet_info) {
  ExtractChloResult result;
  if (packet_info.version.UsesTls()) {
    bool has_full_tls_chlo = false;
    std::string sni;
    std::vector<uint16_t> supported_groups;
    std::vector<uint16_t> cert_compression_algos;
    std::vector<std::string> alpns;
    bool resumption_attempted = false, early_data_attempted = false;
    if (buffered_packets_.HasBufferedPackets(
            packet_info.destination_connection_id)) {
      has_full_tls_chlo = buffered_packets_.IngestPacketForTlsChloExtraction(
          packet_info.destination_connection_id, packet_info.version,
          packet_info.packet, &supported_groups, &cert_compression_algos,
          &alpns, &sni, &resumption_attempted, &early_data_attempted,
          &result.tls_alert);
    } else {
      TlsChloExtractor tls_chlo_extractor;
      tls_chlo_extractor.IngestPacket(packet_info.version, packet_info.packet);
      if (tls_chlo_extractor.HasParsedFullChlo()) {
        has_full_tls_chlo = true;
        supported_groups = tls_chlo_extractor.supported_groups();
        cert_compression_algos = tls_chlo_extractor.cert_compression_algos();
        alpns = tls_chlo_extractor.alpns();
        sni = tls_chlo_extractor.server_name();
        resumption_attempted = tls_chlo_extractor.resumption_attempted();
        early_data_attempted = tls_chlo_extractor.early_data_attempted();
      } else {
        result.tls_alert = tls_chlo_extractor.tls_alert();
      }
    }
    if (result.tls_alert.has_value()) {
      QUIC_BUG_IF(quic_dispatcher_parsed_chlo_and_tls_alert_coexist_2,
                  has_full_tls_chlo)
          << "parsed_chlo and tls_alert should not be set at the same time.";
      return result;
    }
    if (GetQuicFlag(quic_allow_chlo_buffering) && !has_full_tls_chlo) {
      EnqueuePacketResult rs = buffered_packets_.EnqueuePacket(
          packet_info,
          std::nullopt, ConnectionIdGenerator());
      switch (rs) {
        case EnqueuePacketResult::SUCCESS:
          break;
        case EnqueuePacketResult::CID_COLLISION:
          buffered_packets_.DiscardPackets(
              packet_info.destination_connection_id);
          ABSL_FALLTHROUGH_INTENDED;
        case EnqueuePacketResult::TOO_MANY_PACKETS:
          ABSL_FALLTHROUGH_INTENDED;
        case EnqueuePacketResult::TOO_MANY_CONNECTIONS:
          OnBufferPacketFailure(rs, packet_info.destination_connection_id);
          break;
      }
      return result;
    }
    ParsedClientHello& parsed_chlo = result.parsed_chlo.emplace();
    parsed_chlo.sni = std::move(sni);
    parsed_chlo.supported_groups = std::move(supported_groups);
    parsed_chlo.cert_compression_algos = std::move(cert_compression_algos);
    parsed_chlo.alpns = std::move(alpns);
    if (packet_info.retry_token.has_value()) {
      parsed_chlo.retry_token = std::string(*packet_info.retry_token);
    }
    parsed_chlo.resumption_attempted = resumption_attempted;
    parsed_chlo.early_data_attempted = early_data_attempted;
    return result;
  }
  ChloAlpnSniExtractor alpn_extractor;
  if (GetQuicFlag(quic_allow_chlo_buffering) &&
      !ChloExtractor::Extract(packet_info.packet, packet_info.version,
                              config_->create_session_tag_indicators(),
                              &alpn_extractor,
                              packet_info.destination_connection_id.length())) {
    EnqueuePacketResult rs = buffered_packets_.EnqueuePacket(
        packet_info,
        std::nullopt, ConnectionIdGenerator());
    switch (rs) {
      case EnqueuePacketResult::SUCCESS:
        break;
      case EnqueuePacketResult::CID_COLLISION:
        QUIC_BUG(quic_store_cid_collision_from_gquic_packet);
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_PACKETS:
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_CONNECTIONS:
        OnBufferPacketFailure(rs, packet_info.destination_connection_id);
        break;
    }
    return result;
  }
  ParsedClientHello& parsed_chlo = result.parsed_chlo.emplace();
  parsed_chlo.sni = alpn_extractor.ConsumeSni();
  parsed_chlo.uaid = alpn_extractor.ConsumeUaid();
  parsed_chlo.alpns = {alpn_extractor.ConsumeAlpn()};
  return result;
}
std::string QuicDispatcher::SelectAlpn(const std::vector<std::string>& alpns) {
  if (alpns.empty()) {
    return "";
  }
  if (alpns.size() > 1u) {
    const std::vector<std::string>& supported_alpns =
        version_manager_->GetSupportedAlpns();
    for (const std::string& alpn : alpns) {
      if (std::find(supported_alpns.begin(), supported_alpns.end(), alpn) !=
          supported_alpns.end()) {
        return alpn;
      }
    }
  }
  return alpns[0];
}
QuicDispatcher::QuicPacketFate QuicDispatcher::ValidityChecks(
    const ReceivedPacketInfo& packet_info) {
  if (!packet_info.version_flag) {
    QUIC_DLOG(INFO)
        << "Packet without version arrived for unknown connection ID "
        << packet_info.destination_connection_id;
    MaybeResetPacketsWithNoVersion(packet_info);
    return kFateDrop;
  }
  return kFateProcess;
}
void QuicDispatcher::CleanUpSession(QuicConnectionId server_connection_id,
                                    QuicConnection* connection,
                                    QuicErrorCode ,
                                    const std::string& ,
                                    ConnectionCloseSource ) {
  write_blocked_list_.Remove(*connection);
  QuicTimeWaitListManager::TimeWaitAction action =
      QuicTimeWaitListManager::SEND_STATELESS_RESET;
  if (connection->termination_packets() != nullptr &&
      !connection->termination_packets()->empty()) {
    action = QuicTimeWaitListManager::SEND_CONNECTION_CLOSE_PACKETS;
  } else {
    if (!connection->IsHandshakeComplete()) {
      QUIC_CODE_COUNT(quic_v44_add_to_time_wait_list_with_handshake_failed);
      StatelessConnectionTerminator terminator(
          server_connection_id,
          connection->GetOriginalDestinationConnectionId(),
          connection->version(), QuicPacketNumber(),
          helper_.get(), time_wait_list_manager_.get());
      terminator.CloseConnection(
          QUIC_HANDSHAKE_FAILED,
          "Connection is closed by server before handshake confirmed",
          true, connection->GetActiveServerConnectionIds());
      return;
    }
    QUIC_CODE_COUNT(quic_v44_add_to_time_wait_list_with_stateless_reset);
  }
  time_wait_list_manager_->AddConnectionIdToTimeWait(
      action,
      TimeWaitConnectionInfo(
          true, connection->termination_packets(),
          connection->GetActiveServerConnectionIds(),
          connection->sent_packet_manager().GetRttStats()->smoothed_rtt()));
}
void QuicDispatcher::StartAcceptingNewConnections() {
  accept_new_connections_ = true;
}
void QuicDispatcher::StopAcceptingNewConnections() {
  accept_new_connections_ = false;
  buffered_packets_.DiscardAllPackets();
}
void QuicDispatcher::PerformActionOnActiveSessions(
    quiche::UnretainedCallback<void(QuicSession*)> operation) const {
  absl::flat_hash_set<QuicSession*> visited_session;
  visited_session.reserve(reference_counted_session_map_.size());
  for (auto const& kv : reference_counted_session_map_) {
    QuicSession* session = kv.second.get();
    if (visited_session.insert(session).second) {
      operation(session);
    }
  }
}
std::vector<std::shared_ptr<QuicSession>> QuicDispatcher::GetSessionsSnapshot()
    const {
  std::vector<std::shared_ptr<QuicSession>> snapshot;
  snapshot.reserve(reference_counted_session_map_.size());
  absl::flat_hash_set<QuicSession*> visited_session;
  visited_session.reserve(reference_counted_session_map_.size());
  for (auto const& kv : reference_counted_session_map_) {
    QuicSession* session = kv.second.get();
    if (visited_session.insert(session).second) {
      snapshot.push_back(kv.second);
    }
  }
  return snapshot;
}
std::unique_ptr<QuicPerPacketContext> QuicDispatcher::GetPerPacketContext()
    const {
  return nullptr;
}
void QuicDispatcher::DeleteSessions() {
  if (!write_blocked_list_.Empty()) {
    for (const auto& session : closed_session_list_) {
      if (write_blocked_list_.Remove(*session->connection())) {
        QUIC_BUG(quic_bug_12724_2)
            << "QuicConnection was in WriteBlockedList before destruction "
            << session->connection()->connection_id();
      }
    }
  }
  closed_session_list_.clear();
}
void QuicDispatcher::ClearStatelessResetAddresses() {
  recent_stateless_reset_addresses_.clear();
}
void QuicDispatcher::OnCanWrite() {
  writer_->SetWritable();
  write_blocked_list_.OnWriterUnblocked();
}
bool QuicDispatcher::HasPendingWrites() const {
  return !write_blocked_list_.Empty();
}
void QuicDispatcher::Shutdown() {
  while (!reference_counted_session_map_.empty()) {
    QuicSession* session = reference_counted_session_map_.begin()->second.get();
    session->connection()->CloseConnection(
        QUIC_PEER_GOING_AWAY, "Server shutdown imminent",
        ConnectionCloseBehavior::SEND_CONNECTION_CLOSE_PACKET);
    QUICHE_DCHECK(reference_counted_session_map_.empty() ||
                  reference_counted_session_map_.begin()->second.get() !=
                      session);
  }
  DeleteSessions();
}
void QuicDispatcher::OnConnectionClosed(QuicConnectionId server_connection_id,
                                        QuicErrorCode error,
                                        const std::string& error_details,
                                        ConnectionCloseSource source) {
  auto it = reference_counted_session_map_.find(server_connection_id);
  if (it == reference_counted_session_map_.end()) {
    QUIC_BUG(quic_bug_10287_3) << "ConnectionId " << server_connection_id
                               << " does not exist in the session map.  Error: "
                               << QuicErrorCodeToString(error);
    QUIC_BUG(quic_bug_10287_4) << QuicStackTrace();
    return;
  }
  QUIC_DLOG_IF(INFO, error != QUIC_NO_ERROR)
      << "Closing connection (" << server_connection_id
      << ") due to error: " << QuicErrorCodeToString(error)
      << ", with details: " << error_details;
  const QuicSession* session = it->second.get();
  QuicConnection* connection = it->second->connection();
  if (closed_session_list_.empty()) {
    delete_sessions_alarm_->Update(helper()->GetClock()->ApproximateNow(),
                                   QuicTime::Delta::Zero());
  }
  closed_session_list_.push_back(std::move(it->second));
  CleanUpSession(it->first, connection, error, error_details, source);
  bool session_removed = false;
  for (const QuicConnectionId& cid :
       connection->GetActiveServerConnectionIds()) {
    auto it1 = reference_counted_session_map_.find(cid);
    if (it1 != reference_counted_session_map_.end()) {
      const QuicSession* session2 = it1->second.get();
      if (session2 == session || cid == server_connection_id) {
        reference_counted_session_map_.erase(it1);
        session_removed = true;
      } else {
        QUIC_BUG(quic_dispatcher_session_mismatch)
            << "Session is mismatched in the map. server_connection_id: "
            << server_connection_id << ". Current cid: " << cid
            << ". Cid of the other session "
            << (session2 == nullptr
                    ? "null"
                    : session2->connection()->connection_id().ToString());
      }
    } else {
      QUIC_BUG_IF(quic_dispatcher_session_not_found,
                  cid != connection->GetOriginalDestinationConnectionId())
          << "Missing session for cid " << cid
          << ". server_connection_id: " << server_connection_id;
    }
  }
  QUIC_BUG_IF(quic_session_is_not_removed, !session_removed);
  --num_sessions_in_session_map_;
}
void QuicDispatcher::OnWriteBlocked(
    QuicBlockedWriterInterface* blocked_writer) {
  write_blocked_list_.Add(*blocked_writer);
}
void QuicDispatcher::OnRstStreamReceived(const QuicRstStreamFrame& ) {}
void QuicDispatcher::OnStopSendingReceived(
    const QuicStopSendingFrame& ) {}
bool QuicDispatcher::TryAddNewConnectionId(
    const QuicConnectionId& server_connection_id,
    const QuicConnectionId& new_connection_id) {
  auto it = reference_counted_session_map_.find(server_connection_id);
  if (it == reference_counted_session_map_.end()) {
    QUIC_BUG(quic_bug_10287_7)
        << "Couldn't locate the session that issues the connection ID in "
           "reference_counted_session_map_.  server_connection_id:"
        << server_connection_id << " new_connection_id: " << new_connection_id;
    return false;
  }
  auto insertion_result = reference_counted_session_map_.insert(
      std::make_pair(new_connection_id, it->second));
  if (!insertion_result.second) {
    QUIC_CODE_COUNT(quic_cid_already_in_session_map);
  }
  return insertion_result.second;
}
void QuicDispatcher::OnConnectionIdRetired(
    const QuicConnectionId& server_connection_id) {
  reference_counted_session_map_.erase(server_connection_id);
}
void QuicDispatcher::OnConnectionAddedToTimeWaitList(
    QuicConnectionId server_connection_id) {
  QUIC_DLOG(INFO) << "Connection " << server_connection_id
                  << " added to time wait list.";
}
void QuicDispatcher::StatelesslyTerminateConnection(
    const QuicSocketAddress& self_address,
    const QuicSocketAddress& peer_address,
    QuicConnectionId server_connection_id, PacketHeaderFormat format,
    bool version_flag, bool use_length_prefix, ParsedQuicVersion version,
    QuicErrorCode error_code, const std::string& error_details,
    QuicTimeWaitListManager::TimeWaitAction action) {
  const BufferedPacketList* packet_list =
      buffered_packets_.GetPacketList(server_connection_id);
  if (packet_list == nullptr) {
    StatelesslyTerminateConnection(
        self_address, peer_address, server_connection_id, format, version_flag,
        use_length_prefix, version, error_code, error_details, action,
        std::nullopt,
        QuicPacketNumber());
    return;
  }
  QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets, 5, 8);
  StatelesslyTerminateConnection(
      self_address, peer_address, packet_list->original_connection_id, format,
      version_flag, use_length_prefix, version, error_code, error_details,
      action, packet_list->replaced_connection_id,
      packet_list->GetLastSentPacketNumber());
}
void QuicDispatcher::StatelesslyTerminateConnection(
    const QuicSocketAddress& self_address,
    const QuicSocketAddress& peer_address,
    QuicConnectionId server_connection_id, PacketHeaderFormat format,
    bool version_flag, bool use_length_prefix, ParsedQuicVersion version,
    QuicErrorCode error_code, const std::string& error_details,
    QuicTimeWaitListManager::TimeWaitAction action,
    const std::optional<QuicConnectionId>& replaced_connection_id,
    QuicPacketNumber last_sent_packet_number) {
  if (format != IETF_QUIC_LONG_HEADER_PACKET && !version_flag) {
    QUIC_DVLOG(1) << "Statelessly terminating " << server_connection_id
                  << " based on a non-ietf-long packet, action:" << action
                  << ", error_code:" << error_code
                  << ", error_details:" << error_details;
    time_wait_list_manager_->AddConnectionIdToTimeWait(
        action, TimeWaitConnectionInfo(format != GOOGLE_QUIC_PACKET, nullptr,
                                       {server_connection_id}));
    return;
  }
  if (IsSupportedVersion(version)) {
    QUIC_DVLOG(1)
        << "Statelessly terminating " << server_connection_id
        << " based on an ietf-long packet, which has a supported version:"
        << version << ", error_code:" << error_code
        << ", error_details:" << error_details << ", replaced_connection_id:"
        << (replaced_connection_id.has_value()
                ? replaced_connection_id->ToString()
                : "n/a");
    if (ack_buffered_initial_packets()) {
      QuicConnectionId original_connection_id = server_connection_id;
      if (last_sent_packet_number.IsInitialized()) {
        QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets,
                                  6, 8);
      }
      StatelessConnectionTerminator terminator(
          replaced_connection_id.value_or(original_connection_id),
          original_connection_id, version, last_sent_packet_number,
          helper_.get(), time_wait_list_manager_.get());
      std::vector<QuicConnectionId> active_connection_ids = {
          original_connection_id};
      if (replaced_connection_id.has_value()) {
        active_connection_ids.push_back(*replaced_connection_id);
      }
      terminator.CloseConnection(error_code, error_details,
                                 format != GOOGLE_QUIC_PACKET,
                                 std::move(active_connection_ids));
    } else {
      StatelessConnectionTerminator terminator(
          server_connection_id, server_connection_id, version,
          last_sent_packet_number, helper_.get(),
          time_wait_list_manager_.get());
      terminator.CloseConnection(
          error_code, error_details, format != GOOGLE_QUIC_PACKET,
          {server_connection_id});
    }
    QUIC_CODE_COUNT(quic_dispatcher_generated_connection_close);
    QuicSession::RecordConnectionCloseAtServer(
        error_code, ConnectionCloseSource::FROM_SELF);
    OnStatelessConnectionCloseGenerated(self_address, peer_address,
                                        server_connection_id, version,
                                        error_code, error_details);
    return;
  }
  QUIC_DVLOG(1)
      << "Statelessly terminating " << server_connection_id
      << " based on an ietf-long packet, which has an unsupported version:"
      << version << ", error_code:" << error_code
      << ", error_details:" << error_details;
  std::vector<std::unique_ptr<QuicEncryptedPacket>> termination_packets;
  termination_packets.push_back(QuicFramer::BuildVersionNegotiationPacket(
      server_connection_id, EmptyQuicConnectionId(),
      format != GOOGLE_QUIC_PACKET, use_length_prefix,
      {}));
  time_wait_list_manager()->AddConnectionIdToTimeWait(
      QuicTimeWaitListManager::SEND_TERMINATION_PACKETS,
      TimeWaitConnectionInfo(format != GOOGLE_QUIC_PACKET,
                             &termination_packets, {server_connection_id}));
}
bool QuicDispatcher::ShouldCreateSessionForUnknownVersion(
    const ReceivedPacketInfo& ) {
  return false;
}
void QuicDispatcher::OnExpiredPackets(
    QuicConnectionId server_connection_id,
    BufferedPacketList early_arrived_packets) {
  QUIC_CODE_COUNT(quic_reject_buffered_packets_expired);
  QuicErrorCode error_code = QUIC_HANDSHAKE_FAILED_PACKETS_BUFFERED_TOO_LONG;
  QuicSocketAddress self_address, peer_address;
  if (!early_arrived_packets.buffered_packets.empty()) {
    self_address = early_arrived_packets.buffered_packets.front().self_address;
    peer_address = early_arrived_packets.buffered_packets.front().peer_address;
  }
  if (ack_buffered_initial_packets()) {
    QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets, 7,
                              8);
    StatelesslyTerminateConnection(
        self_address, peer_address,
        early_arrived_packets.original_connection_id,
        early_arrived_packets.ietf_quic ? IETF_QUIC_LONG_HEADER_PACKET
                                        : GOOGLE_QUIC_PACKET,
        true,
        early_arrived_packets.version.HasLengthPrefixedConnectionIds(),
        early_arrived_packets.version, error_code,
        "Packets buffered for too long",
        quic::QuicTimeWaitListManager::SEND_STATELESS_RESET,
        early_arrived_packets.replaced_connection_id,
        early_arrived_packets.GetLastSentPacketNumber());
  } else {
    StatelesslyTerminateConnection(
        self_address, peer_address, server_connection_id,
        early_arrived_packets.ietf_quic ? IETF_QUIC_LONG_HEADER_PACKET
                                        : GOOGLE_QUIC_PACKET,
        true,
        early_arrived_packets.version.HasLengthPrefixedConnectionIds(),
        early_arrived_packets.version, error_code,
        "Packets buffered for too long",
        quic::QuicTimeWaitListManager::SEND_STATELESS_RESET);
  }
}
void QuicDispatcher::ProcessBufferedChlos(size_t max_connections_to_create) {
  new_sessions_allowed_per_event_loop_ = max_connections_to_create;
  for (; new_sessions_allowed_per_event_loop_ > 0;
       --new_sessions_allowed_per_event_loop_) {
    QuicConnectionId server_connection_id;
    BufferedPacketList packet_list =
        buffered_packets_.DeliverPacketsForNextConnection(
            &server_connection_id);
    const std::list<BufferedPacket>& packets = packet_list.buffered_packets;
    if (packets.empty()) {
      return;
    }
    if (!packet_list.parsed_chlo.has_value()) {
      QUIC_BUG(quic_dispatcher_no_parsed_chlo_in_buffered_packets)
          << "Buffered connection has no CHLO. connection_id:"
          << server_connection_id;
      continue;
    }
    auto session_ptr = CreateSessionFromChlo(
        server_connection_id, packet_list.replaced_connection_id,
        *packet_list.parsed_chlo, packet_list.version,
        packets.front().self_address, packets.front().peer_address,
        packet_list.tls_chlo_extractor.state(),
        packet_list.connection_id_generator,
        packet_list.dispatcher_sent_packets);
    if (session_ptr != nullptr) {
      DeliverPacketsToSession(packets, session_ptr.get());
    }
  }
}
bool QuicDispatcher::HasChlosBuffered() const {
  return buffered_packets_.HasChlosBuffered();
}
bool QuicDispatcher::HasBufferedPackets(QuicConnectionId server_connection_id) {
  return buffered_packets_.HasBufferedPackets(server_connection_id);
}
void QuicDispatcher::OnBufferPacketFailure(
    EnqueuePacketResult result, QuicConnectionId server_connection_id) {
  QUIC_DLOG(INFO) << "Fail to buffer packet on connection "
                  << server_connection_id << " because of " << result;
}
QuicTimeWaitListManager* QuicDispatcher::CreateQuicTimeWaitListManager() {
  return new QuicTimeWaitListManager(writer_.get(), this, helper_->GetClock(),
                                     alarm_factory_.get());
}
void QuicDispatcher::ProcessChlo(ParsedClientHello parsed_chlo,
                                 ReceivedPacketInfo* packet_info) {
  if (GetQuicFlag(quic_allow_chlo_buffering) &&
      new_sessions_allowed_per_event_loop_ <= 0) {
    QUIC_BUG_IF(quic_bug_12724_7, buffered_packets_.HasChloForConnection(
                                      packet_info->destination_connection_id));
    EnqueuePacketResult rs = buffered_packets_.EnqueuePacket(
        *packet_info, std::move(parsed_chlo), ConnectionIdGenerator());
    switch (rs) {
      case EnqueuePacketResult::SUCCESS:
        break;
      case EnqueuePacketResult::CID_COLLISION:
        buffered_packets_.DiscardPackets(
            packet_info->destination_connection_id);
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_PACKETS:
        ABSL_FALLTHROUGH_INTENDED;
      case EnqueuePacketResult::TOO_MANY_CONNECTIONS:
        OnBufferPacketFailure(rs, packet_info->destination_connection_id);
        break;
    }
    return;
  }
  BufferedPacketList packet_list =
      buffered_packets_.DeliverPackets(packet_info->destination_connection_id);
  QuicConnectionId original_connection_id =
      packet_list.buffered_packets.empty()
          ? packet_info->destination_connection_id
          : packet_list.original_connection_id;
  TlsChloExtractor::State chlo_extractor_state =
      packet_list.buffered_packets.empty()
          ? TlsChloExtractor::State::kParsedFullSinglePacketChlo
          : packet_list.tls_chlo_extractor.state();
  auto session_ptr = CreateSessionFromChlo(
      original_connection_id, packet_list.replaced_connection_id, parsed_chlo,
      packet_info->version, packet_info->self_address,
      packet_info->peer_address, chlo_extractor_state,
      packet_list.connection_id_generator, packet_list.dispatcher_sent_packets);
  if (session_ptr == nullptr) {
    QUICHE_DCHECK_EQ(packet_list.connection_id_generator, nullptr);
    return;
  }
  session_ptr->ProcessUdpPacket(packet_info->self_address,
                                packet_info->peer_address, packet_info->packet);
  DeliverPacketsToSession(packet_list.buffered_packets, session_ptr.get());
  --new_sessions_allowed_per_event_loop_;
}
void QuicDispatcher::SetLastError(QuicErrorCode error) { last_error_ = error; }
bool QuicDispatcher::OnFailedToDispatchPacket(
    const ReceivedPacketInfo& ) {
  return false;
}
const ParsedQuicVersionVector& QuicDispatcher::GetSupportedVersions() {
  return version_manager_->GetSupportedVersions();
}
void QuicDispatcher::DeliverPacketsToSession(
    const std::list<BufferedPacket>& packets, QuicSession* session) {
  for (const BufferedPacket& packet : packets) {
    session->ProcessUdpPacket(packet.self_address, packet.peer_address,
                              *(packet.packet));
  }
}
bool QuicDispatcher::IsSupportedVersion(const ParsedQuicVersion version) {
  for (const ParsedQuicVersion& supported_version :
       version_manager_->GetSupportedVersions()) {
    if (version == supported_version) {
      return true;
    }
  }
  return false;
}
bool QuicDispatcher::IsServerConnectionIdTooShort(
    QuicConnectionId connection_id) const {
  if (connection_id.length() >= kQuicMinimumInitialConnectionIdLength ||
      connection_id.length() >= expected_server_connection_id_length_) {
    return false;
  }
  uint8_t generator_output =
      connection_id.IsEmpty()
          ? connection_id_generator_.ConnectionIdLength(0x00)
          : connection_id_generator_.ConnectionIdLength(
                static_cast<uint8_t>(*connection_id.data()));
  return connection_id.length() < generator_output;
}
std::shared_ptr<QuicSession> QuicDispatcher::CreateSessionFromChlo(
    const QuicConnectionId original_connection_id,
    const std::optional<QuicConnectionId>& replaced_connection_id,
    const ParsedClientHello& parsed_chlo, const ParsedQuicVersion version,
    const QuicSocketAddress self_address, const QuicSocketAddress peer_address,
    TlsChloExtractor::State chlo_extractor_state,
    ConnectionIdGeneratorInterface* connection_id_generator,
    absl::Span<const DispatcherSentPacket> dispatcher_sent_packets) {
  bool should_generate_cid = false;
  if (connection_id_generator == nullptr) {
    should_generate_cid = true;
    connection_id_generator = &ConnectionIdGenerator();
  }
  std::optional<QuicConnectionId> server_connection_id;
  if (should_generate_cid) {
    server_connection_id = connection_id_generator->MaybeReplaceConnectionId(
        original_connection_id, version);
    if (server_connection_id.has_value() &&
        (server_connection_id->IsEmpty() ||
         *server_connection_id == original_connection_id)) {
      server_connection_id.reset();
    }
    QUIC_DVLOG(1) << "MaybeReplaceConnectionId(" << original_connection_id
                  << ") = "
                  << (server_connection_id.has_value()
                          ? server_connection_id->ToString()
                          : "nullopt");
    if (server_connection_id.has_value()) {
      switch (HandleConnectionIdCollision(
          original_connection_id, *server_connection_id, self_address,
          peer_address, version, &parsed_chlo)) {
        case VisitorInterface::HandleCidCollisionResult::kOk:
          break;
        case VisitorInterface::HandleCidCollisionResult::kCollision:
          return nullptr;
      }
    }
  } else {
    server_connection_id = replaced_connection_id;
  }
  const bool connection_id_replaced = server_connection_id.has_value();
  if (!connection_id_replaced) {
    server_connection_id = original_connection_id;
  }
  std::string alpn = SelectAlpn(parsed_chlo.alpns);
  std::unique_ptr<QuicSession> session =
      CreateQuicSession(*server_connection_id, self_address, peer_address, alpn,
                        version, parsed_chlo, *connection_id_generator);
  if (ABSL_PREDICT_FALSE(session == nullptr)) {
    QUIC_BUG(quic_bug_10287_8)
        << "CreateQuicSession returned nullptr for " << *server_connection_id
        << " from " << peer_address << " to " << self_address << " ALPN \""
        << alpn << "\" version " << version;
    return nullptr;
  }
  ++stats_.sessions_created;
  if (chlo_extractor_state ==
      TlsChloExtractor::State::kParsedFullMultiPacketChlo) {
    QUIC_CODE_COUNT(quic_connection_created_multi_packet_chlo);
    session->connection()->SetMultiPacketClientHello();
  } else {
    QUIC_CODE_COUNT(quic_connection_created_single_packet_chlo);
  }
  if (ack_buffered_initial_packets() && !dispatcher_sent_packets.empty()) {
    QUIC_RESTART_FLAG_COUNT_N(quic_dispatcher_ack_buffered_initial_packets, 8,
                              8);
    session->connection()->AddDispatcherSentPackets(dispatcher_sent_packets);
  }
  if (connection_id_replaced) {
    session->connection()->SetOriginalDestinationConnectionId(
        original_connection_id);
  }
  session->connection()->OnParsedClientHelloInfo(parsed_chlo);
  QUIC_DLOG(INFO) << "Created new session for " << *server_connection_id;
  auto insertion_result = reference_counted_session_map_.insert(std::make_pair(
      *server_connection_id, std::shared_ptr<QuicSession>(std::move(session))));
  std::shared_ptr<QuicSession> session_ptr = insertion_result.first->second;
  if (!insertion_result.second) {
    QUIC_BUG(quic_bug_10287_9)
        << "Tried to add a session to session_map with existing "
           "connection id: "
        << *server_connection_id;
  } else {
    ++num_sessions_in_session_map_;
    if (connection_id_replaced) {
      auto insertion_result2 = reference_counted_session_map_.insert(
          std::make_pair(original_connection_id, session_ptr));
      QUIC_BUG_IF(quic_460317833_02, !insertion_result2.second)
          << "Original connection ID already in session_map: "
          << original_connection_id;
    }
  }
  return session_ptr;
}
QuicDispatcher::HandleCidCollisionResult
QuicDispatcher::HandleConnectionIdCollision(
    const QuicConnectionId& original_connection_id,
    const QuicConnectionId& replaced_connection_id,
    const QuicSocketAddress& self_address,
    const QuicSocketAddress& peer_address, ParsedQuicVersion version,
    const ParsedClientHello* parsed_chlo) {
  HandleCidCollisionResult result = HandleCidCollisionResult::kOk;
  auto existing_session_iter =
      reference_counted_session_map_.find(replaced_connection_id);
  if (existing_session_iter != reference_counted_session_map_.end()) {
    result = HandleCidCollisionResult::kCollision;
    QUIC_CODE_COUNT(quic_connection_id_collision);
    QuicConnection* other_connection =
        existing_session_iter->second->connection();
    if (other_connection != nullptr) {  
      QUIC_LOG_EVERY_N_SEC(ERROR, 10)
          << "QUIC Connection ID collision. original_connection_id:"
          << original_connection_id
          << ", replaced_connection_id:" << replaced_connection_id
          << ", version:" << version << ", self_address:" << self_address
          << ", peer_address:" << peer_address << ", parsed_chlo:"
          << (parsed_chlo == nullptr ? "null" : parsed_chlo->ToString())
          << ", other peer address: " << other_connection->peer_address()
          << ", other CIDs: "
          << quiche::PrintElements(
                 other_connection->GetActiveServerConnectionIds())
          << ", other stats: " << other_connection->GetStats();
    }
  } else if (buffered_packets_.HasBufferedPackets(replaced_connection_id)) {
    result = HandleCidCollisionResult::kCollision;
    QUIC_CODE_COUNT(quic_connection_id_collision_with_buffered_session);
  }
  if (result == HandleCidCollisionResult::kOk) {
    return result;
  }
  const bool collide_with_active_session =
      existing_session_iter != reference_counted_session_map_.end();
  QUIC_DLOG(INFO) << "QUIC Connection ID collision with "
                  << (collide_with_active_session ? "active session"
                                                  : "buffered session")
                  << " for original_connection_id:" << original_connection_id
                  << ", replaced_connection_id:" << replaced_connection_id;
  StatelesslyTerminateConnection(
      self_address, peer_address, original_connection_id,
      IETF_QUIC_LONG_HEADER_PACKET,
      true, version.HasLengthPrefixedConnectionIds(), version,
      QUIC_HANDSHAKE_FAILED, "Connection ID collision, please retry",
      QuicTimeWaitListManager::SEND_CONNECTION_CLOSE_PACKETS);
  return result;
}
void QuicDispatcher::MaybeResetPacketsWithNoVersion(
    const ReceivedPacketInfo& packet_info) {
  QUICHE_DCHECK(!packet_info.version_flag);
  if (recent_stateless_reset_addresses_.contains(packet_info.peer_address)) {
    QUIC_CODE_COUNT(quic_donot_send_reset_repeatedly);
    return;
  }
  if (packet_info.form != GOOGLE_QUIC_PACKET) {
    if (packet_info.packet.length() <=
        QuicFramer::GetMinStatelessResetPacketLength()) {
      QUIC_CODE_COUNT(quic_drop_too_small_short_header_packets);
      return;
    }
  } else {
    const size_t MinValidPacketLength =
        kPacketHeaderTypeSize + expected_server_connection_id_length_ +
        PACKET_1BYTE_PACKET_NUMBER + 1 + 12;
    if (packet_info.packet.length() < MinValidPacketLength) {
      QUIC_CODE_COUNT(drop_too_small_packets);
      return;
    }
  }
  if (recent_stateless_reset_addresses_.size() >=
      GetQuicFlag(quic_max_recent_stateless_reset_addresses)) {
    QUIC_CODE_COUNT(quic_too_many_recent_reset_addresses);
    return;
  }
  if (recent_stateless_reset_addresses_.empty()) {
    clear_stateless_reset_addresses_alarm_->Update(
        helper()->GetClock()->ApproximateNow() +
            QuicTime::Delta::FromMilliseconds(
                GetQuicFlag(quic_recent_stateless_reset_addresses_lifetime_ms)),
        QuicTime::Delta::Zero());
  }
  recent_stateless_reset_addresses_.emplace(packet_info.peer_address);
  time_wait_list_manager()->SendPublicReset(
      packet_info.self_address, packet_info.peer_address,
      packet_info.destination_connection_id,
      packet_info.form != GOOGLE_QUIC_PACKET, packet_info.packet.length(),
      GetPerPacketContext());
}
void QuicDispatcher::MaybeSendVersionNegotiationPacket(
    const ReceivedPacketInfo& packet_info) {
  if (crypto_config()->validate_chlo_size() &&
      packet_info.packet.length() < kMinPacketSizeForVersionNegotiation) {
    return;
  }
  time_wait_list_manager()->SendVersionNegotiationPacket(
      packet_info.destination_connection_id, packet_info.source_connection_id,
      packet_info.form != GOOGLE_QUIC_PACKET, packet_info.use_length_prefix,
      GetSupportedVersions(), packet_info.self_address,
      packet_info.peer_address, GetPerPacketContext());
}
size_t QuicDispatcher::NumSessions() const {
  return num_sessions_in_session_map_;
}
}  