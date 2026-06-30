#include "quiche/quic/test_tools/crypto_test_utils.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "quiche/quic/core/crypto/certificate_view.h"
#include "quiche/quic/core/crypto/crypto_handshake.h"
#include "quiche/quic/core/crypto/crypto_utils.h"
#include "quiche/quic/core/crypto/proof_source_x509.h"
#include "quiche/quic/core/crypto/quic_crypto_server_config.h"
#include "quiche/quic/core/crypto/quic_decrypter.h"
#include "quiche/quic/core/crypto/quic_encrypter.h"
#include "quiche/quic/core/crypto/quic_random.h"
#include "quiche/quic/core/proto/crypto_server_config_proto.h"
#include "quiche/quic/core/quic_clock.h"
#include "quiche/quic/core/quic_connection.h"
#include "quiche/quic/core/quic_crypto_client_stream.h"
#include "quiche/quic/core/quic_crypto_server_stream_base.h"
#include "quiche/quic/core/quic_crypto_stream.h"
#include "quiche/quic/core/quic_packets.h"
#include "quiche/quic/core/quic_server_id.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/core/quic_utils.h"
#include "quiche/quic/core/quic_versions.h"
#include "quiche/quic/platform/api/quic_hostname_utils.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/quic/platform/api/quic_socket_address.h"
#include "quiche/quic/platform/api/quic_test.h"
#include "quiche/quic/test_tools/quic_connection_peer.h"
#include "quiche/quic/test_tools/quic_framer_peer.h"
#include "quiche/quic/test_tools/quic_stream_peer.h"
#include "quiche/quic/test_tools/quic_test_utils.h"
#include "quiche/quic/test_tools/simple_quic_framer.h"
#include "quiche/quic/test_tools/test_certificates.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/quiche_callbacks.h"
#include "quiche/common/test_tools/quiche_test_utils.h"
namespace quic {
namespace test {
namespace crypto_test_utils {
namespace {
using testing::_;
class CryptoFramerVisitor : public CryptoFramerVisitorInterface {
 public:
  CryptoFramerVisitor() : error_(false) {}
  void OnError(CryptoFramer* ) override { error_ = true; }
  void OnHandshakeMessage(const CryptoHandshakeMessage& message) override {
    messages_.push_back(message);
  }
  bool error() const { return error_; }
  const std::vector<CryptoHandshakeMessage>& messages() const {
    return messages_;
  }
 private:
  bool error_;
  std::vector<CryptoHandshakeMessage> messages_;
};
bool HexChar(char c, uint8_t* value) {
  if (c >= '0' && c <= '9') {
    *value = c - '0';
    return true;
  }
  if (c >= 'a' && c <= 'f') {
    *value = c - 'a' + 10;
    return true;
  }
  if (c >= 'A' && c <= 'F') {
    *value = c - 'A' + 10;
    return true;
  }
  return false;
}
void MovePackets(const QuicConnection& source_conn,
                 absl::Span<const QuicEncryptedPacket* const> packets,
                 QuicCryptoStream& dest_stream, QuicConnection& dest_conn,
                 Perspective dest_perspective, bool process_stream_data) {
  QUICHE_CHECK(!packets.empty());
  SimpleQuicFramer framer(source_conn.supported_versions(), dest_perspective);
  QuicFramerPeer::SetLastSerializedServerConnectionId(framer.framer(),
                                                      TestConnectionId());
  SimpleQuicFramer null_encryption_framer(source_conn.supported_versions(),
                                          dest_perspective);
  QuicFramerPeer::SetLastSerializedServerConnectionId(
      null_encryption_framer.framer(), TestConnectionId());
  for (const QuicEncryptedPacket* const packet : packets) {
    if (!dest_conn.connected()) {
      QUIC_LOG(INFO) << "Destination connection disconnected. Skipping packet.";
      continue;
    }
    QuicConnectionPeer::SwapCrypters(&dest_conn, framer.framer());
    QuicConnectionPeer::AddBytesReceived(&dest_conn, packet->length());
    if (!framer.ProcessPacket(*packet)) {
      QuicConnectionPeer::SwapCrypters(&dest_conn, framer.framer());
      continue;
    }
    QuicConnectionPeer::SwapCrypters(&dest_conn, framer.framer());
    QuicConnection::ScopedPacketFlusher flusher(&dest_conn);
    dest_conn.OnDecryptedPacket(packet->length(),
                                framer.last_decrypted_level());
    if (dest_stream.handshake_protocol() == PROTOCOL_TLS1_3) {
      QUIC_LOG(INFO) << "Attempting to decrypt with NullDecrypter: "
                        "expect a decryption failure on the next log line.";
      ASSERT_FALSE(null_encryption_framer.ProcessPacket(*packet))
          << "No TLS packets should be encrypted with the NullEncrypter";
    }
    dest_conn.OnDecryptedPacket(packet->length(),
                                framer.last_decrypted_level());
    QuicConnectionPeer::SetCurrentPacket(&dest_conn, packet->AsStringPiece());
    for (const auto& stream_frame : framer.stream_frames()) {
      if (process_stream_data &&
          dest_stream.handshake_protocol() == PROTOCOL_TLS1_3) {
        dest_conn.OnStreamFrame(*stream_frame);
      } else {
        if (stream_frame->stream_id == dest_stream.id()) {
          dest_stream.OnStreamFrame(*stream_frame);
        }
      }
    }
    for (const auto& crypto_frame : framer.crypto_frames()) {
      dest_stream.OnCryptoFrame(*crypto_frame);
    }
    if (!framer.connection_close_frames().empty() && dest_conn.connected()) {
      dest_conn.OnConnectionCloseFrame(framer.connection_close_frames()[0]);
    }
  }
  QuicConnectionPeer::SetCurrentPacket(&dest_conn,
                                       absl::string_view(nullptr, 0));
}
}  
FakeClientOptions::FakeClientOptions() {}
FakeClientOptions::~FakeClientOptions() {}
namespace {
class FullChloGenerator {
 public:
  FullChloGenerator(
      QuicCryptoServerConfig* crypto_config, QuicSocketAddress server_addr,
      QuicSocketAddress client_addr, const QuicClock* clock,
      ParsedQuicVersion version,
      quiche::QuicheReferenceCountedPointer<QuicSignedServerConfig>
          signed_config,
      QuicCompressedCertsCache* compressed_certs_cache,
      CryptoHandshakeMessage* out)
      : crypto_config_(crypto_config),
        server_addr_(server_addr),
        client_addr_(client_addr),
        clock_(clock),
        version_(version),
        signed_config_(signed_config),
        compressed_certs_cache_(compressed_certs_cache),
        out_(out),
        params_(new QuicCryptoNegotiatedParameters) {}
  class ValidateClientHelloCallback : public ValidateClientHelloResultCallback {
   public:
    explicit ValidateClientHelloCallback(FullChloGenerator* generator)
        : generator_(generator) {}
    void Run(quiche::QuicheReferenceCountedPointer<
                 ValidateClientHelloResultCallback::Result>
                 result,
             std::unique_ptr<ProofSource::Details> ) override {
      generator_->ValidateClientHelloDone(std::move(result));
    }
   private:
    FullChloGenerator* generator_;
  };
  std::unique_ptr<ValidateClientHelloCallback>
  GetValidateClientHelloCallback() {
    return std::make_unique<ValidateClientHelloCallback>(this);
  }
 private:
  void ValidateClientHelloDone(quiche::QuicheReferenceCountedPointer<
                               ValidateClientHelloResultCallback::Result>
                                   result) {
    result_ = result;
    crypto_config_->ProcessClientHello(
        result_, false, TestConnectionId(1), server_addr_,
        client_addr_, version_, {version_}, clock_, QuicRandom::GetInstance(),
        compressed_certs_cache_, params_, signed_config_,
        50, kDefaultMaxPacketSize,
        GetProcessClientHelloCallback());
  }
  class ProcessClientHelloCallback : public ProcessClientHelloResultCallback {
   public:
    explicit ProcessClientHelloCallback(FullChloGenerator* generator)
        : generator_(generator) {}
    void Run(QuicErrorCode error, const std::string& error_details,
             std::unique_ptr<CryptoHandshakeMessage> message,
             std::unique_ptr<DiversificationNonce> ,
             std::unique_ptr<ProofSource::Details> )
        override {
      ASSERT_TRUE(message) << QuicErrorCodeToString(error) << " "
                           << error_details;
      generator_->ProcessClientHelloDone(std::move(message));
    }
   private:
    FullChloGenerator* generator_;
  };
  std::unique_ptr<ProcessClientHelloCallback> GetProcessClientHelloCallback() {
    return std::make_unique<ProcessClientHelloCallback>(this);
  }
  void ProcessClientHelloDone(std::unique_ptr<CryptoHandshakeMessage> rej) {
    EXPECT_THAT(rej->tag(), testing::Eq(kREJ));
    QUIC_VLOG(1) << "Extract valid STK and SCID from\n" << rej->DebugString();
    absl::string_view srct;
    ASSERT_TRUE(rej->GetStringPiece(kSourceAddressTokenTag, &srct));
    absl::string_view scfg;
    ASSERT_TRUE(rej->GetStringPiece(kSCFG, &scfg));
    std::unique_ptr<CryptoHandshakeMessage> server_config(
        CryptoFramer::ParseMessage(scfg));
    absl::string_view scid;
    ASSERT_TRUE(server_config->GetStringPiece(kSCID, &scid));
    *out_ = result_->client_hello;
    out_->SetStringPiece(kSCID, scid);
    out_->SetStringPiece(kSourceAddressTokenTag, srct);
    uint64_t xlct = LeafCertHashForTesting();
    out_->SetValue(kXLCT, xlct);
  }
 protected:
  QuicCryptoServerConfig* crypto_config_;
  QuicSocketAddress server_addr_;
  QuicSocketAddress client_addr_;
  const QuicClock* clock_;
  ParsedQuicVersion version_;
  quiche::QuicheReferenceCountedPointer<QuicSignedServerConfig> signed_config_;
  QuicCompressedCertsCache* compressed_certs_cache_;
  CryptoHandshakeMessage* out_;
  quiche::QuicheReferenceCountedPointer<QuicCryptoNegotiatedParameters> params_;
  quiche::QuicheReferenceCountedPointer<
      ValidateClientHelloResultCallback::Result>
      result_;
};
}  
std::unique_ptr<QuicCryptoServerConfig> CryptoServerConfigForTesting() {
  return std::make_unique<QuicCryptoServerConfig>(
      QuicCryptoServerConfig::TESTING, QuicRandom::GetInstance(),
      ProofSourceForTesting(), KeyExchangeSource::Default());
}
int HandshakeWithFakeServer(QuicConfig* server_quic_config,
                            QuicCryptoServerConfig* crypto_config,
                            MockQuicConnectionHelper* helper,
                            MockAlarmFactory* alarm_factory,
                            PacketSavingConnection* client_conn,
                            QuicCryptoClientStreamBase* client,
                            std::string alpn) {
  auto* server_conn = new testing::NiceMock<PacketSavingConnection>(
      helper, alarm_factory, Perspective::IS_SERVER,
      ParsedVersionOfIndex(client_conn->supported_versions(), 0));
  QuicCompressedCertsCache compressed_certs_cache(
      QuicCompressedCertsCache::kQuicCompressedCertsCacheSize);
  SetupCryptoServerConfigForTest(
      server_conn->clock(), server_conn->random_generator(), crypto_config);
  TestQuicSpdyServerSession server_session(
      server_conn, *server_quic_config, client_conn->supported_versions(),
      crypto_config, &compressed_certs_cache);
  server_session.Initialize();
  server_session.GetMutableCryptoStream()
      ->SetServerApplicationStateForResumption(
          std::make_unique<ApplicationState>());
  EXPECT_CALL(*server_session.helper(),
              CanAcceptClientHello(testing::_, testing::_, testing::_,
                                   testing::_, testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(*server_conn, OnCanWrite()).Times(testing::AnyNumber());
  EXPECT_CALL(*client_conn, OnCanWrite()).Times(testing::AnyNumber());
  EXPECT_CALL(*server_conn, SendCryptoData(_, _, _))
      .Times(testing::AnyNumber());
  EXPECT_CALL(server_session, SelectAlpn(_))
      .WillRepeatedly([alpn](const std::vector<absl::string_view>& alpns) {
        return std::find(alpns.cbegin(), alpns.cend(), alpn);
      });
  QUICHE_CHECK_NE(0u, client_conn->encrypted_packets_.size());
  CommunicateHandshakeMessages(client_conn, client, server_conn,
                               server_session.GetMutableCryptoStream());
  if (client_conn->connected() && server_conn->connected()) {
    CompareClientAndServerKeys(client, server_session.GetMutableCryptoStream());
  }
  return client->num_sent_client_hellos();
}
int HandshakeWithFakeClient(MockQuicConnectionHelper* helper,
                            MockAlarmFactory* alarm_factory,
                            PacketSavingConnection* server_conn,
                            QuicCryptoServerStreamBase* server,
                            const QuicServerId& server_id,
                            const FakeClientOptions& options,
                            std::string alpn) {
  ParsedQuicVersionVector supported_versions =
      server_conn->supported_versions();
  if (options.only_tls_versions) {
    supported_versions.erase(
        std::remove_if(supported_versions.begin(), supported_versions.end(),
                       [](const ParsedQuicVersion& version) {
                         return version.handshake_protocol != PROTOCOL_TLS1_3;
                       }),
        supported_versions.end());
    QUICHE_CHECK(!options.only_quic_crypto_versions);
  } else if (options.only_quic_crypto_versions) {
    supported_versions.erase(
        std::remove_if(supported_versions.begin(), supported_versions.end(),
                       [](const ParsedQuicVersion& version) {
                         return version.handshake_protocol !=
                                PROTOCOL_QUIC_CRYPTO;
                       }),
        supported_versions.end());
  }
  PacketSavingConnection* client_conn = new PacketSavingConnection(
      helper, alarm_factory, Perspective::IS_CLIENT, supported_versions);
  client_conn->AdvanceTime(QuicTime::Delta::FromSeconds(1));
  QuicCryptoClientConfig crypto_config(ProofVerifierForTesting());
  TestQuicSpdyClientSession client_session(client_conn, DefaultQuicConfig(),
                                           supported_versions, server_id,
                                           &crypto_config);
  EXPECT_CALL(client_session, OnProofValid(testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(client_session, OnProofVerifyDetailsAvailable(testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(*client_conn, OnCanWrite()).Times(testing::AnyNumber());
  if (!alpn.empty()) {
    EXPECT_CALL(client_session, GetAlpnsToOffer())
        .WillRepeatedly(testing::Return(std::vector<std::string>({alpn})));
  } else {
    EXPECT_CALL(client_session, GetAlpnsToOffer())
        .WillRepeatedly(testing::Return(std::vector<std::string>(
            {AlpnForVersion(client_conn->version())})));
  }
  client_session.GetMutableCryptoStream()->CryptoConnect();
  QUICHE_CHECK_EQ(1u, client_conn->encrypted_packets_.size());
  CommunicateHandshakeMessages(client_conn,
                               client_session.GetMutableCryptoStream(),
                               server_conn, server);
  if (server->one_rtt_keys_available() && server->encryption_established()) {
    CompareClientAndServerKeys(client_session.GetMutableCryptoStream(), server);
  }
  return client_session.GetCryptoStream()->num_sent_client_hellos();
}
void SetupCryptoServerConfigForTest(const QuicClock* clock, QuicRandom* rand,
                                    QuicCryptoServerConfig* crypto_config) {
  QuicCryptoServerConfig::ConfigOptions options;
  options.channel_id_enabled = true;
  std::unique_ptr<CryptoHandshakeMessage> scfg =
      crypto_config->AddDefaultConfig(rand, clock, options);
}
void SendHandshakeMessageToStream(QuicCryptoStream* stream,
                                  const CryptoHandshakeMessage& message,
                                  Perspective ) {
  const QuicData& data = message.GetSerialized();
  QuicSession* session = QuicStreamPeer::session(stream);
  if (!QuicVersionUsesCryptoFrames(session->transport_version())) {
    QuicStreamFrame frame(
        QuicUtils::GetCryptoStreamId(session->transport_version()), false,
        stream->crypto_bytes_read(), data.AsStringPiece());
    stream->OnStreamFrame(frame);
  } else {
    EncryptionLevel level = session->connection()->last_decrypted_level();
    QuicCryptoFrame frame(level, stream->BytesReadOnLevel(level),
                          data.AsStringPiece());
    stream->OnCryptoFrame(frame);
  }
}
void CommunicateHandshakeMessages(PacketSavingConnection* client_conn,
                                  QuicCryptoStream* client,
                                  PacketSavingConnection* server_conn,
                                  QuicCryptoStream* server) {
  CommunicateHandshakeMessages(*client_conn, *client, *server_conn, *server,
                               *client_conn,
                               *server_conn);
}
void CommunicateHandshakeMessages(QuicConnection& client_conn,
                                  QuicCryptoStream& client,
                                  QuicConnection& server_conn,
                                  QuicCryptoStream& server,
                                  PacketProvider& packets_from_client,
                                  PacketProvider& packets_from_server) {
  while (
      client_conn.connected() && server_conn.connected() &&
      (!client.one_rtt_keys_available() || !server.one_rtt_keys_available())) {
    QUICHE_CHECK(!packets_from_client.GetPackets().empty());
    QUIC_LOG(INFO) << "Processing " << packets_from_client.GetPackets().size()
                   << " packets client->server";
    MovePackets(client_conn, packets_from_client.GetPackets(), server,
                server_conn, Perspective::IS_SERVER,
                false);
    packets_from_client.ClearPackets();
    if (client.one_rtt_keys_available() && server.one_rtt_keys_available() &&
        packets_from_server.GetPackets().empty()) {
      break;
    }
    QUIC_LOG(INFO) << "Processing " << packets_from_server.GetPackets().size()
                   << " packets server->client";
    MovePackets(server_conn, packets_from_server.GetPackets(), client,
                client_conn, Perspective::IS_CLIENT,
                false);
    packets_from_server.ClearPackets();
  }
}
bool CommunicateHandshakeMessagesUntil(
    PacketSavingConnection* client_conn, QuicCryptoStream* client,
    quiche::UnretainedCallback<bool()> client_condition,
    PacketSavingConnection* server_conn, QuicCryptoStream* server,
    quiche::UnretainedCallback<bool()> server_condition,
    bool process_stream_data) {
  return CommunicateHandshakeMessagesUntil(
      *client_conn, *client, client_condition, *server_conn, *server,
      server_condition, process_stream_data,
      *client_conn,
      *server_conn);
}
bool CommunicateHandshakeMessagesUntil(
    QuicConnection& client_conn, QuicCryptoStream& client,
    quiche::UnretainedCallback<bool()> client_condition,
    QuicConnection& server_conn, QuicCryptoStream& server,
    quiche::UnretainedCallback<bool()> server_condition,
    bool process_stream_data, PacketProvider& packets_from_client,
    PacketProvider& packets_from_server) {
  while (client_conn.connected() && server_conn.connected() &&
         (!client_condition() || !server_condition()) &&
         (!packets_from_client.GetPackets().empty() ||
          !packets_from_server.GetPackets().empty())) {
    if (!server_condition() && !packets_from_client.GetPackets().empty()) {
      QUIC_LOG(INFO) << "Processing " << packets_from_client.GetPackets().size()
                     << " packets client->server";
      MovePackets(client_conn, packets_from_client.GetPackets(), server,
                  server_conn, Perspective::IS_SERVER, process_stream_data);
      packets_from_client.ClearPackets();
    }
    if (!client_condition() && !packets_from_server.GetPackets().empty()) {
      QUIC_LOG(INFO) << "Processing " << packets_from_server.GetPackets().size()
                     << " packets server->client";
      MovePackets(server_conn, packets_from_server.GetPackets(), client,
                  client_conn, Perspective::IS_CLIENT, process_stream_data);
      packets_from_server.ClearPackets();
    }
  }
  bool result = client_condition() && server_condition();
  if (!result) {
    QUIC_LOG(INFO) << "CommunicateHandshakeMessagesUnti failed with state: "
                      "client connected? "
                   << client_conn.connected() << " server connected? "
                   << server_conn.connected() << " client condition met? "
                   << client_condition() << " server condition met? "
                   << server_condition();
  }
  return result;
}
std::pair<size_t, size_t> AdvanceHandshake(PacketSavingConnection* client_conn,
                                           QuicCryptoStream* client,
                                           size_t client_i,
                                           PacketSavingConnection* server_conn,
                                           QuicCryptoStream* server,
                                           size_t server_i) {
  std::vector<QuicEncryptedPacket*> client_packets;
  for (; client_i < client_conn->encrypted_packets_.size(); ++client_i) {
    client_packets.push_back(client_conn->encrypted_packets_[client_i].get());
  }
  AdvanceHandshake(client_packets, *client_conn, *client, {}, *server_conn,
                   *server);
  std::vector<QuicEncryptedPacket*> server_packets;
  for (; server_i < server_conn->encrypted_packets_.size(); ++server_i) {
    server_packets.push_back(server_conn->encrypted_packets_[server_i].get());
  }
  AdvanceHandshake({}, *client_conn, *client, server_packets, *server_conn,
                   *server);
  return std::make_pair(client_i, server_i);
}
void AdvanceHandshake(
    absl::Span<const QuicEncryptedPacket* const> packets_from_client,
    QuicConnection& client_conn, QuicCryptoStream& client,
    absl::Span<const QuicEncryptedPacket* const> packets_from_server,
    QuicConnection& server_conn, QuicCryptoStream& server) {
  if (!packets_from_client.empty()) {
    QUIC_LOG(INFO) << "Processing " << packets_from_client.size()
                   << " packets client->server";
    MovePackets(client_conn, packets_from_client, server, server_conn,
                Perspective::IS_SERVER, false);
  }
  if (!packets_from_server.empty()) {
    QUIC_LOG(INFO) << "Processing " << packets_from_server.size()
                   << " packets server->client";
    MovePackets(server_conn, packets_from_server, client, client_conn,
                Perspective::IS_CLIENT, false);
  }
}
std::string GetValueForTag(const CryptoHandshakeMessage& message, QuicTag tag) {
  auto it = message.tag_value_map().find(tag);
  if (it == message.tag_value_map().end()) {
    return std::string();
  }
  return it->second;
}
uint64_t LeafCertHashForTesting() {
  quiche::QuicheReferenceCountedPointer<ProofSource::Chain> chain;
  QuicSocketAddress server_address(QuicIpAddress::Any4(), 42);
  QuicSocketAddress client_address(QuicIpAddress::Any4(), 43);
  QuicCryptoProof proof;
  std::unique_ptr<ProofSource> proof_source(ProofSourceForTesting());
  class Callback : public ProofSource::Callback {
   public:
    Callback(bool* ok,
             quiche::QuicheReferenceCountedPointer<ProofSource::Chain>* chain)
        : ok_(ok), chain_(chain) {}
    void Run(
        bool ok,
        const quiche::QuicheReferenceCountedPointer<ProofSource::Chain>& chain,
        const QuicCryptoProof& ,
        std::unique_ptr<ProofSource::Details> ) override {
      *ok_ = ok;
      *chain_ = chain;
    }
   private:
    bool* ok_;
    quiche::QuicheReferenceCountedPointer<ProofSource::Chain>* chain_;
  };
  bool ok = false;
  proof_source->GetProof(
      server_address, client_address, "", "",
      AllSupportedVersionsWithQuicCrypto().front().transport_version, "",
      std::unique_ptr<ProofSource::Callback>(new Callback(&ok, &chain)));
  if (!ok || chain->certs.empty()) {
    QUICHE_DCHECK(false) << "Proof generation failed";
    return 0;
  }
  return QuicUtils::FNV1a_64_Hash(chain->certs[0]);
}
void FillInDummyReject(CryptoHandshakeMessage* rej) {
  rej->set_tag(kREJ);
  unsigned char scfg[] = {
    0x53, 0x43, 0x46, 0x47,
    0x01, 0x00,
    0x00, 0x00,
    0x45, 0x58, 0x50, 0x59,
    0x08, 0x00, 0x00, 0x00,
    '1',  '2',  '3',  '4',
    '5',  '6',  '7',  '8'
  };
  rej->SetValue(kSCFG, scfg);
  rej->SetStringPiece(kServerNonceTag, "SERVER_NONCE");
  int64_t ttl = 2 * 24 * 60 * 60;
  rej->SetValue(kSTTL, ttl);
  std::vector<QuicTag> reject_reasons;
  reject_reasons.push_back(CLIENT_NONCE_INVALID_FAILURE);
  rej->SetVector(kRREJ, reject_reasons);
}
namespace {
#define RETURN_STRING_LITERAL(x) \
  case x:                        \
    return #x
std::string EncryptionLevelString(EncryptionLevel level) {
  switch (level) {
    RETURN_STRING_LITERAL(ENCRYPTION_INITIAL);
    RETURN_STRING_LITERAL(ENCRYPTION_HANDSHAKE);
    RETURN_STRING_LITERAL(ENCRYPTION_ZERO_RTT);
    RETURN_STRING_LITERAL(ENCRYPTION_FORWARD_SECURE);
    default:
      return "";
  }
}
void CompareCrypters(const QuicEncrypter* encrypter,
                     const QuicDecrypter* decrypter, std::string label) {
  if (encrypter == nullptr || decrypter == nullptr) {
    ADD_FAILURE() << "Expected non-null crypters; have " << encrypter << " and "
                  << decrypter << " for " << label;
    return;
  }
  absl::string_view encrypter_key = encrypter->GetKey();
  absl::string_view encrypter_iv = encrypter->GetNoncePrefix();
  absl::string_view decrypter_key = decrypter->GetKey();
  absl::string_view decrypter_iv = decrypter->GetNoncePrefix();
  quiche::test::CompareCharArraysWithHexError(
      label + " key", encrypter_key.data(), encrypter_key.length(),
      decrypter_key.data(), decrypter_key.length());
  quiche::test::CompareCharArraysWithHexError(
      label + " iv", encrypter_iv.data(), encrypter_iv.length(),
      decrypter_iv.data(), decrypter_iv.length());
}
}  
void CompareClientAndServerKeys(QuicCryptoClientStreamBase* client,
                                QuicCryptoServerStreamBase* server) {
  QuicFramer* client_framer = QuicConnectionPeer::GetFramer(
      QuicStreamPeer::session(client)->connection());
  QuicFramer* server_framer = QuicConnectionPeer::GetFramer(
      QuicStreamPeer::session(server)->connection());
  for (EncryptionLevel level :
       {ENCRYPTION_HANDSHAKE, ENCRYPTION_ZERO_RTT, ENCRYPTION_FORWARD_SECURE}) {
    SCOPED_TRACE(EncryptionLevelString(level));
    const QuicEncrypter* client_encrypter(
        QuicFramerPeer::GetEncrypter(client_framer, level));
    const QuicDecrypter* server_decrypter(
        QuicFramerPeer::GetDecrypter(server_framer, level));
    if (level == ENCRYPTION_FORWARD_SECURE ||
        !((level == ENCRYPTION_HANDSHAKE || level == ENCRYPTION_ZERO_RTT ||
           client_encrypter == nullptr) &&
          (level == ENCRYPTION_ZERO_RTT || server_decrypter == nullptr))) {
      CompareCrypters(client_encrypter, server_decrypter,
                      "client " + EncryptionLevelString(level) + " write");
    }
    const QuicEncrypter* server_encrypter(
        QuicFramerPeer::GetEncrypter(server_framer, level));
    const QuicDecrypter* client_decrypter(
        QuicFramerPeer::GetDecrypter(client_framer, level));
    if (level == ENCRYPTION_FORWARD_SECURE ||
        !(server_encrypter == nullptr &&
          (level == ENCRYPTION_HANDSHAKE || level == ENCRYPTION_ZERO_RTT ||
           client_decrypter == nullptr))) {
      CompareCrypters(server_encrypter, client_decrypter,
                      "server " + EncryptionLevelString(level) + " write");
    }
  }
  absl::string_view client_subkey_secret =
      client->crypto_negotiated_params().subkey_secret;
  absl::string_view server_subkey_secret =
      server->crypto_negotiated_params().subkey_secret;
  quiche::test::CompareCharArraysWithHexError(
      "subkey secret", client_subkey_secret.data(),
      client_subkey_secret.length(), server_subkey_secret.data(),
      server_subkey_secret.length());
}
QuicTag ParseTag(const char* tagstr) {
  const size_t len = strlen(tagstr);
  QUICHE_CHECK_NE(0u, len);
  QuicTag tag = 0;
  if (tagstr[0] == '#') {
    QUICHE_CHECK_EQ(static_cast<size_t>(1 + 2 * 4), len);
    tagstr++;
    for (size_t i = 0; i < 8; i++) {
      tag <<= 4;
      uint8_t v = 0;
      QUICHE_CHECK(HexChar(tagstr[i], &v));
      tag |= v;
    }
    return tag;
  }
  QUICHE_CHECK_LE(len, 4u);
  for (size_t i = 0; i < 4; i++) {
    tag >>= 8;
    if (i < len) {
      tag |= static_cast<uint32_t>(tagstr[i]) << 24;
    }
  }
  return tag;
}
CryptoHandshakeMessage CreateCHLO(
    std::vector<std::pair<std::string, std::string>> tags_and_values) {
  return CreateCHLO(tags_and_values, -1);
}
CryptoHandshakeMessage CreateCHLO(
    std::vector<std::pair<std::string, std::string>> tags_and_values,
    int minimum_size_bytes) {
  CryptoHandshakeMessage msg;
  msg.set_tag(MakeQuicTag('C', 'H', 'L', 'O'));
  if (minimum_size_bytes > 0) {
    msg.set_minimum_size(minimum_size_bytes);
  }
  for (const auto& tag_and_value : tags_and_values) {
    const std::string& tag = tag_and_value.first;
    const std::string& value = tag_and_value.second;
    const QuicTag quic_tag = ParseTag(tag.c_str());
    size_t value_len = value.length();
    if (value_len > 0 && value[0] == '#') {
      std::string hex_value =
          absl::HexStringToBytes(absl::string_view(&value[1]));
      msg.SetStringPiece(quic_tag, hex_value);
      continue;
    }
    msg.SetStringPiece(quic_tag, value);
  }
  std::unique_ptr<QuicData> bytes =
      CryptoFramer::ConstructHandshakeMessage(msg);
  std::unique_ptr<CryptoHandshakeMessage> parsed(
      CryptoFramer::ParseMessage(bytes->AsStringPiece()));
  QUICHE_CHECK(parsed);
  return *parsed;
}
CryptoHandshakeMessage GenerateDefaultInchoateCHLO(
    const QuicClock* clock, QuicTransportVersion version,
    QuicCryptoServerConfig* crypto_config) {
  return CreateCHLO(
      {{"PDMD", "X509"},
       {"AEAD", "AESG"},
       {"KEXS", "C255"},
       {"PUBS", GenerateClientPublicValuesHex().c_str()},
       {"NONC", GenerateClientNonceHex(clock, crypto_config).c_str()},
       {"VER\0", QuicVersionLabelToString(
           CreateQuicVersionLabel(
            ParsedQuicVersion(PROTOCOL_QUIC_CRYPTO, version))).c_str()}},
      kClientHelloMinimumSize);
}
std::string GenerateClientNonceHex(const QuicClock* clock,
                                   QuicCryptoServerConfig* crypto_config) {
  QuicCryptoServerConfig::ConfigOptions old_config_options;
  QuicCryptoServerConfig::ConfigOptions new_config_options;
  old_config_options.id = "old-config-id";
  crypto_config->AddDefaultConfig(QuicRandom::GetInstance(), clock,
                                  old_config_options);
  QuicServerConfigProtobuf primary_config = crypto_config->GenerateConfig(
      QuicRandom::GetInstance(), clock, new_config_options);
  primary_config.set_primary_time(clock->WallNow().ToUNIXSeconds());
  std::unique_ptr<CryptoHandshakeMessage> msg =
      crypto_config->AddConfig(primary_config, clock->WallNow());
  absl::string_view orbit;
  QUICHE_CHECK(msg->GetStringPiece(kORBT, &orbit));
  std::string nonce;
  CryptoUtils::GenerateNonce(clock->WallNow(), QuicRandom::GetInstance(), orbit,
                             &nonce);
  return ("#" + absl::BytesToHexString(nonce));
}
std::string GenerateClientPublicValuesHex() {
  char public_value[32];
  memset(public_value, 42, sizeof(public_value));
  return ("#" + absl::BytesToHexString(
                    absl::string_view(public_value, sizeof(public_value))));
}
void GenerateFullCHLO(
    const CryptoHandshakeMessage& inchoate_chlo,
    QuicCryptoServerConfig* crypto_config, QuicSocketAddress server_addr,
    QuicSocketAddress client_addr, QuicTransportVersion transport_version,
    const QuicClock* clock,
    quiche::QuicheReferenceCountedPointer<QuicSignedServerConfig> signed_config,
    QuicCompressedCertsCache* compressed_certs_cache,
    CryptoHandshakeMessage* out) {
  FullChloGenerator generator(
      crypto_config, server_addr, client_addr, clock,
      ParsedQuicVersion(PROTOCOL_QUIC_CRYPTO, transport_version), signed_config,
      compressed_certs_cache, out);
  crypto_config->ValidateClientHello(
      inchoate_chlo, client_addr, server_addr, transport_version, clock,
      signed_config, generator.GetValidateClientHelloCallback());
}
namespace {
constexpr char kTestProofHostname[] = "test.example.com";
class TestProofSource : public ProofSourceX509 {
 public:
  TestProofSource()
      : ProofSourceX509(
            quiche::QuicheReferenceCountedPointer<ProofSource::Chain>(
                new ProofSource::Chain(
                    std::vector<std::string>{std::string(kTestCertificate)})),
            std::move(*CertificatePrivateKey::LoadFromDer(
                kTestCertificatePrivateKey))) {
    QUICHE_DCHECK(valid());
  }
 protected:
  void MaybeAddSctsForHostname(absl::string_view ,
                               std::string& leaf_cert_scts) override {
    leaf_cert_scts = "Certificate Transparency is really nice";
  }
};
class TestProofVerifier : public ProofVerifier {
 public:
  TestProofVerifier()
      : certificate_(std::move(
            *CertificateView::ParseSingleCertificate(kTestCertificate))) {}
  class Details : public ProofVerifyDetails {
   public:
    ProofVerifyDetails* Clone() const override { return new Details(*this); }
  };
  QuicAsyncStatus VerifyProof(
      const std::string& hostname, const uint16_t port,
      const std::string& server_config,
      QuicTransportVersion , absl::string_view chlo_hash,
      const std::vector<std::string>& certs, const std::string& cert_sct,
      const std::string& signature, const ProofVerifyContext* context,
      std::string* error_details, std::unique_ptr<ProofVerifyDetails>* details,
      std::unique_ptr<ProofVerifierCallback> callback) override {
    std::optional<std::string> payload =
        CryptoUtils::GenerateProofPayloadToBeSigned(chlo_hash, server_config);
    if (!payload.has_value()) {
      *error_details = "Failed to serialize signed payload";
      return QUIC_FAILURE;
    }
    if (!certificate_.VerifySignature(*payload, signature,
                                      SSL_SIGN_RSA_PSS_RSAE_SHA256)) {
      *error_details = "Invalid signature";
      return QUIC_FAILURE;
    }
    uint8_t out_alert;
    return VerifyCertChain(hostname, port, certs, "",
                           cert_sct, context, error_details, details,
                           &out_alert, std::move(callback));
  }
  QuicAsyncStatus VerifyCertChain(
      const std::string& hostname, const uint16_t ,
      const std::vector<std::string>& certs,
      const std::string& , const std::string& ,
      const ProofVerifyContext* , std::string* error_details,
      std::unique_ptr<ProofVerifyDetails>* details, uint8_t* ,
      std::unique_ptr<ProofVerifierCallback> ) override {
    std::string normalized_hostname =
        QuicHostnameUtils::NormalizeHostname(hostname);
    if (normalized_hostname != kTestProofHostname) {
      *error_details = absl::StrCat("Invalid hostname, expected ",
                                    kTestProofHostname, " got ", hostname);
      return QUIC_FAILURE;
    }
    if (certs.empty() || certs.front() != kTestCertificate) {
      *error_details = "Received certificate different from the expected";
      return QUIC_FAILURE;
    }
    *details = std::make_unique<Details>();
    return QUIC_SUCCESS;
  }
  std::unique_ptr<ProofVerifyContext> CreateDefaultContext() override {
    return nullptr;
  }
 private:
  CertificateView certificate_;
};
}  
std::unique_ptr<ProofSource> ProofSourceForTesting() {
  return std::make_unique<TestProofSource>();
}
std::unique_ptr<ProofVerifier> ProofVerifierForTesting() {
  return std::make_unique<TestProofVerifier>();
}
std::string CertificateHostnameForTesting() { return kTestProofHostname; }
std::unique_ptr<ProofVerifyContext> ProofVerifyContextForTesting() {
  return nullptr;
}
}  
}  
}  