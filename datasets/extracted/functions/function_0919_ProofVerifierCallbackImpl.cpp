#include "quiche/quic/core/quic_crypto_client_handshaker.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/strings/str_cat.h"
#include "quiche/quic/core/crypto/crypto_protocol.h"
#include "quiche/quic/core/crypto/crypto_utils.h"
#include "quiche/quic/core/quic_session.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/platform/api/quic_client_stats.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace quic {
QuicCryptoClientHandshaker::ProofVerifierCallbackImpl::
    ProofVerifierCallbackImpl(QuicCryptoClientHandshaker* parent)
    : parent_(parent) {}
QuicCryptoClientHandshaker::ProofVerifierCallbackImpl::
    ~ProofVerifierCallbackImpl() {}
void QuicCryptoClientHandshaker::ProofVerifierCallbackImpl::Run(
    bool ok, const std::string& error_details,
    std::unique_ptr<ProofVerifyDetails>* details) {
  if (parent_ == nullptr) {
    return;
  }
  parent_->verify_ok_ = ok;
  parent_->verify_error_details_ = error_details;
  parent_->verify_details_ = std::move(*details);
  parent_->proof_verify_callback_ = nullptr;
  parent_->DoHandshakeLoop(nullptr);
}
void QuicCryptoClientHandshaker::ProofVerifierCallbackImpl::Cancel() {
  parent_ = nullptr;
}
QuicCryptoClientHandshaker::QuicCryptoClientHandshaker(
    const QuicServerId& server_id, QuicCryptoClientStream* stream,
    QuicSession* session, std::unique_ptr<ProofVerifyContext> verify_context,
    QuicCryptoClientConfig* crypto_config,
    QuicCryptoClientStream::ProofHandler* proof_handler)
    : QuicCryptoHandshaker(stream, session),
      stream_(stream),
      session_(session),
      delegate_(session),
      next_state_(STATE_IDLE),
      num_client_hellos_(0),
      crypto_config_(crypto_config),
      server_id_(server_id),
      generation_counter_(0),
      verify_context_(std::move(verify_context)),
      proof_verify_callback_(nullptr),
      proof_handler_(proof_handler),
      verify_ok_(false),
      proof_verify_start_time_(QuicTime::Zero()),
      num_scup_messages_received_(0),
      encryption_established_(false),
      one_rtt_keys_available_(false),
      crypto_negotiated_params_(new QuicCryptoNegotiatedParameters) {}
QuicCryptoClientHandshaker::~QuicCryptoClientHandshaker() {
  if (proof_verify_callback_) {
    proof_verify_callback_->Cancel();
  }
}
void QuicCryptoClientHandshaker::OnHandshakeMessage(
    const CryptoHandshakeMessage& message) {
  QuicCryptoHandshaker::OnHandshakeMessage(message);
  if (message.tag() == kSCUP) {
    if (!one_rtt_keys_available()) {
      stream_->OnUnrecoverableError(
          QUIC_CRYPTO_UPDATE_BEFORE_HANDSHAKE_COMPLETE,
          "Early SCUP disallowed");
      return;
    }
    HandleServerConfigUpdateMessage(message);
    num_scup_messages_received_++;
    return;
  }
  if (one_rtt_keys_available()) {
    stream_->OnUnrecoverableError(QUIC_CRYPTO_MESSAGE_AFTER_HANDSHAKE_COMPLETE,
                                  "Unexpected handshake message");
    return;
  }
  DoHandshakeLoop(&message);
}
bool QuicCryptoClientHandshaker::CryptoConnect() {
  next_state_ = STATE_INITIALIZE;
  DoHandshakeLoop(nullptr);
  return session()->connection()->connected();
}
int QuicCryptoClientHandshaker::num_sent_client_hellos() const {
  return num_client_hellos_;
}
bool QuicCryptoClientHandshaker::ResumptionAttempted() const {
  QUICHE_DCHECK(false);
  return false;
}
bool QuicCryptoClientHandshaker::IsResumption() const {
  QUIC_BUG_IF(quic_bug_12522_1, !one_rtt_keys_available_);
  return false;
}
bool QuicCryptoClientHandshaker::EarlyDataAccepted() const {
  QUIC_BUG_IF(quic_bug_12522_2, !one_rtt_keys_available_);
  return num_client_hellos_ == 1;
}
ssl_early_data_reason_t QuicCryptoClientHandshaker::EarlyDataReason() const {
  return early_data_reason_;
}
bool QuicCryptoClientHandshaker::ReceivedInchoateReject() const {
  QUIC_BUG_IF(quic_bug_12522_3, !one_rtt_keys_available_);
  return num_client_hellos_ >= 3;
}
int QuicCryptoClientHandshaker::num_scup_messages_received() const {
  return num_scup_messages_received_;
}
std::string QuicCryptoClientHandshaker::chlo_hash() const { return chlo_hash_; }
bool QuicCryptoClientHandshaker::encryption_established() const {
  return encryption_established_;
}
bool QuicCryptoClientHandshaker::IsCryptoFrameExpectedForEncryptionLevel(
    EncryptionLevel ) const {
  return true;
}
EncryptionLevel
QuicCryptoClientHandshaker::GetEncryptionLevelToSendCryptoDataOfSpace(
    PacketNumberSpace space) const {
  if (space == INITIAL_DATA) {
    return ENCRYPTION_INITIAL;
  }
  QUICHE_DCHECK(false);
  return NUM_ENCRYPTION_LEVELS;
}
bool QuicCryptoClientHandshaker::one_rtt_keys_available() const {
  return one_rtt_keys_available_;
}
const QuicCryptoNegotiatedParameters&
QuicCryptoClientHandshaker::crypto_negotiated_params() const {
  return *crypto_negotiated_params_;
}
CryptoMessageParser* QuicCryptoClientHandshaker::crypto_message_parser() {
  return QuicCryptoHandshaker::crypto_message_parser();
}
HandshakeState QuicCryptoClientHandshaker::GetHandshakeState() const {
  return one_rtt_keys_available() ? HANDSHAKE_COMPLETE : HANDSHAKE_START;
}
void QuicCryptoClientHandshaker::OnHandshakeDoneReceived() {
  QUICHE_DCHECK(false);
}
void QuicCryptoClientHandshaker::OnNewTokenReceived(
    absl::string_view ) {
  QUICHE_DCHECK(false);
}
size_t QuicCryptoClientHandshaker::BufferSizeLimitForLevel(
    EncryptionLevel level) const {
  return QuicCryptoHandshaker::BufferSizeLimitForLevel(level);
}
std::unique_ptr<QuicDecrypter>
QuicCryptoClientHandshaker::AdvanceKeysAndCreateCurrentOneRttDecrypter() {
  QUICHE_DCHECK(false);
  return nullptr;
}
std::unique_ptr<QuicEncrypter>
QuicCryptoClientHandshaker::CreateCurrentOneRttEncrypter() {
  QUICHE_DCHECK(false);
  return nullptr;
}
void QuicCryptoClientHandshaker::OnConnectionClosed(
    QuicErrorCode , ConnectionCloseSource ) {
  next_state_ = STATE_CONNECTION_CLOSED;
}
void QuicCryptoClientHandshaker::HandleServerConfigUpdateMessage(
    const CryptoHandshakeMessage& server_config_update) {
  QUICHE_DCHECK(server_config_update.tag() == kSCUP);
  std::string error_details;
  QuicCryptoClientConfig::CachedState* cached =
      crypto_config_->LookupOrCreate(server_id_);
  QuicErrorCode error = crypto_config_->ProcessServerConfigUpdate(
      server_config_update, session()->connection()->clock()->WallNow(),
      session()->transport_version(), chlo_hash_, cached,
      crypto_negotiated_params_, &error_details);
  if (error != QUIC_NO_ERROR) {
    stream_->OnUnrecoverableError(
        error, "Server config update invalid: " + error_details);
    return;
  }
  QUICHE_DCHECK(one_rtt_keys_available());
  if (proof_verify_callback_) {
    proof_verify_callback_->Cancel();
  }
  next_state_ = STATE_INITIALIZE_SCUP;
  DoHandshakeLoop(nullptr);
}
void QuicCryptoClientHandshaker::DoHandshakeLoop(
    const CryptoHandshakeMessage* in) {
  QuicCryptoClientConfig::CachedState* cached =
      crypto_config_->LookupOrCreate(server_id_);
  QuicAsyncStatus rv = QUIC_SUCCESS;
  do {
    QUICHE_CHECK_NE(STATE_NONE, next_state_);
    const State state = next_state_;
    next_state_ = STATE_IDLE;
    rv = QUIC_SUCCESS;
    switch (state) {
      case STATE_INITIALIZE:
        DoInitialize(cached);
        break;
      case STATE_SEND_CHLO:
        DoSendCHLO(cached);
        return;  
      case STATE_RECV_REJ:
        DoReceiveREJ(in, cached);
        break;
      case STATE_VERIFY_PROOF:
        rv = DoVerifyProof(cached);
        break;
      case STATE_VERIFY_PROOF_COMPLETE:
        DoVerifyProofComplete(cached);
        break;
      case STATE_RECV_SHLO:
        DoReceiveSHLO(in, cached);
        break;
      case STATE_IDLE:
        stream_->OnUnrecoverableError(QUIC_INVALID_CRYPTO_MESSAGE_TYPE,
                                      "Handshake in idle state");
        return;
      case STATE_INITIALIZE_SCUP:
        DoInitializeServerConfigUpdate(cached);
        break;
      case STATE_NONE:
        QUICHE_NOTREACHED();
        return;
      case STATE_CONNECTION_CLOSED:
        rv = QUIC_FAILURE;
        return;  
    }
  } while (rv != QUIC_PENDING && next_state_ != STATE_NONE);
}
void QuicCryptoClientHandshaker::DoInitialize(
    QuicCryptoClientConfig::CachedState* cached) {
  if (!cached->IsEmpty() && !cached->signature().empty()) {
    QUICHE_DCHECK(crypto_config_->proof_verifier());
    proof_verify_start_time_ = session()->connection()->clock()->Now();
    chlo_hash_ = cached->chlo_hash();
    next_state_ = STATE_VERIFY_PROOF;
  } else {
    next_state_ = STATE_SEND_CHLO;
  }
}
void QuicCryptoClientHandshaker::DoSendCHLO(
    QuicCryptoClientConfig::CachedState* cached) {
  session()->connection()->SetDefaultEncryptionLevel(ENCRYPTION_INITIAL);
  encryption_established_ = false;
  if (num_client_hellos_ >= QuicCryptoClientStream::kMaxClientHellos) {
    stream_->OnUnrecoverableError(
        QUIC_CRYPTO_TOO_MANY_REJECTS,
        absl::StrCat("More than ", QuicCryptoClientStream::kMaxClientHellos,
                     " rejects"));
    return;
  }
  num_client_hellos_++;
  CryptoHandshakeMessage out;
  QUICHE_DCHECK(session() != nullptr);
  QUICHE_DCHECK(session()->config() != nullptr);
  session()->config()->ToHandshakeMessage(&out, session()->transport_version());
  bool fill_inchoate_client_hello = false;
  if (!cached->IsComplete(session()->connection()->clock()->WallNow())) {
    early_data_reason_ = ssl_early_data_no_session_offered;
    fill_inchoate_client_hello = true;
  } else if (session()->config()->HasClientRequestedIndependentOption(
                 kQNZ2, session()->perspective()) &&
             num_client_hellos_ == 1) {
    early_data_reason_ = ssl_early_data_disabled;
    fill_inchoate_client_hello = true;
  }
  if (fill_inchoate_client_hello) {
    crypto_config_->FillInchoateClientHello(
        server_id_, session()->supported_versions().front(), cached,
        session()->connection()->random_generator(),
         true, crypto_negotiated_params_, &out);
    const QuicByteCount kFramingOverhead = 50;  
    const QuicByteCount max_packet_size =
        session()->connection()->max_packet_length();
    if (max_packet_size <= kFramingOverhead) {
      QUIC_DLOG(DFATAL) << "max_packet_length (" << max_packet_size
                        << ") has no room for framing overhead.";
      stream_->OnUnrecoverableError(QUIC_INTERNAL_ERROR,
                                    "max_packet_size too smalll");
      return;
    }
    if (kClientHelloMinimumSize > max_packet_size - kFramingOverhead) {
      QUIC_DLOG(DFATAL) << "Client hello won't fit in a single packet.";
      stream_->OnUnrecoverableError(QUIC_INTERNAL_ERROR, "CHLO too large");
      return;
    }
    next_state_ = STATE_RECV_REJ;
    chlo_hash_ = CryptoUtils::HashHandshakeMessage(out, Perspective::IS_CLIENT);
    session()->connection()->set_fully_pad_crypto_handshake_packets(
        crypto_config_->pad_inchoate_hello());
    SendHandshakeMessage(out, ENCRYPTION_INITIAL);
    return;
  }
  std::string error_details;
  QuicErrorCode error = crypto_config_->FillClientHello(
      server_id_, session()->connection()->connection_id(),
      session()->supported_versions().front(),
      session()->connection()->version(), cached,
      session()->connection()->clock()->WallNow(),
      session()->connection()->random_generator(), crypto_negotiated_params_,
      &out, &error_details);
  if (error != QUIC_NO_ERROR) {
    cached->InvalidateServerConfig();
    stream_->OnUnrecoverableError(error, error_details);
    return;
  }
  chlo_hash_ = CryptoUtils::HashHandshakeMessage(out, Perspective::IS_CLIENT);
  if (cached->proof_verify_details()) {
    proof_handler_->OnProofVerifyDetailsAvailable(
        *cached->proof_verify_details());
  }
  next_state_ = STATE_RECV_SHLO;
  session()->connection()->set_fully_pad_crypto_handshake_packets(
      crypto_config_->pad_full_hello());
  SendHandshakeMessage(out, ENCRYPTION_INITIAL);
  delegate_->OnNewEncryptionKeyAvailable(
      ENCRYPTION_ZERO_RTT,
      std::move(crypto_negotiated_params_->initial_crypters.encrypter));
  delegate_->OnNewDecryptionKeyAvailable(
      ENCRYPTION_ZERO_RTT,
      std::move(crypto_negotiated_params_->initial_crypters.decrypter),
      true,
      true);
  encryption_established_ = true;
  delegate_->SetDefaultEncryptionLevel(ENCRYPTION_ZERO_RTT);
  if (early_data_reason_ == ssl_early_data_unknown && num_client_hellos_ > 1) {
    early_data_reason_ = ssl_early_data_peer_declined;
  }
}
void QuicCryptoClientHandshaker::DoReceiveREJ(
    const CryptoHandshakeMessage* in,
    QuicCryptoClientConfig::CachedState* cached) {
  if (in->tag() != kREJ) {
    next_state_ = STATE_NONE;
    stream_->OnUnrecoverableError(QUIC_INVALID_CRYPTO_MESSAGE_TYPE,
                                  "Expected REJ");
    return;
  }
  QuicTagVector reject_reasons;
  static_assert(sizeof(QuicTag) == sizeof(uint32_t), "header out of sync");
  if (in->GetTaglist(kRREJ, &reject_reasons) == QUIC_NO_ERROR) {
    uint32_t packed_error = 0;
    for (size_t i = 0; i < reject_reasons.size(); ++i) {
      if (reject_reasons[i] == HANDSHAKE_OK || reject_reasons[i] >= 32) {
        continue;
      }
      HandshakeFailureReason reason =
          static_cast<HandshakeFailureReason>(reject_reasons[i]);
      packed_error |= 1 << (reason - 1);
    }
    QUIC_DVLOG(1) << "Reasons for rejection: " << packed_error;
  }
  delegate_->NeuterUnencryptedData();
  std::string error_details;
  QuicErrorCode error = crypto_config_->ProcessRejection(
      *in, session()->connection()->clock()->WallNow(),
      session()->transport_version(), chlo_hash_, cached,
      crypto_negotiated_params_, &error_details);
  if (error != QUIC_NO_ERROR) {
    next_state_ = STATE_NONE;
    stream_->OnUnrecoverableError(error, error_details);
    return;
  }
  if (!cached->proof_valid()) {
    if (!cached->signature().empty()) {
      next_state_ = STATE_VERIFY_PROOF;
      return;
    }
  }
  next_state_ = STATE_SEND_CHLO;
}
QuicAsyncStatus QuicCryptoClientHandshaker::DoVerifyProof(
    QuicCryptoClientConfig::CachedState* cached) {
  ProofVerifier* verifier = crypto_config_->proof_verifier();
  QUICHE_DCHECK(verifier);
  next_state_ = STATE_VERIFY_PROOF_COMPLETE;
  generation_counter_ = cached->generation_counter();
  ProofVerifierCallbackImpl* proof_verify_callback =
      new ProofVerifierCallbackImpl(this);
  verify_ok_ = false;
  QuicAsyncStatus status = verifier->VerifyProof(
      server_id_.host(), server_id_.port(), cached->server_config(),
      session()->transport_version(), chlo_hash_, cached->certs(),
      cached->cert_sct(), cached->signature(), verify_context_.get(),
      &verify_error_details_, &verify_details_,
      std::unique_ptr<ProofVerifierCallback>(proof_verify_callback));
  switch (status) {
    case QUIC_PENDING:
      proof_verify_callback_ = proof_verify_callback;
      QUIC_DVLOG(1) << "Doing VerifyProof";
      break;
    case QUIC_FAILURE:
      break;
    case QUIC_SUCCESS:
      verify_ok_ = true;
      break;
  }
  return status;
}
void QuicCryptoClientHandshaker::DoVerifyProofComplete(
    QuicCryptoClientConfig::CachedState* cached) {
  if (proof_verify_start_time_.IsInitialized()) {
    QUIC_CLIENT_HISTOGRAM_TIMES(
        "QuicSession.VerifyProofTime.CachedServerConfig",
        (session()->connection()->clock()->Now() - proof_verify_start_time_),
        QuicTime::Delta::FromMilliseconds(1), QuicTime::Delta::FromSeconds(10),
        50, "");
  }
  if (!verify_ok_) {
    if (verify_details_) {
      proof_handler_->OnProofVerifyDetailsAvailable(*verify_details_);
    }
    if (num_client_hellos_ == 0) {
      cached->Clear();
      next_state_ = STATE_INITIALIZE;
      return;
    }
    next_state_ = STATE_NONE;
    QUIC_CLIENT_HISTOGRAM_BOOL("QuicVerifyProofFailed.HandshakeConfirmed",
                               one_rtt_keys_available(), "");
    stream_->OnUnrecoverableError(QUIC_PROOF_INVALID,
                                  "Proof invalid: " + verify_error_details_);
    return;
  }
  if (generation_counter_ != cached->generation_counter()) {
    next_state_ = STATE_VERIFY_PROOF;
  } else {
    SetCachedProofValid(cached);
    cached->SetProofVerifyDetails(verify_details_.release());
    if (!one_rtt_keys_available()) {
      next_state_ = STATE_SEND_CHLO;
    } else {
      next_state_ = STATE_NONE;
    }
  }
}
void QuicCryptoClientHandshaker::DoReceiveSHLO(
    const CryptoHandshakeMessage* in,
    QuicCryptoClientConfig::CachedState* cached) {
  next_state_ = STATE_NONE;
  if (in->tag() == kREJ) {
    if (session()->connection()->last_decrypted_level() != ENCRYPTION_INITIAL) {
      stream_->OnUnrecoverableError(QUIC_CRYPTO_ENCRYPTION_LEVEL_INCORRECT,
                                    "encrypted REJ message");
      return;
    }
    next_state_ = STATE_RECV_REJ;
    return;
  }
  if (in->tag() != kSHLO) {
    stream_->OnUnrecoverableError(
        QUIC_INVALID_CRYPTO_MESSAGE_TYPE,
        absl::StrCat("Expected SHLO or REJ. Received: ",
                     QuicTagToString(in->tag())));
    return;
  }
  if (session()->connection()->last_decrypted_level() == ENCRYPTION_INITIAL) {
    stream_->OnUnrecoverableError(QUIC_CRYPTO_ENCRYPTION_LEVEL_INCORRECT,
                                  "unencrypted SHLO message");
    return;
  }
  if (num_client_hellos_ == 1) {
    early_data_reason_ = ssl_early_data_accepted;
  }
  std::string error_details;
  QuicErrorCode error = crypto_config_->ProcessServerHello(
      *in, session()->connection()->connection_id(),
      session()->connection()->version(),
      session()->connection()->server_supported_versions(), cached,
      crypto_negotiated_params_, &error_details);
  if (error != QUIC_NO_ERROR) {
    stream_->OnUnrecoverableError(error,
                                  "Server hello invalid: " + error_details);
    return;
  }
  error = session()->config()->ProcessPeerHello(*in, SERVER, &error_details);
  if (error != QUIC_NO_ERROR) {
    stream_->OnUnrecoverableError(error,
                                  "Server hello invalid: " + error_details);
    return;
  }
  session()->OnConfigNegotiated();
  CrypterPair* crypters = &crypto_negotiated_params_->forward_secure_crypters;
  delegate_->OnNewEncryptionKeyAvailable(ENCRYPTION_FORWARD_SECURE,
                                         std::move(crypters->encrypter));
  delegate_->OnNewDecryptionKeyAvailable(ENCRYPTION_FORWARD_SECURE,
                                         std::move(crypters->decrypter),
                                         true,
                                         false);
  one_rtt_keys_available_ = true;
  delegate_->SetDefaultEncryptionLevel(ENCRYPTION_FORWARD_SECURE);
  delegate_->DiscardOldEncryptionKey(ENCRYPTION_INITIAL);
  delegate_->NeuterHandshakeData();
}
void QuicCryptoClientHandshaker::DoInitializeServerConfigUpdate(
    QuicCryptoClientConfig::CachedState* cached) {
  bool update_ignored = false;
  if (!cached->IsEmpty() && !cached->signature().empty()) {
    QUICHE_DCHECK(crypto_config_->proof_verifier());
    next_state_ = STATE_VERIFY_PROOF;
  } else {
    update_ignored = true;
    next_state_ = STATE_NONE;
  }
  QUIC_CLIENT_HISTOGRAM_COUNTS("QuicNumServerConfig.UpdateMessagesIgnored",
                               update_ignored, 1, 1000000, 50, "");
}
void QuicCryptoClientHandshaker::SetCachedProofValid(
    QuicCryptoClientConfig::CachedState* cached) {
  cached->SetProofValid();
  proof_handler_->OnProofValid(*cached);
}
}  