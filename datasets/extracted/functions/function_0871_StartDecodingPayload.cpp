#include "quiche/http2/decoder/payload_decoders/ping_payload_decoder.h"
#include "quiche/http2/decoder/http2_frame_decoder_listener.h"
#include "quiche/http2/http2_constants.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
namespace {
constexpr auto kOpaqueSize = Http2PingFields::EncodedSize();
}
DecodeStatus PingPayloadDecoder::StartDecodingPayload(FrameDecoderState* state,
                                                      DecodeBuffer* db) {
  const Http2FrameHeader& frame_header = state->frame_header();
  const uint32_t total_length = frame_header.payload_length;
  QUICHE_DVLOG(2) << "PingPayloadDecoder::StartDecodingPayload: "
                  << frame_header;
  QUICHE_DCHECK_EQ(Http2FrameType::PING, frame_header.type);
  QUICHE_DCHECK_LE(db->Remaining(), total_length);
  QUICHE_DCHECK_EQ(0, frame_header.flags & ~(Http2FrameFlag::ACK));
  if (db->Remaining() == kOpaqueSize && total_length == kOpaqueSize) {
    static_assert(sizeof(Http2PingFields) == kOpaqueSize,
                  "If not, then can't enter this block!");
    auto* ping = reinterpret_cast<const Http2PingFields*>(db->cursor());
    if (frame_header.IsAck()) {
      state->listener()->OnPingAck(frame_header, *ping);
    } else {
      state->listener()->OnPing(frame_header, *ping);
    }
    db->AdvanceCursor(kOpaqueSize);
    return DecodeStatus::kDecodeDone;
  }
  state->InitializeRemainders();
  return HandleStatus(
      state, state->StartDecodingStructureInPayload(&ping_fields_, db));
}
DecodeStatus PingPayloadDecoder::ResumeDecodingPayload(FrameDecoderState* state,
                                                       DecodeBuffer* db) {
  QUICHE_DVLOG(2) << "ResumeDecodingPayload: remaining_payload="
                  << state->remaining_payload();
  QUICHE_DCHECK_EQ(Http2FrameType::PING, state->frame_header().type);
  QUICHE_DCHECK_LE(db->Remaining(), state->frame_header().payload_length);
  return HandleStatus(
      state, state->ResumeDecodingStructureInPayload(&ping_fields_, db));
}
DecodeStatus PingPayloadDecoder::HandleStatus(FrameDecoderState* state,
                                              DecodeStatus status) {
  QUICHE_DVLOG(2) << "HandleStatus: status=" << status
                  << "; remaining_payload=" << state->remaining_payload();
  if (status == DecodeStatus::kDecodeDone) {
    if (state->remaining_payload() == 0) {
      const Http2FrameHeader& frame_header = state->frame_header();
      if (frame_header.IsAck()) {
        state->listener()->OnPingAck(frame_header, ping_fields_);
      } else {
        state->listener()->OnPing(frame_header, ping_fields_);
      }
      return DecodeStatus::kDecodeDone;
    }
    return state->ReportFrameSizeError();
  }
  QUICHE_DCHECK(
      (status == DecodeStatus::kDecodeInProgress &&
       state->remaining_payload() > 0) ||
      (status == DecodeStatus::kDecodeError && state->remaining_payload() == 0))
      << "\n status=" << status
      << "; remaining_payload=" << state->remaining_payload();
  return status;
}
}  