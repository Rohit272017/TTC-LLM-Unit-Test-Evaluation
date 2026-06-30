#include "quiche/http2/decoder/payload_decoders/altsvc_payload_decoder.h"
#include <stddef.h>
#include <ostream>
#include "absl/base/macros.h"
#include "quiche/http2/decoder/decode_buffer.h"
#include "quiche/http2/decoder/http2_frame_decoder_listener.h"
#include "quiche/http2/http2_constants.h"
#include "quiche/http2/http2_structures.h"
#include "quiche/common/platform/api/quiche_bug_tracker.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
std::ostream& operator<<(std::ostream& out,
                         AltSvcPayloadDecoder::PayloadState v) {
  switch (v) {
    case AltSvcPayloadDecoder::PayloadState::kStartDecodingStruct:
      return out << "kStartDecodingStruct";
    case AltSvcPayloadDecoder::PayloadState::kMaybeDecodedStruct:
      return out << "kMaybeDecodedStruct";
    case AltSvcPayloadDecoder::PayloadState::kDecodingStrings:
      return out << "kDecodingStrings";
    case AltSvcPayloadDecoder::PayloadState::kResumeDecodingStruct:
      return out << "kResumeDecodingStruct";
  }
  int unknown = static_cast<int>(v);
  QUICHE_BUG(http2_bug_163_1)
      << "Invalid AltSvcPayloadDecoder::PayloadState: " << unknown;
  return out << "AltSvcPayloadDecoder::PayloadState(" << unknown << ")";
}
DecodeStatus AltSvcPayloadDecoder::StartDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  QUICHE_DVLOG(2) << "AltSvcPayloadDecoder::StartDecodingPayload: "
                  << state->frame_header();
  QUICHE_DCHECK_EQ(Http2FrameType::ALTSVC, state->frame_header().type);
  QUICHE_DCHECK_LE(db->Remaining(), state->frame_header().payload_length);
  QUICHE_DCHECK_EQ(0, state->frame_header().flags);
  state->InitializeRemainders();
  payload_state_ = PayloadState::kStartDecodingStruct;
  return ResumeDecodingPayload(state, db);
}
DecodeStatus AltSvcPayloadDecoder::ResumeDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  const Http2FrameHeader& frame_header = state->frame_header();
  QUICHE_DVLOG(2) << "AltSvcPayloadDecoder::ResumeDecodingPayload: "
                  << frame_header;
  QUICHE_DCHECK_EQ(Http2FrameType::ALTSVC, frame_header.type);
  QUICHE_DCHECK_LE(state->remaining_payload(), frame_header.payload_length);
  QUICHE_DCHECK_LE(db->Remaining(), state->remaining_payload());
  QUICHE_DCHECK_NE(PayloadState::kMaybeDecodedStruct, payload_state_);
  DecodeStatus status = DecodeStatus::kDecodeError;
  while (true) {
    QUICHE_DVLOG(2)
        << "AltSvcPayloadDecoder::ResumeDecodingPayload payload_state_="
        << payload_state_;
    switch (payload_state_) {
      case PayloadState::kStartDecodingStruct:
        status = state->StartDecodingStructureInPayload(&altsvc_fields_, db);
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kMaybeDecodedStruct:
        if (status == DecodeStatus::kDecodeDone &&
            altsvc_fields_.origin_length <= state->remaining_payload()) {
          size_t origin_length = altsvc_fields_.origin_length;
          size_t value_length = state->remaining_payload() - origin_length;
          state->listener()->OnAltSvcStart(frame_header, origin_length,
                                           value_length);
        } else if (status != DecodeStatus::kDecodeDone) {
          QUICHE_DCHECK(state->remaining_payload() > 0 ||
                        status == DecodeStatus::kDecodeError)
              << "\nremaining_payload: " << state->remaining_payload()
              << "\nstatus: " << status << "\nheader: " << frame_header;
          payload_state_ = PayloadState::kResumeDecodingStruct;
          return status;
        } else {
          QUICHE_DCHECK_GT(altsvc_fields_.origin_length,
                           state->remaining_payload());
          return state->ReportFrameSizeError();
        }
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kDecodingStrings:
        return DecodeStrings(state, db);
      case PayloadState::kResumeDecodingStruct:
        status = state->ResumeDecodingStructureInPayload(&altsvc_fields_, db);
        payload_state_ = PayloadState::kMaybeDecodedStruct;
        continue;
    }
    QUICHE_BUG(http2_bug_163_2) << "PayloadState: " << payload_state_;
  }
}
DecodeStatus AltSvcPayloadDecoder::DecodeStrings(FrameDecoderState* state,
                                                 DecodeBuffer* db) {
  QUICHE_DVLOG(2) << "AltSvcPayloadDecoder::DecodeStrings remaining_payload="
                  << state->remaining_payload()
                  << ", db->Remaining=" << db->Remaining();
  size_t origin_length = altsvc_fields_.origin_length;
  size_t value_length = state->frame_header().payload_length - origin_length -
                        Http2AltSvcFields::EncodedSize();
  if (state->remaining_payload() > value_length) {
    size_t remaining_origin_length = state->remaining_payload() - value_length;
    size_t avail = db->MinLengthRemaining(remaining_origin_length);
    state->listener()->OnAltSvcOriginData(db->cursor(), avail);
    db->AdvanceCursor(avail);
    state->ConsumePayload(avail);
    if (remaining_origin_length > avail) {
      payload_state_ = PayloadState::kDecodingStrings;
      return DecodeStatus::kDecodeInProgress;
    }
  }
  QUICHE_DCHECK_LE(state->remaining_payload(), value_length);
  QUICHE_DCHECK_LE(db->Remaining(), state->remaining_payload());
  if (db->HasData()) {
    size_t avail = db->Remaining();
    state->listener()->OnAltSvcValueData(db->cursor(), avail);
    db->AdvanceCursor(avail);
    state->ConsumePayload(avail);
  }
  if (state->remaining_payload() == 0) {
    state->listener()->OnAltSvcEnd();
    return DecodeStatus::kDecodeDone;
  }
  payload_state_ = PayloadState::kDecodingStrings;
  return DecodeStatus::kDecodeInProgress;
}
}  