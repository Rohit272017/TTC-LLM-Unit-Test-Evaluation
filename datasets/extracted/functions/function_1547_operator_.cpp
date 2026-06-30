#include "quiche/http2/decoder/payload_decoders/goaway_payload_decoder.h"
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
                         GoAwayPayloadDecoder::PayloadState v) {
  switch (v) {
    case GoAwayPayloadDecoder::PayloadState::kStartDecodingFixedFields:
      return out << "kStartDecodingFixedFields";
    case GoAwayPayloadDecoder::PayloadState::kHandleFixedFieldsStatus:
      return out << "kHandleFixedFieldsStatus";
    case GoAwayPayloadDecoder::PayloadState::kReadOpaqueData:
      return out << "kReadOpaqueData";
    case GoAwayPayloadDecoder::PayloadState::kResumeDecodingFixedFields:
      return out << "kResumeDecodingFixedFields";
  }
  int unknown = static_cast<int>(v);
  QUICHE_BUG(http2_bug_167_1)
      << "Invalid GoAwayPayloadDecoder::PayloadState: " << unknown;
  return out << "GoAwayPayloadDecoder::PayloadState(" << unknown << ")";
}
DecodeStatus GoAwayPayloadDecoder::StartDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  QUICHE_DVLOG(2) << "GoAwayPayloadDecoder::StartDecodingPayload: "
                  << state->frame_header();
  QUICHE_DCHECK_EQ(Http2FrameType::GOAWAY, state->frame_header().type);
  QUICHE_DCHECK_LE(db->Remaining(), state->frame_header().payload_length);
  QUICHE_DCHECK_EQ(0, state->frame_header().flags);
  state->InitializeRemainders();
  payload_state_ = PayloadState::kStartDecodingFixedFields;
  return ResumeDecodingPayload(state, db);
}
DecodeStatus GoAwayPayloadDecoder::ResumeDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  QUICHE_DVLOG(2)
      << "GoAwayPayloadDecoder::ResumeDecodingPayload: remaining_payload="
      << state->remaining_payload() << ", db->Remaining=" << db->Remaining();
  const Http2FrameHeader& frame_header = state->frame_header();
  QUICHE_DCHECK_EQ(Http2FrameType::GOAWAY, frame_header.type);
  QUICHE_DCHECK_LE(db->Remaining(), frame_header.payload_length);
  QUICHE_DCHECK_NE(PayloadState::kHandleFixedFieldsStatus, payload_state_);
  DecodeStatus status = DecodeStatus::kDecodeError;
  size_t avail;
  while (true) {
    QUICHE_DVLOG(2)
        << "GoAwayPayloadDecoder::ResumeDecodingPayload payload_state_="
        << payload_state_;
    switch (payload_state_) {
      case PayloadState::kStartDecodingFixedFields:
        status = state->StartDecodingStructureInPayload(&goaway_fields_, db);
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kHandleFixedFieldsStatus:
        if (status == DecodeStatus::kDecodeDone) {
          state->listener()->OnGoAwayStart(frame_header, goaway_fields_);
        } else {
          QUICHE_DCHECK((status == DecodeStatus::kDecodeInProgress &&
                         state->remaining_payload() > 0) ||
                        (status == DecodeStatus::kDecodeError &&
                         state->remaining_payload() == 0))
              << "\n status=" << status
              << "; remaining_payload=" << state->remaining_payload();
          payload_state_ = PayloadState::kResumeDecodingFixedFields;
          return status;
        }
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kReadOpaqueData:
        avail = db->Remaining();
        if (avail > 0) {
          state->listener()->OnGoAwayOpaqueData(db->cursor(), avail);
          db->AdvanceCursor(avail);
          state->ConsumePayload(avail);
        }
        if (state->remaining_payload() > 0) {
          payload_state_ = PayloadState::kReadOpaqueData;
          return DecodeStatus::kDecodeInProgress;
        }
        state->listener()->OnGoAwayEnd();
        return DecodeStatus::kDecodeDone;
      case PayloadState::kResumeDecodingFixedFields:
        status = state->ResumeDecodingStructureInPayload(&goaway_fields_, db);
        payload_state_ = PayloadState::kHandleFixedFieldsStatus;
        continue;
    }
    QUICHE_BUG(http2_bug_167_2) << "PayloadState: " << payload_state_;
  }
}
}  