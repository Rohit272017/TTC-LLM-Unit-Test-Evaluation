#include "quiche/http2/decoder/payload_decoders/headers_payload_decoder.h"
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
                         HeadersPayloadDecoder::PayloadState v) {
  switch (v) {
    case HeadersPayloadDecoder::PayloadState::kReadPadLength:
      return out << "kReadPadLength";
    case HeadersPayloadDecoder::PayloadState::kStartDecodingPriorityFields:
      return out << "kStartDecodingPriorityFields";
    case HeadersPayloadDecoder::PayloadState::kResumeDecodingPriorityFields:
      return out << "kResumeDecodingPriorityFields";
    case HeadersPayloadDecoder::PayloadState::kReadPayload:
      return out << "kReadPayload";
    case HeadersPayloadDecoder::PayloadState::kSkipPadding:
      return out << "kSkipPadding";
  }
  int unknown = static_cast<int>(v);
  QUICHE_BUG(http2_bug_189_1)
      << "Invalid HeadersPayloadDecoder::PayloadState: " << unknown;
  return out << "HeadersPayloadDecoder::PayloadState(" << unknown << ")";
}
DecodeStatus HeadersPayloadDecoder::StartDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  const Http2FrameHeader& frame_header = state->frame_header();
  const uint32_t total_length = frame_header.payload_length;
  QUICHE_DVLOG(2) << "HeadersPayloadDecoder::StartDecodingPayload: "
                  << frame_header;
  QUICHE_DCHECK_EQ(Http2FrameType::HEADERS, frame_header.type);
  QUICHE_DCHECK_LE(db->Remaining(), total_length);
  QUICHE_DCHECK_EQ(
      0, frame_header.flags &
             ~(Http2FrameFlag::END_STREAM | Http2FrameFlag::END_HEADERS |
               Http2FrameFlag::PADDED | Http2FrameFlag::PRIORITY));
  const auto payload_flags = Http2FrameFlag::PADDED | Http2FrameFlag::PRIORITY;
  if (!frame_header.HasAnyFlags(payload_flags)) {
    QUICHE_DVLOG(2) << "StartDecodingPayload !IsPadded && !HasPriority";
    if (db->Remaining() == total_length) {
      QUICHE_DVLOG(2) << "StartDecodingPayload all present";
      state->listener()->OnHeadersStart(frame_header);
      if (total_length > 0) {
        state->listener()->OnHpackFragment(db->cursor(), total_length);
        db->AdvanceCursor(total_length);
      }
      state->listener()->OnHeadersEnd();
      return DecodeStatus::kDecodeDone;
    }
    payload_state_ = PayloadState::kReadPayload;
  } else if (frame_header.IsPadded()) {
    payload_state_ = PayloadState::kReadPadLength;
  } else {
    QUICHE_DCHECK(frame_header.HasPriority()) << frame_header;
    payload_state_ = PayloadState::kStartDecodingPriorityFields;
  }
  state->InitializeRemainders();
  state->listener()->OnHeadersStart(frame_header);
  return ResumeDecodingPayload(state, db);
}
DecodeStatus HeadersPayloadDecoder::ResumeDecodingPayload(
    FrameDecoderState* state, DecodeBuffer* db) {
  QUICHE_DVLOG(2) << "HeadersPayloadDecoder::ResumeDecodingPayload "
                  << "remaining_payload=" << state->remaining_payload()
                  << "; db->Remaining=" << db->Remaining();
  const Http2FrameHeader& frame_header = state->frame_header();
  QUICHE_DCHECK_EQ(Http2FrameType::HEADERS, frame_header.type);
  QUICHE_DCHECK_LE(state->remaining_payload_and_padding(),
                   frame_header.payload_length);
  QUICHE_DCHECK_LE(db->Remaining(), state->remaining_payload_and_padding());
  DecodeStatus status;
  size_t avail;
  while (true) {
    QUICHE_DVLOG(2)
        << "HeadersPayloadDecoder::ResumeDecodingPayload payload_state_="
        << payload_state_;
    switch (payload_state_) {
      case PayloadState::kReadPadLength:
        status = state->ReadPadLength(db,  true);
        if (status != DecodeStatus::kDecodeDone) {
          return status;
        }
        if (!frame_header.HasPriority()) {
          payload_state_ = PayloadState::kReadPayload;
          continue;
        }
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kStartDecodingPriorityFields:
        status = state->StartDecodingStructureInPayload(&priority_fields_, db);
        if (status != DecodeStatus::kDecodeDone) {
          payload_state_ = PayloadState::kResumeDecodingPriorityFields;
          return status;
        }
        state->listener()->OnHeadersPriority(priority_fields_);
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kReadPayload:
        avail = state->AvailablePayload(db);
        if (avail > 0) {
          state->listener()->OnHpackFragment(db->cursor(), avail);
          db->AdvanceCursor(avail);
          state->ConsumePayload(avail);
        }
        if (state->remaining_payload() > 0) {
          payload_state_ = PayloadState::kReadPayload;
          return DecodeStatus::kDecodeInProgress;
        }
        ABSL_FALLTHROUGH_INTENDED;
      case PayloadState::kSkipPadding:
        if (state->SkipPadding(db)) {
          state->listener()->OnHeadersEnd();
          return DecodeStatus::kDecodeDone;
        }
        payload_state_ = PayloadState::kSkipPadding;
        return DecodeStatus::kDecodeInProgress;
      case PayloadState::kResumeDecodingPriorityFields:
        status = state->ResumeDecodingStructureInPayload(&priority_fields_, db);
        if (status != DecodeStatus::kDecodeDone) {
          return status;
        }
        state->listener()->OnHeadersPriority(priority_fields_);
        payload_state_ = PayloadState::kReadPayload;
        continue;
    }
    QUICHE_BUG(http2_bug_189_2) << "PayloadState: " << payload_state_;
  }
}
}  