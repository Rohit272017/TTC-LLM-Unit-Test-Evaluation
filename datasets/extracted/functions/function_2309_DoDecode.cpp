#include "quiche/http2/decoder/decode_http2_structures.h"
#include <cstdint>
#include <cstring>
#include "quiche/http2/decoder/decode_buffer.h"
#include "quiche/http2/http2_constants.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
void DoDecode(Http2FrameHeader* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2FrameHeader::EncodedSize(), b->Remaining());
  out->payload_length = b->DecodeUInt24();
  out->type = static_cast<Http2FrameType>(b->DecodeUInt8());
  out->flags = static_cast<Http2FrameFlag>(b->DecodeUInt8());
  out->stream_id = b->DecodeUInt31();
}
void DoDecode(Http2PriorityFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2PriorityFields::EncodedSize(), b->Remaining());
  uint32_t stream_id_and_flag = b->DecodeUInt32();
  out->stream_dependency = stream_id_and_flag & StreamIdMask();
  if (out->stream_dependency == stream_id_and_flag) {
    out->is_exclusive = false;
  } else {
    out->is_exclusive = true;
  }
  out->weight = b->DecodeUInt8() + 1;
}
void DoDecode(Http2RstStreamFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2RstStreamFields::EncodedSize(), b->Remaining());
  out->error_code = static_cast<Http2ErrorCode>(b->DecodeUInt32());
}
void DoDecode(Http2SettingFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2SettingFields::EncodedSize(), b->Remaining());
  out->parameter = static_cast<Http2SettingsParameter>(b->DecodeUInt16());
  out->value = b->DecodeUInt32();
}
void DoDecode(Http2PushPromiseFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2PushPromiseFields::EncodedSize(), b->Remaining());
  out->promised_stream_id = b->DecodeUInt31();
}
void DoDecode(Http2PingFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2PingFields::EncodedSize(), b->Remaining());
  memcpy(out->opaque_bytes, b->cursor(), Http2PingFields::EncodedSize());
  b->AdvanceCursor(Http2PingFields::EncodedSize());
}
void DoDecode(Http2GoAwayFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2GoAwayFields::EncodedSize(), b->Remaining());
  out->last_stream_id = b->DecodeUInt31();
  out->error_code = static_cast<Http2ErrorCode>(b->DecodeUInt32());
}
void DoDecode(Http2WindowUpdateFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2WindowUpdateFields::EncodedSize(), b->Remaining());
  out->window_size_increment = b->DecodeUInt31();
}
void DoDecode(Http2PriorityUpdateFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2PriorityUpdateFields::EncodedSize(), b->Remaining());
  out->prioritized_stream_id = b->DecodeUInt31();
}
void DoDecode(Http2AltSvcFields* out, DecodeBuffer* b) {
  QUICHE_DCHECK_NE(nullptr, out);
  QUICHE_DCHECK_NE(nullptr, b);
  QUICHE_DCHECK_LE(Http2AltSvcFields::EncodedSize(), b->Remaining());
  out->origin_length = b->DecodeUInt16();
}
}  