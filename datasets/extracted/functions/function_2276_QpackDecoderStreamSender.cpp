#include "quiche/quic/core/qpack/qpack_decoder_stream_sender.h"
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "quiche/quic/core/qpack/qpack_instructions.h"
#include "quiche/quic/platform/api/quic_flag_utils.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
QpackDecoderStreamSender::QpackDecoderStreamSender()
    : delegate_(nullptr),
      instruction_encoder_(HuffmanEncoding::kEnabled) {}
void QpackDecoderStreamSender::SendInsertCountIncrement(uint64_t increment) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::InsertCountIncrement(increment), &buffer_);
}
void QpackDecoderStreamSender::SendHeaderAcknowledgement(
    QuicStreamId stream_id) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::HeaderAcknowledgement(stream_id), &buffer_);
}
void QpackDecoderStreamSender::SendStreamCancellation(QuicStreamId stream_id) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::StreamCancellation(stream_id), &buffer_);
}
void QpackDecoderStreamSender::Flush() {
  if (buffer_.empty() || delegate_ == nullptr) {
    return;
  }
  std::string copy;
  std::swap(copy, buffer_);
  delegate_->WriteStreamData(copy);
}
}  