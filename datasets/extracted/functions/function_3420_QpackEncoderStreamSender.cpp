#include "quiche/quic/core/qpack/qpack_encoder_stream_sender.h"
#include <cstddef>
#include <limits>
#include <string>
#include "absl/strings/string_view.h"
#include "quiche/quic/core/qpack/qpack_instructions.h"
#include "quiche/quic/platform/api/quic_logging.h"
namespace quic {
namespace {
constexpr uint64_t kMaxBytesBufferedByStream = 64 * 1024;
}  
QpackEncoderStreamSender::QpackEncoderStreamSender(
    HuffmanEncoding huffman_encoding)
    : delegate_(nullptr), instruction_encoder_(huffman_encoding) {}
void QpackEncoderStreamSender::SendInsertWithNameReference(
    bool is_static, uint64_t name_index, absl::string_view value) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::InsertWithNameReference(is_static, name_index,
                                                          value),
      &buffer_);
}
void QpackEncoderStreamSender::SendInsertWithoutNameReference(
    absl::string_view name, absl::string_view value) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::InsertWithoutNameReference(name, value),
      &buffer_);
}
void QpackEncoderStreamSender::SendDuplicate(uint64_t index) {
  instruction_encoder_.Encode(QpackInstructionWithValues::Duplicate(index),
                              &buffer_);
}
void QpackEncoderStreamSender::SendSetDynamicTableCapacity(uint64_t capacity) {
  instruction_encoder_.Encode(
      QpackInstructionWithValues::SetDynamicTableCapacity(capacity), &buffer_);
}
bool QpackEncoderStreamSender::CanWrite() const {
  return delegate_ && delegate_->NumBytesBuffered() + buffer_.size() <=
                          kMaxBytesBufferedByStream;
}
void QpackEncoderStreamSender::Flush() {
  if (buffer_.empty()) {
    return;
  }
  delegate_->WriteStreamData(buffer_);
  buffer_.clear();
}
}  