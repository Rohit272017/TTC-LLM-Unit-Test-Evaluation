#include "quiche/quic/core/qpack/qpack_encoder_stream_receiver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "absl/strings/string_view.h"
#include "quiche/quic/core/quic_error_codes.h"
#include "quiche/quic/core/qpack/qpack_instruction_decoder.h"
#include "quiche/quic/core/qpack/qpack_instructions.h"

namespace quic {
namespace {
using ::testing::_;
using ::testing::InSequence;
using ::testing::StrictMock;

class MockQpackEncoderStreamReceiverDelegate
    : public QpackEncoderStreamReceiver::Delegate {
 public:
  MOCK_METHOD(void, OnInsertWithNameReference,
              (bool is_static, uint64_t name_index, absl::string_view value),
              (override));
  MOCK_METHOD(void, OnInsertWithoutNameReference,
              (absl::string_view name, absl::string_view value), (override));
  MOCK_METHOD(void, OnDuplicate, (uint64_t index), (override));
  MOCK_METHOD(void, OnSetDynamicTableCapacity, (uint64_t capacity), (override));
  MOCK_METHOD(void, OnErrorDetected,
              (QuicErrorCode error_code, absl::string_view error_message),
              (override));
};

TEST(QpackEncoderStreamReceiverTest, EmptyDecodeDoesNotNotifyDelegate) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  receiver.Decode("");
}

TEST(QpackEncoderStreamReceiverTest, DecodesInsertWithStaticNameReference) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnInsertWithNameReference(
                            true, 0, absl::string_view("value")));

  receiver.Decode(std::string("\xc0\x05value", 7));
}

TEST(QpackEncoderStreamReceiverTest, DecodesInsertWithDynamicNameReference) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnInsertWithNameReference(
                            false, 0, absl::string_view("value")));

  receiver.Decode(std::string("\x80\x05value", 7));
}

TEST(QpackEncoderStreamReceiverTest, DecodesInsertWithNameReferenceEmptyValue) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnInsertWithNameReference(true, 1, absl::string_view("")));

  receiver.Decode(std::string("\xc1\x00", 2));
}

TEST(QpackEncoderStreamReceiverTest, DecodesInsertWithoutNameReference) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnInsertWithoutNameReference(
                            absl::string_view("name"),
                            absl::string_view("value")));

  receiver.Decode(std::string("\x40\x04name\x05value", 12));
}

TEST(QpackEncoderStreamReceiverTest, DecodesInsertWithoutNameReferenceEmptyNameAndValue) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnInsertWithoutNameReference(
                            absl::string_view(""),
                            absl::string_view("")));

  receiver.Decode(std::string("\x40\x00\x00", 3));
}

TEST(QpackEncoderStreamReceiverTest, DecodesDuplicateInstructionZeroIndex) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnDuplicate(0));

  receiver.Decode(std::string("\x00", 1));
}

TEST(QpackEncoderStreamReceiverTest, DecodesDuplicateInstructionNonZeroIndex) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnDuplicate(10));

  receiver.Decode(std::string("\x0a", 1));
}

TEST(QpackEncoderStreamReceiverTest, DecodesSetDynamicTableCapacityZero) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnSetDynamicTableCapacity(0));

  receiver.Decode(std::string("\x20", 1));
}

TEST(QpackEncoderStreamReceiverTest, DecodesSetDynamicTableCapacityNonZero) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnSetDynamicTableCapacity(31));

  receiver.Decode(std::string("\x3f\x00", 2));
}

TEST(QpackEncoderStreamReceiverTest, DecodesMultipleInstructionsInOneDecodeCall) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  {
    InSequence sequence;
    EXPECT_CALL(delegate, OnSetDynamicTableCapacity(0));
    EXPECT_CALL(delegate, OnDuplicate(0));
  }

  receiver.Decode(std::string("\x20\x00", 2));
}

TEST(QpackEncoderStreamReceiverTest, DecodesInstructionSplitAcrossMultipleDecodeCalls) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  receiver.Decode(std::string("\x40\x04na", 4));

  EXPECT_CALL(delegate, OnInsertWithoutNameReference(
                            absl::string_view("name"),
                            absl::string_view("value")));

  receiver.Decode(std::string("me\x05value", 8));
}

TEST(QpackEncoderStreamReceiverTest, IntegerTooLargeErrorIsMappedToQpackIntegerTooLarge) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_INTEGER_TOO_LARGE, _));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::INTEGER_TOO_LARGE,
      "integer too large");
}

TEST(QpackEncoderStreamReceiverTest, StringLiteralTooLongErrorIsMappedToQpackStringTooLong) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_STRING_LITERAL_TOO_LONG, _));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::STRING_LITERAL_TOO_LONG,
      "string literal too long");
}

TEST(QpackEncoderStreamReceiverTest, HuffmanEncodingErrorIsMappedToQpackHuffmanError) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_HUFFMAN_ENCODING_ERROR, _));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::HUFFMAN_ENCODING_ERROR,
      "huffman error");
}

TEST(QpackEncoderStreamReceiverTest, UnknownDecoderErrorIsMappedToInternalError) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate, OnErrorDetected(QUIC_INTERNAL_ERROR, _));

  receiver.OnInstructionDecodingError(
      static_cast<QpackInstructionDecoder::ErrorCode>(999),
      "unknown error");
}

TEST(QpackEncoderStreamReceiverTest, ErrorMessageIsForwardedToDelegate) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_INTEGER_TOO_LARGE,
                  absl::string_view("integer too large")));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::INTEGER_TOO_LARGE,
      "integer too large");
}

TEST(QpackEncoderStreamReceiverTest, DecodeAfterErrorDoesNotNotifyDelegate) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_INTEGER_TOO_LARGE,
                  absl::string_view("integer too large")));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::INTEGER_TOO_LARGE,
      "integer too large");

  receiver.Decode(std::string("\x20", 1));
}

TEST(QpackEncoderStreamReceiverTest, EmptyDecodeAfterErrorDoesNothing) {
  StrictMock<MockQpackEncoderStreamReceiverDelegate> delegate;
  QpackEncoderStreamReceiver receiver(&delegate);

  EXPECT_CALL(delegate,
              OnErrorDetected(
                  QUIC_QPACK_ENCODER_STREAM_STRING_LITERAL_TOO_LONG,
                  absl::string_view("string too long")));

  receiver.OnInstructionDecodingError(
      QpackInstructionDecoder::ErrorCode::STRING_LITERAL_TOO_LONG,
      "string too long");

  receiver.Decode("");
}

}  // namespace
}  // namespace quic