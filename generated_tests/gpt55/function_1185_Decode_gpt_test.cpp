#include "quiche/http2/hpack/decoder/hpack_block_decoder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "quiche/http2/decoder/decode_buffer.h"
#include "quiche/http2/decoder/decode_status.h"
#include "quiche/http2/hpack/decoder/hpack_decoder_listener.h"

namespace http2 {
namespace {

using ::testing::_;

class MockHpackDecoderListener : public HpackDecoderListener {
 public:
  MOCK_METHOD(void, OnHeaderListStart, (), (override));
  MOCK_METHOD(void, OnHeader, (const std::string& name,
                              const std::string& value), (override));
  MOCK_METHOD(void, OnHeaderListEnd, (), (override));
  MOCK_METHOD(void, OnHeaderErrorDetected, (absl::string_view error_message),
              (override));
};

TEST(HpackBlockDecoderTest, EmptyBufferReturnsDecodeDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  std::string input;
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, IndexedHeaderFieldDecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader(_, _)).Times(1);

  const std::string input(1, static_cast<char>(0x82));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, MultipleIndexedHeaderFieldsDecodeDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader(_, _)).Times(3);

  const std::string input = std::string(1, static_cast<char>(0x82)) +
                            std::string(1, static_cast<char>(0x83)) +
                            std::string(1, static_cast<char>(0x84));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, LiteralHeaderWithoutIndexingWithIndexedNameDecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader(_, "abc")).Times(1);

  const std::string input =
      std::string(1, static_cast<char>(0x04)) +
      std::string(1, static_cast<char>(0x03)) +
      "abc";
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, LiteralHeaderWithoutIndexingWithEmptyValueDecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader(_, "")).Times(1);

  const std::string input =
      std::string(1, static_cast<char>(0x04)) +
      std::string(1, static_cast<char>(0x00));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, LiteralHeaderWithoutIndexingWithNewNameDecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader("abc", "xyz")).Times(1);

  const std::string input =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x03)) +
      "abc" +
      std::string(1, static_cast<char>(0x03)) +
      "xyz";
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, IncompleteLiteralHeaderReturnsDecodeInProgress) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x03)) +
      "a";
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeInProgress);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, ResumeIncompleteLiteralHeaderReturnsDecodeDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  EXPECT_CALL(listener, OnHeader("abc", "xyz")).Times(1);

  const std::string part1 =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x03)) +
      "a";
  DecodeBuffer db1(part1);

  EXPECT_EQ(decoder.Decode(&db1), DecodeStatus::kDecodeInProgress);
  EXPECT_EQ(db1.Remaining(), 0u);

  const std::string part2 =
      "bc" +
      std::string(1, static_cast<char>(0x03)) +
      "xyz";
  DecodeBuffer db2(part2);

  EXPECT_EQ(decoder.Decode(&db2), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db2.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, ResumeThenDecodeNextEntryInSameBuffer) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  {
    ::testing::InSequence sequence;
    EXPECT_CALL(listener, OnHeader("abc", "xyz")).Times(1);
    EXPECT_CALL(listener, OnHeader(_, _)).Times(1);
  }

  const std::string part1 =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x03)) +
      "a";
  DecodeBuffer db1(part1);

  EXPECT_EQ(decoder.Decode(&db1), DecodeStatus::kDecodeInProgress);
  EXPECT_EQ(db1.Remaining(), 0u);

  const std::string part2 =
      "bc" +
      std::string(1, static_cast<char>(0x03)) +
      "xyz" +
      std::string(1, static_cast<char>(0x82));
  DecodeBuffer db2(part2);

  EXPECT_EQ(decoder.Decode(&db2), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db2.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, DynamicTableSizeUpdateDecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input(1, static_cast<char>(0x20));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, DynamicTableSizeUpdateBoundaryValue31DecodesDone) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input(1, static_cast<char>(0x3f));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeInProgress);
  EXPECT_EQ(db.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, DynamicTableSizeUpdateExtendedValueCompletesOnResume) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string part1(1, static_cast<char>(0x3f));
  DecodeBuffer db1(part1);

  EXPECT_EQ(decoder.Decode(&db1), DecodeStatus::kDecodeInProgress);
  EXPECT_EQ(db1.Remaining(), 0u);

  const std::string part2(1, static_cast<char>(0x00));
  DecodeBuffer db2(part2);

  EXPECT_EQ(decoder.Decode(&db2), DecodeStatus::kDecodeDone);
  EXPECT_EQ(db2.Remaining(), 0u);
}

TEST(HpackBlockDecoderTest, InvalidIndexedHeaderZeroReturnsDecodeError) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input(1, static_cast<char>(0x80));
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeError);
}

TEST(HpackBlockDecoderTest, InvalidIndexedNameZeroReturnsDecodeError) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x01)) +
      "x";
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeError);
}

TEST(HpackBlockDecoderTest, DebugStringBetweenEntriesContainsState) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string debug = decoder.DebugString();

  EXPECT_NE(debug.find("HpackBlockDecoder("), std::string::npos);
  EXPECT_NE(debug.find("listener@"), std::string::npos);
  EXPECT_NE(debug.find("between entries"), std::string::npos);
}

TEST(HpackBlockDecoderTest, DebugStringInEntryContainsState) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  const std::string input =
      std::string(1, static_cast<char>(0x00)) +
      std::string(1, static_cast<char>(0x03)) +
      "a";
  DecodeBuffer db(input);

  EXPECT_EQ(decoder.Decode(&db), DecodeStatus::kDecodeInProgress);

  const std::string debug = decoder.DebugString();

  EXPECT_NE(debug.find("HpackBlockDecoder("), std::string::npos);
  EXPECT_NE(debug.find("listener@"), std::string::npos);
  EXPECT_NE(debug.find("in an entry"), std::string::npos);
}

TEST(HpackBlockDecoderTest, StreamOperatorOutputsDebugString) {
  MockHpackDecoderListener listener;
  HpackBlockDecoder decoder(&listener);

  std::ostringstream oss;
  oss << decoder;

  EXPECT_EQ(oss.str(), decoder.DebugString());
}

}  // namespace
}  // namespace http2