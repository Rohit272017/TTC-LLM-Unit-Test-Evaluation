#include "tensorstore/internal/compression/zlib.h"

#include <gtest/gtest.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"

namespace tensorstore {
namespace zlib {
namespace {

std::string CordToString(const absl::Cord& cord) {
  std::string result;
  absl::CopyCordToString(cord, &result);
  return result;
}

absl::Cord CordFromString(const std::string& value) {
  return absl::Cord(value);
}

Options DefaultOptions(bool use_gzip_header = false, int level = Z_DEFAULT_COMPRESSION) {
  Options options;
  options.use_gzip_header = use_gzip_header;
  options.level = level;
  return options;
}

void RoundTrip(const std::string& input, const Options& options) {
  absl::Cord compressed;
  Encode(CordFromString(input), &compressed, options);

  EXPECT_FALSE(compressed.empty() || !input.empty());

  absl::Cord decompressed;
  absl::Status status =
      Decode(compressed, &decompressed, options.use_gzip_header);

  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(CordToString(decompressed), input);
}

TEST(ZlibCompressionTest, BVA_EmptyInputRoundTripsWithZlibHeader) {
  RoundTrip("", DefaultOptions(false));
}

TEST(ZlibCompressionTest, BVA_EmptyInputRoundTripsWithGzipHeader) {
  RoundTrip("", DefaultOptions(true));
}

TEST(ZlibCompressionTest, BVA_SingleByteInputRoundTrips) {
  RoundTrip("a", DefaultOptions(false));
}

TEST(ZlibCompressionTest, BVA_TwoByteInputRoundTrips) {
  RoundTrip("ab", DefaultOptions(false));
}

TEST(ZlibCompressionTest, BVA_ThreeByteInputRoundTrips) {
  RoundTrip("abc", DefaultOptions(false));
}

TEST(ZlibCompressionTest, ECP_TextInputRoundTripsWithZlibHeader) {
  RoundTrip("hello tensorstore zlib compression", DefaultOptions(false));
}

TEST(ZlibCompressionTest, ECP_TextInputRoundTripsWithGzipHeader) {
  RoundTrip("hello tensorstore gzip compression", DefaultOptions(true));
}

TEST(ZlibCompressionTest, ECP_RepeatedPatternRoundTrips) {
  std::string input;
  for (int i = 0; i < 1000; ++i) {
    input += "abcabcabcabc";
  }

  RoundTrip(input, DefaultOptions(false));
}

TEST(ZlibCompressionTest, ECP_BinaryDataWithNullBytesRoundTrips) {
  std::string input("abc\0def\0ghi", 11);

  RoundTrip(input, DefaultOptions(false));
}

TEST(ZlibCompressionTest, Edge_AllByteValuesRoundTrip) {
  std::string input;
  for (int i = 0; i <= 255; ++i) {
    input.push_back(static_cast<char>(i));
  }

  RoundTrip(input, DefaultOptions(false));
}

TEST(ZlibCompressionTest, Edge_LargeInputRoundTrips) {
  std::string input;
  for (int i = 0; i < 10000; ++i) {
    input += "large input block ";
    input += std::to_string(i);
    input += '\n';
  }

  RoundTrip(input, DefaultOptions(false));
}

TEST(ZlibCompressionTest, ECP_CompressionLevelZeroRoundTrips) {
  RoundTrip("compression level zero input", DefaultOptions(false, 0));
}

TEST(ZlibCompressionTest, BVA_CompressionLevelOneRoundTrips) {
  RoundTrip("compression level one input", DefaultOptions(false, 1));
}

TEST(ZlibCompressionTest, BVA_CompressionLevelNineRoundTrips) {
  RoundTrip("compression level nine input", DefaultOptions(false, 9));
}

TEST(ZlibCompressionTest, ECP_DefaultCompressionLevelRoundTrips) {
  RoundTrip("default compression level input",
            DefaultOptions(false, Z_DEFAULT_COMPRESSION));
}

TEST(ZlibCompressionTest, ECP_DecodeInvalidPlainTextReturnsInvalidArgument) {
  absl::Cord output;

  absl::Status status =
      Decode(CordFromString("this is not compressed data"), &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, ECP_DecodeInvalidPlainTextWithGzipHeaderReturnsInvalidArgument) {
  absl::Cord output;

  absl::Status status =
      Decode(CordFromString("this is not gzip data"), &output, true);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, BVA_DecodeEmptyInputReturnsInvalidArgument) {
  absl::Cord output;

  absl::Status status = Decode(absl::Cord(), &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeTruncatedZlibDataReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  std::string truncated = CordToString(compressed);
  ASSERT_GT(truncated.size(), 1u);
  truncated.resize(truncated.size() - 1);

  absl::Cord output;
  absl::Status status = Decode(CordFromString(truncated), &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeTruncatedGzipDataReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(true));

  std::string truncated = CordToString(compressed);
  ASSERT_GT(truncated.size(), 1u);
  truncated.resize(truncated.size() - 1);

  absl::Cord output;
  absl::Status status = Decode(CordFromString(truncated), &output, true);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeZlibDataAsGzipReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  absl::Cord output;
  absl::Status status = Decode(compressed, &output, true);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeGzipDataAsZlibReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(true));

  absl::Cord output;
  absl::Status status = Decode(compressed, &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeCorruptedZlibDataReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  std::string corrupted = CordToString(compressed);
  ASSERT_GT(corrupted.size(), 2u);
  corrupted[corrupted.size() / 2] ^= 0x7F;

  absl::Cord output;
  absl::Status status = Decode(CordFromString(corrupted), &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Invalid_DecodeCompressedDataWithTrailingBytesReturnsInvalidArgument) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  compressed.Append("trailing");

  absl::Cord output;
  absl::Status status = Decode(compressed, &output, false);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(ZlibCompressionTest, Edge_EncodeAppendsToExistingOutputCord) {
  absl::Cord compressed;
  compressed.Append("prefix");

  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  EXPECT_FALSE(compressed.empty());
  EXPECT_NE(CordToString(compressed), "prefix");
}

TEST(ZlibCompressionTest, Edge_DecodeAppendsToExistingOutputCord) {
  absl::Cord compressed;
  Encode(CordFromString("payload"), &compressed, DefaultOptions(false));

  absl::Cord output;
  output.Append("prefix");

  absl::Status status = Decode(compressed, &output, false);

  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(CordToString(output), "prefixpayload");
}

TEST(ZlibCompressionTest, Edge_MultipleIndependentEncodesProduceDecodableOutputs) {
  absl::Cord compressed_a;
  absl::Cord compressed_b;

  Encode(CordFromString("first payload"), &compressed_a, DefaultOptions(false));
  Encode(CordFromString("second payload"), &compressed_b, DefaultOptions(false));

  absl::Cord output_a;
  absl::Cord output_b;

  ASSERT_TRUE(Decode(compressed_a, &output_a, false).ok());
  ASSERT_TRUE(Decode(compressed_b, &output_b, false).ok());

  EXPECT_EQ(CordToString(output_a), "first payload");
  EXPECT_EQ(CordToString(output_b), "second payload");
}

TEST(ZlibCompressionTest, Edge_RepeatedRoundTripsRemainStable) {
  std::string input = "stable repeated round trip payload";

  for (int i = 0; i < 10; ++i) {
    RoundTrip(input, DefaultOptions(false));
  }
}

}  // namespace
}  // namespace zlib
}  // namespace tensorstore