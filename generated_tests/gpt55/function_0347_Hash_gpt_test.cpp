#include "util/hash.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace leveldb {
namespace {

uint32_t ReferenceHash(const char* data, size_t n, uint32_t seed) {
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = seed ^ static_cast<uint32_t>(n * m);

  while (data + 4 <= limit) {
    uint32_t w = 0;
    std::memcpy(&w, data, sizeof(w));
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  switch (limit - data) {
    case 3:
      h += static_cast<uint8_t>(data[2]) << 16;
      [[fallthrough]];
    case 2:
      h += static_cast<uint8_t>(data[1]) << 8;
      [[fallthrough]];
    case 1:
      h += static_cast<uint8_t>(data[0]);
      h *= m;
      h ^= (h >> r);
      break;
    default:
      break;
  }

  return h;
}

TEST(HashTest, EmptyInputReturnsSeed) {
  EXPECT_EQ(Hash("", 0, 0), 0u);
  EXPECT_EQ(Hash("", 0, 12345), 12345u);
  EXPECT_EQ(Hash("", 0, UINT32_MAX), UINT32_MAX);
}

TEST(HashTest, OneByteInput) {
  const std::string data = "a";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, TwoByteInput) {
  const std::string data = "ab";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, ThreeByteInput) {
  const std::string data = "abc";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, FourByteInputProcessesFullWord) {
  const std::string data = "abcd";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, FiveByteInputProcessesWordAndRemainder) {
  const std::string data = "abcde";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, SevenByteInputProcessesThreeByteRemainder) {
  const std::string data = "abcdefg";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, EightByteInputProcessesTwoFullWords) {
  const std::string data = "abcdefgh";

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, DifferentSeedsProduceExpectedDifferentHashes) {
  const std::string data = "leveldb";

  const uint32_t hash_seed_0 = Hash(data.data(), data.size(), 0);
  const uint32_t hash_seed_1 = Hash(data.data(), data.size(), 1);
  const uint32_t hash_seed_max = Hash(data.data(), data.size(), UINT32_MAX);

  EXPECT_EQ(hash_seed_0, ReferenceHash(data.data(), data.size(), 0));
  EXPECT_EQ(hash_seed_1, ReferenceHash(data.data(), data.size(), 1));
  EXPECT_EQ(hash_seed_max, ReferenceHash(data.data(), data.size(), UINT32_MAX));

  EXPECT_NE(hash_seed_0, hash_seed_1);
  EXPECT_NE(hash_seed_0, hash_seed_max);
}

TEST(HashTest, SameInputAndSeedAreDeterministic) {
  const std::string data = "same input";

  EXPECT_EQ(Hash(data.data(), data.size(), 123),
            Hash(data.data(), data.size(), 123));
}

TEST(HashTest, DifferentInputsProduceDifferentHashes) {
  const std::string data1 = "same input";
  const std::string data2 = "same inpuu";

  EXPECT_NE(Hash(data1.data(), data1.size(), 123),
            Hash(data2.data(), data2.size(), 123));
}

TEST(HashTest, EmbeddedNullBytesAreIncluded) {
  const std::string data("ab\0cd", 5);

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));

  EXPECT_NE(Hash(data.data(), data.size(), 0),
            Hash("ab", 2, 0));
}

TEST(HashTest, HighBitBytesAreHandledAsUnsignedRemainderBytes) {
  const char raw[] = {
      static_cast<char>(0x80),
      static_cast<char>(0xFF),
      static_cast<char>(0x7F)
  };

  EXPECT_EQ(Hash(raw, sizeof(raw), 0),
            ReferenceHash(raw, sizeof(raw), 0));
}

TEST(HashTest, BinaryDataWithAllByteValues) {
  std::string data;
  for (int i = 0; i <= 255; ++i) {
    data.push_back(static_cast<char>(i));
  }

  EXPECT_EQ(Hash(data.data(), data.size(), 0),
            ReferenceHash(data.data(), data.size(), 0));
}

TEST(HashTest, LongInput) {
  std::string data;
  for (int i = 0; i < 4096; ++i) {
    data.push_back(static_cast<char>(i & 0xFF));
  }

  EXPECT_EQ(Hash(data.data(), data.size(), 987654321),
            ReferenceHash(data.data(), data.size(), 987654321));
}

TEST(HashTest, PrefixLengthMatters) {
  const std::string data = "abcdef";

  EXPECT_EQ(Hash(data.data(), 3, 42),
            ReferenceHash(data.data(), 3, 42));
  EXPECT_NE(Hash(data.data(), 3, 42),
            Hash(data.data(), data.size(), 42));
}

TEST(HashTest, HandlesUnalignedInputPointer) {
  const std::string storage = "xabcdefgh";
  const char* unaligned = storage.data() + 1;

  EXPECT_EQ(Hash(unaligned, 8, 99),
            ReferenceHash(unaligned, 8, 99));
}

}  // namespace
}  // namespace leveldb