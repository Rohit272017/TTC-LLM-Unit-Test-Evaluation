#include "quiche/quic/core/quic_socket_address_coder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "quiche/quic/core/quic_socket_address.h"
#include "quiche/quic/platform/api/quic_ip_address.h"

namespace quic {
namespace {

constexpr uint16_t kExpectedIPv4Family = 2;
constexpr uint16_t kExpectedIPv6Family = 10;

QuicIpAddress MakeIpAddress(const std::string& ip) {
  QuicIpAddress address;
  EXPECT_TRUE(address.FromString(ip));
  return address;
}

QuicSocketAddress MakeSocketAddress(const std::string& ip, uint16_t port) {
  return QuicSocketAddress(MakeIpAddress(ip), port);
}

uint16_t ReadUint16(const std::string& data, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

std::string WithFamily(uint16_t family) {
  std::string data;
  data.append(reinterpret_cast<const char*>(&family), sizeof(family));
  return data;
}

TEST(QuicSocketAddressCoderTest, ECP_EncodeValidIPv4Address) {
  QuicSocketAddress address = MakeSocketAddress("127.0.0.1", 443);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv4AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 0), kExpectedIPv4Family);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv4AddressSize), 443);
}

TEST(QuicSocketAddressCoderTest, ECP_EncodeValidIPv6Address) {
  QuicSocketAddress address = MakeSocketAddress("2001:db8::1", 8443);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv6AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 0), kExpectedIPv6Family);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv6AddressSize), 8443);
}

TEST(QuicSocketAddressCoderTest, BVA_EncodeIPv4PortZero) {
  QuicSocketAddress address = MakeSocketAddress("192.168.1.1", 0);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv4AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv4AddressSize), 0);
}

TEST(QuicSocketAddressCoderTest, BVA_EncodeIPv4PortMax) {
  QuicSocketAddress address = MakeSocketAddress("192.168.1.1", 65535);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv4AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv4AddressSize), 65535);
}

TEST(QuicSocketAddressCoderTest, BVA_EncodeIPv6PortZero) {
  QuicSocketAddress address = MakeSocketAddress("::1", 0);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv6AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv6AddressSize), 0);
}

TEST(QuicSocketAddressCoderTest, BVA_EncodeIPv6PortMax) {
  QuicSocketAddress address = MakeSocketAddress("::1", 65535);
  QuicSocketAddressCoder coder(address);

  std::string encoded = coder.Encode();

  ASSERT_EQ(encoded.size(), 2u + QuicIpAddress::kIPv6AddressSize + 2u);
  EXPECT_EQ(ReadUint16(encoded, 2 + QuicIpAddress::kIPv6AddressSize), 65535);
}

TEST(QuicSocketAddressCoderTest, Invalid_EncodeUninitializedAddressReturnsEmpty) {
  QuicSocketAddressCoder coder;

  std::string encoded = coder.Encode();

  EXPECT_TRUE(encoded.empty());
}

TEST(QuicSocketAddressCoderTest, ECP_DecodeValidIPv4Address) {
  QuicSocketAddress original = MakeSocketAddress("8.8.8.8", 53);
  QuicSocketAddressCoder encoder(original);
  std::string encoded = encoder.Encode();

  QuicSocketAddressCoder decoder;

  EXPECT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().host(), original.host());
  EXPECT_EQ(decoder.address().port(), original.port());
}

TEST(QuicSocketAddressCoderTest, ECP_DecodeValidIPv6Address) {
  QuicSocketAddress original = MakeSocketAddress("2001:4860:4860::8888", 443);
  QuicSocketAddressCoder encoder(original);
  std::string encoded = encoder.Encode();

  QuicSocketAddressCoder decoder;

  EXPECT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().host(), original.host());
  EXPECT_EQ(decoder.address().port(), original.port());
}

TEST(QuicSocketAddressCoderTest, BVA_DecodeIPv4PortZero) {
  QuicSocketAddress original = MakeSocketAddress("10.0.0.1", 0);
  std::string encoded = QuicSocketAddressCoder(original).Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().port(), 0);
  EXPECT_EQ(decoder.address().host(), original.host());
}

TEST(QuicSocketAddressCoderTest, BVA_DecodeIPv4PortMax) {
  QuicSocketAddress original = MakeSocketAddress("10.0.0.1", 65535);
  std::string encoded = QuicSocketAddressCoder(original).Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().port(), 65535);
  EXPECT_EQ(decoder.address().host(), original.host());
}

TEST(QuicSocketAddressCoderTest, BVA_DecodeIPv6PortZero) {
  QuicSocketAddress original = MakeSocketAddress("::1", 0);
  std::string encoded = QuicSocketAddressCoder(original).Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().port(), 0);
  EXPECT_EQ(decoder.address().host(), original.host());
}

TEST(QuicSocketAddressCoderTest, BVA_DecodeIPv6PortMax) {
  QuicSocketAddress original = MakeSocketAddress("::1", 65535);
  std::string encoded = QuicSocketAddressCoder(original).Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().port(), 65535);
  EXPECT_EQ(decoder.address().host(), original.host());
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeEmptyBufferFails) {
  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode("", 0));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeOneByteBufferFails) {
  const char data[1] = {0};
  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(data, sizeof(data)));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeUnknownAddressFamilyFails) {
  std::string data = WithFamily(999);

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(data.data(), data.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv4MissingAddressBytesFails) {
  std::string data = WithFamily(kExpectedIPv4Family);
  data.append(QuicIpAddress::kIPv4AddressSize - 1, '\0');

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(data.data(), data.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv6MissingAddressBytesFails) {
  std::string data = WithFamily(kExpectedIPv6Family);
  data.append(QuicIpAddress::kIPv6AddressSize - 1, '\0');

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(data.data(), data.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv4MissingPortFails) {
  QuicSocketAddress original = MakeSocketAddress("1.2.3.4", 80);
  std::string encoded = QuicSocketAddressCoder(original).Encode();
  encoded.resize(encoded.size() - 1);

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(encoded.data(), encoded.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv6MissingPortFails) {
  QuicSocketAddress original = MakeSocketAddress("2001:db8::1", 80);
  std::string encoded = QuicSocketAddressCoder(original).Encode();
  encoded.resize(encoded.size() - 1);

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(encoded.data(), encoded.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv4ExtraTrailingByteFails) {
  QuicSocketAddress original = MakeSocketAddress("1.2.3.4", 80);
  std::string encoded = QuicSocketAddressCoder(original).Encode();
  encoded.push_back('\0');

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(encoded.data(), encoded.size()));
}

TEST(QuicSocketAddressCoderTest, Invalid_DecodeIPv6ExtraTrailingByteFails) {
  QuicSocketAddress original = MakeSocketAddress("2001:db8::1", 80);
  std::string encoded = QuicSocketAddressCoder(original).Encode();
  encoded.push_back('\0');

  QuicSocketAddressCoder decoder;

  EXPECT_FALSE(decoder.Decode(encoded.data(), encoded.size()));
}

TEST(QuicSocketAddressCoderTest, Edge_DecodeFailureDoesNotOverwriteExistingAddress) {
  QuicSocketAddress existing = MakeSocketAddress("192.0.2.1", 1234);
  QuicSocketAddressCoder decoder(existing);

  std::string invalid = WithFamily(999);

  EXPECT_FALSE(decoder.Decode(invalid.data(), invalid.size()));
  EXPECT_EQ(decoder.address().host(), existing.host());
  EXPECT_EQ(decoder.address().port(), existing.port());
}

TEST(QuicSocketAddressCoderTest, Edge_RoundTripIPv4AnyAddress) {
  QuicSocketAddress original = MakeSocketAddress("0.0.0.0", 8080);
  QuicSocketAddressCoder encoder(original);
  std::string encoded = encoder.Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().host(), original.host());
  EXPECT_EQ(decoder.address().port(), original.port());
}

TEST(QuicSocketAddressCoderTest, Edge_RoundTripIPv4BroadcastAddress) {
  QuicSocketAddress original = MakeSocketAddress("255.255.255.255", 8080);
  QuicSocketAddressCoder encoder(original);
  std::string encoded = encoder.Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().host(), original.host());
  EXPECT_EQ(decoder.address().port(), original.port());
}

TEST(QuicSocketAddressCoderTest, Edge_RoundTripIPv6UnspecifiedAddress) {
  QuicSocketAddress original = MakeSocketAddress("::", 8080);
  QuicSocketAddressCoder encoder(original);
  std::string encoded = encoder.Encode();

  QuicSocketAddressCoder decoder;

  ASSERT_TRUE(decoder.Decode(encoded.data(), encoded.size()));
  EXPECT_EQ(decoder.address().host(), original.host());
  EXPECT_EQ(decoder.address().port(), original.port());
}

}  // namespace
}  // namespace quic