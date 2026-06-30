#include "quiche/quic/test_tools/test_ip_packets.h"
#include <cstdint>
#include <limits>
#include <string>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "quiche/quic/core/internet_checksum.h"
#include "quiche/quic/platform/api/quic_socket_address.h"
#include "quiche/common/platform/api/quiche_logging.h"
#include "quiche/common/quiche_data_writer.h"
#include "quiche/common/quiche_endian.h"
#include "quiche/common/quiche_ip_address.h"
#include "quiche/common/quiche_ip_address_family.h"
#if defined(__linux__)
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/udp.h>
#endif
namespace quic::test {
namespace {
constexpr uint16_t kIpv4HeaderSize = 20;
constexpr uint16_t kIpv6HeaderSize = 40;
constexpr uint16_t kUdpHeaderSize = 8;
constexpr uint8_t kUdpProtocol = 0x11;
#if defined(__linux__)
static_assert(kIpv4HeaderSize == sizeof(iphdr));
static_assert(kIpv6HeaderSize == sizeof(ip6_hdr));
static_assert(kUdpHeaderSize == sizeof(udphdr));
static_assert(kUdpProtocol == IPPROTO_UDP);
#endif
std::string CreateIpv4Header(int payload_length,
                             quiche::QuicheIpAddress source_address,
                             quiche::QuicheIpAddress destination_address,
                             uint8_t protocol) {
  QUICHE_CHECK_GT(payload_length, 0);
  QUICHE_CHECK_LE(payload_length,
                  std::numeric_limits<uint16_t>::max() - kIpv4HeaderSize);
  QUICHE_CHECK(source_address.address_family() ==
               quiche::IpAddressFamily::IP_V4);
  QUICHE_CHECK(destination_address.address_family() ==
               quiche::IpAddressFamily::IP_V4);
  std::string header(kIpv4HeaderSize, '\0');
  quiche::QuicheDataWriter header_writer(header.size(), header.data());
  header_writer.WriteUInt8(0x45);  
  header_writer.WriteUInt8(0x00);  
  header_writer.WriteUInt16(kIpv4HeaderSize + payload_length);  
  header_writer.WriteUInt16(0x0000);  
  header_writer.WriteUInt16(0x0000);  
  header_writer.WriteUInt8(64);       
  header_writer.WriteUInt8(protocol);
  header_writer.WriteUInt16(0x0000);  
  header_writer.WriteStringPiece(source_address.ToPackedString());
  header_writer.WriteStringPiece(destination_address.ToPackedString());
  QUICHE_CHECK_EQ(header_writer.remaining(), 0u);
  return header;
}
std::string CreateIpv6Header(int payload_length,
                             quiche::QuicheIpAddress source_address,
                             quiche::QuicheIpAddress destination_address,
                             uint8_t next_header) {
  QUICHE_CHECK_GT(payload_length, 0);
  QUICHE_CHECK_LE(payload_length, std::numeric_limits<uint16_t>::max());
  QUICHE_CHECK(source_address.address_family() ==
               quiche::IpAddressFamily::IP_V6);
  QUICHE_CHECK(destination_address.address_family() ==
               quiche::IpAddressFamily::IP_V6);
  std::string header(kIpv6HeaderSize, '\0');
  quiche::QuicheDataWriter header_writer(header.size(), header.data());
  header_writer.WriteUInt32(0x60000000);
  header_writer.WriteUInt16(payload_length);
  header_writer.WriteUInt8(next_header);
  header_writer.WriteUInt8(64);  
  header_writer.WriteStringPiece(source_address.ToPackedString());
  header_writer.WriteStringPiece(destination_address.ToPackedString());
  QUICHE_CHECK_EQ(header_writer.remaining(), 0u);
  return header;
}
}  
std::string CreateIpPacket(const quiche::QuicheIpAddress& source_address,
                           const quiche::QuicheIpAddress& destination_address,
                           absl::string_view payload,
                           IpPacketPayloadType payload_type) {
  QUICHE_CHECK(source_address.address_family() ==
               destination_address.address_family());
  uint8_t payload_protocol;
  switch (payload_type) {
    case IpPacketPayloadType::kUdp:
      payload_protocol = kUdpProtocol;
      break;
    default:
      QUICHE_NOTREACHED();
      return "";
  }
  std::string header;
  switch (source_address.address_family()) {
    case quiche::IpAddressFamily::IP_V4:
      header = CreateIpv4Header(payload.size(), source_address,
                                destination_address, payload_protocol);
      break;
    case quiche::IpAddressFamily::IP_V6:
      header = CreateIpv6Header(payload.size(), source_address,
                                destination_address, payload_protocol);
      break;
    default:
      QUICHE_NOTREACHED();
      return "";
  }
  return absl::StrCat(header, payload);
}
std::string CreateUdpPacket(const QuicSocketAddress& source_address,
                            const QuicSocketAddress& destination_address,
                            absl::string_view payload) {
  QUICHE_CHECK(source_address.host().address_family() ==
               destination_address.host().address_family());
  QUICHE_CHECK(!payload.empty());
  QUICHE_CHECK_LE(payload.size(),
                  static_cast<uint16_t>(std::numeric_limits<uint16_t>::max() -
                                        kUdpHeaderSize));
  std::string header(kUdpHeaderSize, '\0');
  quiche::QuicheDataWriter header_writer(header.size(), header.data());
  header_writer.WriteUInt16(source_address.port());
  header_writer.WriteUInt16(destination_address.port());
  header_writer.WriteUInt16(kUdpHeaderSize + payload.size());
  InternetChecksum checksum;
  switch (source_address.host().address_family()) {
    case quiche::IpAddressFamily::IP_V4: {
      checksum.Update(source_address.host().ToPackedString());
      checksum.Update(destination_address.host().ToPackedString());
      uint8_t protocol[] = {0x00, kUdpProtocol};
      checksum.Update(protocol, sizeof(protocol));
      uint16_t udp_length =
          quiche::QuicheEndian::HostToNet16(kUdpHeaderSize + payload.size());
      checksum.Update(reinterpret_cast<uint8_t*>(&udp_length),
                      sizeof(udp_length));
      break;
    }
    case quiche::IpAddressFamily::IP_V6: {
      checksum.Update(source_address.host().ToPackedString());
      checksum.Update(destination_address.host().ToPackedString());
      uint32_t udp_length =
          quiche::QuicheEndian::HostToNet32(kUdpHeaderSize + payload.size());
      checksum.Update(reinterpret_cast<uint8_t*>(&udp_length),
                      sizeof(udp_length));
      uint8_t protocol[] = {0x00, 0x00, 0x00, kUdpProtocol};
      checksum.Update(protocol, sizeof(protocol));
      break;
    }
    default:
      QUICHE_NOTREACHED();
      return "";
  }
  checksum.Update(header.data(), header.size());
  checksum.Update(payload.data(), payload.size());
  uint16_t checksum_val = checksum.Value();
  header_writer.WriteBytes(&checksum_val, sizeof(checksum_val));
  QUICHE_CHECK_EQ(header_writer.remaining(), 0u);
  return absl::StrCat(header, payload);
}
}  