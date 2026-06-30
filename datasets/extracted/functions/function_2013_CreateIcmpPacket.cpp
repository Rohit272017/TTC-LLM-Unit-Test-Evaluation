#include "quiche/quic/qbone/platform/icmp_packet.h"
#include <netinet/ip6.h>
#include <algorithm>
#include "absl/strings/string_view.h"
#include "quiche/quic/core/internet_checksum.h"
#include "quiche/common/quiche_callbacks.h"
#include "quiche/common/quiche_endian.h"
namespace quic {
namespace {
constexpr size_t kIPv6AddressSize = sizeof(in6_addr);
constexpr size_t kIPv6HeaderSize = sizeof(ip6_hdr);
constexpr size_t kICMPv6HeaderSize = sizeof(icmp6_hdr);
constexpr size_t kIPv6MinPacketSize = 1280;
constexpr size_t kIcmpTtl = 255;
constexpr size_t kICMPv6BodyMaxSize =
    kIPv6MinPacketSize - kIPv6HeaderSize - kICMPv6HeaderSize;
struct ICMPv6Packet {
  ip6_hdr ip_header;
  icmp6_hdr icmp_header;
  uint8_t body[kICMPv6BodyMaxSize];
};
struct IPv6PseudoHeader {
  uint32_t payload_size{};
  uint8_t zeros[3] = {0, 0, 0};
  uint8_t next_header = IPPROTO_ICMPV6;
};
}  
void CreateIcmpPacket(in6_addr src, in6_addr dst, const icmp6_hdr& icmp_header,
                      absl::string_view body,
                      quiche::UnretainedCallback<void(absl::string_view)> cb) {
  const size_t body_size = std::min(body.size(), kICMPv6BodyMaxSize);
  const size_t payload_size = kICMPv6HeaderSize + body_size;
  ICMPv6Packet icmp_packet{};
  icmp_packet.ip_header.ip6_vfc = 0x6 << 4;
  icmp_packet.ip_header.ip6_plen =
      quiche::QuicheEndian::HostToNet16(payload_size);
  icmp_packet.ip_header.ip6_nxt = IPPROTO_ICMPV6;
  icmp_packet.ip_header.ip6_hops = kIcmpTtl;
  icmp_packet.ip_header.ip6_src = src;
  icmp_packet.ip_header.ip6_dst = dst;
  icmp_packet.icmp_header = icmp_header;
  icmp_packet.icmp_header.icmp6_cksum = 0;
  IPv6PseudoHeader pseudo_header{};
  pseudo_header.payload_size = quiche::QuicheEndian::HostToNet32(payload_size);
  InternetChecksum checksum;
  checksum.Update(icmp_packet.ip_header.ip6_src.s6_addr, kIPv6AddressSize);
  checksum.Update(icmp_packet.ip_header.ip6_dst.s6_addr, kIPv6AddressSize);
  checksum.Update(reinterpret_cast<char*>(&pseudo_header),
                  sizeof(pseudo_header));
  checksum.Update(reinterpret_cast<const char*>(&icmp_packet.icmp_header),
                  sizeof(icmp_packet.icmp_header));
  checksum.Update(body.data(), body_size);
  icmp_packet.icmp_header.icmp6_cksum = checksum.Value();
  memcpy(icmp_packet.body, body.data(), body_size);
  const char* packet = reinterpret_cast<char*>(&icmp_packet);
  const size_t packet_size = offsetof(ICMPv6Packet, body) + body_size;
  cb(absl::string_view(packet, packet_size));
}
}  