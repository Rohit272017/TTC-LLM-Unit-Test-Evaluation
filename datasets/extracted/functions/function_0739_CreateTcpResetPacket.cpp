#include "quiche/quic/qbone/platform/tcp_packet.h"
#include <netinet/ip6.h>
#include "absl/base/optimization.h"
#include "absl/strings/string_view.h"
#include "quiche/quic/core/internet_checksum.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/common/quiche_callbacks.h"
#include "quiche/common/quiche_endian.h"
namespace quic {
namespace {
constexpr size_t kIPv6AddressSize = sizeof(in6_addr);
constexpr size_t kTcpTtl = 64;
struct TCPv6Packet {
  ip6_hdr ip_header;
  tcphdr tcp_header;
};
struct TCPv6PseudoHeader {
  uint32_t payload_size{};
  uint8_t zeros[3] = {0, 0, 0};
  uint8_t next_header = IPPROTO_TCP;
};
}  
void CreateTcpResetPacket(
    absl::string_view original_packet,
    quiche::UnretainedCallback<void(absl::string_view)> cb) {
  if (ABSL_PREDICT_FALSE(original_packet.size() < sizeof(ip6_hdr))) {
    return;
  }
  auto* ip6_header = reinterpret_cast<const ip6_hdr*>(original_packet.data());
  if (ABSL_PREDICT_FALSE(ip6_header->ip6_vfc >> 4 != 6)) {
    return;
  }
  if (ABSL_PREDICT_FALSE(ip6_header->ip6_nxt != IPPROTO_TCP)) {
    return;
  }
  if (ABSL_PREDICT_FALSE(quiche::QuicheEndian::NetToHost16(
                             ip6_header->ip6_plen) < sizeof(tcphdr))) {
    return;
  }
  auto* tcp_header = reinterpret_cast<const tcphdr*>(ip6_header + 1);
  TCPv6Packet tcp_packet{};
  const size_t payload_size = sizeof(tcphdr);
  tcp_packet.ip_header.ip6_vfc = 0x6 << 4;
  tcp_packet.ip_header.ip6_plen =
      quiche::QuicheEndian::HostToNet16(payload_size);
  tcp_packet.ip_header.ip6_nxt = IPPROTO_TCP;
  tcp_packet.ip_header.ip6_hops = kTcpTtl;
  tcp_packet.ip_header.ip6_src = ip6_header->ip6_dst;
  tcp_packet.ip_header.ip6_dst = ip6_header->ip6_src;
  tcp_packet.tcp_header.dest = tcp_header->source;
  tcp_packet.tcp_header.source = tcp_header->dest;
  tcp_packet.tcp_header.doff = sizeof(tcphdr) >> 2;
  tcp_packet.tcp_header.check = 0;
  tcp_packet.tcp_header.rst = 1;
  if (tcp_header->ack) {
    tcp_packet.tcp_header.seq = tcp_header->ack_seq;
  } else {
    tcp_packet.tcp_header.ack = 1;
    tcp_packet.tcp_header.seq = 0;
    tcp_packet.tcp_header.ack_seq = quiche::QuicheEndian::HostToNet32(
        quiche::QuicheEndian::NetToHost32(tcp_header->seq) + 1);
  }
  TCPv6PseudoHeader pseudo_header{};
  pseudo_header.payload_size = quiche::QuicheEndian::HostToNet32(payload_size);
  InternetChecksum checksum;
  checksum.Update(tcp_packet.ip_header.ip6_src.s6_addr, kIPv6AddressSize);
  checksum.Update(tcp_packet.ip_header.ip6_dst.s6_addr, kIPv6AddressSize);
  checksum.Update(reinterpret_cast<char*>(&pseudo_header),
                  sizeof(pseudo_header));
  checksum.Update(reinterpret_cast<const char*>(&tcp_packet.tcp_header),
                  sizeof(tcp_packet.tcp_header));
  tcp_packet.tcp_header.check = checksum.Value();
  const char* packet = reinterpret_cast<char*>(&tcp_packet);
  cb(absl::string_view(packet, sizeof(tcp_packet)));
}
}  