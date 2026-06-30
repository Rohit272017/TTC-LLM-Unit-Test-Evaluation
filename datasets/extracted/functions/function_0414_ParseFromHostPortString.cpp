#include "quiche/quic/core/quic_server_id.h"
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "quiche/common/platform/api/quiche_googleurl.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace quic {
std::optional<QuicServerId> QuicServerId::ParseFromHostPortString(
    absl::string_view host_port_string) {
  url::Component username_component;
  url::Component password_component;
  url::Component host_component;
  url::Component port_component;
  url::ParseAuthority(host_port_string.data(),
                      url::Component(0, host_port_string.size()),
                      &username_component, &password_component, &host_component,
                      &port_component);
  if (username_component.is_valid() || password_component.is_valid() ||
      !host_component.is_nonempty() || !port_component.is_nonempty()) {
    QUICHE_DVLOG(1) << "QuicServerId could not be parsed: " << host_port_string;
    return std::nullopt;
  }
  std::string hostname(host_port_string.data() + host_component.begin,
                       host_component.len);
  int parsed_port_number =
      url::ParsePort(host_port_string.data(), port_component);
  if (parsed_port_number <= 0) {
    QUICHE_DVLOG(1)
        << "Port could not be parsed while parsing QuicServerId from: "
        << host_port_string;
    return std::nullopt;
  }
  QUICHE_DCHECK_LE(parsed_port_number, std::numeric_limits<uint16_t>::max());
  return QuicServerId(std::move(hostname),
                      static_cast<uint16_t>(parsed_port_number));
}
QuicServerId::QuicServerId() : QuicServerId("", 0) {}
QuicServerId::QuicServerId(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port) {}
QuicServerId::~QuicServerId() {}
bool QuicServerId::operator<(const QuicServerId& other) const {
  return std::tie(port_, host_) < std::tie(other.port_, other.host_);
}
bool QuicServerId::operator==(const QuicServerId& other) const {
  return host_ == other.host_ && port_ == other.port_;
}
bool QuicServerId::operator!=(const QuicServerId& other) const {
  return !(*this == other);
}
std::string QuicServerId::ToHostPortString() const {
  return absl::StrCat(GetHostWithIpv6Brackets(), ":", port_);
}
absl::string_view QuicServerId::GetHostWithoutIpv6Brackets() const {
  if (host_.length() > 2 && host_.front() == '[' && host_.back() == ']') {
    return absl::string_view(host_.data() + 1, host_.length() - 2);
  } else {
    return host_;
  }
}
std::string QuicServerId::GetHostWithIpv6Brackets() const {
  if (!absl::StrContains(host_, ':') || host_.length() <= 2 ||
      (host_.front() == '[' && host_.back() == ']')) {
    return host_;
  } else {
    return absl::StrCat("[", host_, "]");
  }
}
}  