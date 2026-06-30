#ifndef QUICHE_QUIC_LOAD_BALANCER_LOAD_BALANCER_SERVER_ID_MAP_H_
#define QUICHE_QUIC_LOAD_BALANCER_LOAD_BALANCER_SERVER_ID_MAP_H_
#include <cstdint>
#include <memory>
#include <optional>
#include "absl/container/flat_hash_map.h"
#include "quiche/quic/load_balancer/load_balancer_server_id.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
namespace quic {
template <typename T>
class QUIC_EXPORT_PRIVATE LoadBalancerServerIdMap {
 public:
  static std::shared_ptr<LoadBalancerServerIdMap> Create(uint8_t server_id_len);
  std::optional<const T> Lookup(LoadBalancerServerId server_id) const;
  const T* LookupNoCopy(LoadBalancerServerId server_id) const;
  void AddOrReplace(LoadBalancerServerId server_id, T value);
  void Erase(const LoadBalancerServerId server_id) {
    server_id_table_.erase(server_id);
  }
  uint8_t server_id_len() const { return server_id_len_; }
 private:
  LoadBalancerServerIdMap(uint8_t server_id_len)
      : server_id_len_(server_id_len) {}
  const uint8_t server_id_len_;  
  absl::flat_hash_map<LoadBalancerServerId, T> server_id_table_;
};
template <typename T>
std::shared_ptr<LoadBalancerServerIdMap<T>> LoadBalancerServerIdMap<T>::Create(
    const uint8_t server_id_len) {
  if (server_id_len == 0 || server_id_len > kLoadBalancerMaxServerIdLen) {
    QUIC_BUG(quic_bug_434893339_01)
        << "Tried to configure map with server ID length "
        << static_cast<int>(server_id_len);
    return nullptr;
  }
  return std::make_shared<LoadBalancerServerIdMap<T>>(
      LoadBalancerServerIdMap(server_id_len));
}
template <typename T>
std::optional<const T> LoadBalancerServerIdMap<T>::Lookup(
    const LoadBalancerServerId server_id) const {
  if (server_id.length() != server_id_len_) {
    QUIC_BUG(quic_bug_434893339_02)
        << "Lookup with a " << static_cast<int>(server_id.length())
        << " byte server ID, map requires " << static_cast<int>(server_id_len_);
    return std::optional<T>();
  }
  auto it = server_id_table_.find(server_id);
  return (it != server_id_table_.end()) ? it->second : std::optional<const T>();
}
template <typename T>
const T* LoadBalancerServerIdMap<T>::LookupNoCopy(
    const LoadBalancerServerId server_id) const {
  if (server_id.length() != server_id_len_) {
    QUIC_BUG(quic_bug_434893339_02)
        << "Lookup with a " << static_cast<int>(server_id.length())
        << " byte server ID, map requires " << static_cast<int>(server_id_len_);
    return nullptr;
  }
  auto it = server_id_table_.find(server_id);
  return (it != server_id_table_.end()) ? &it->second : nullptr;
}
template <typename T>
void LoadBalancerServerIdMap<T>::AddOrReplace(
    const LoadBalancerServerId server_id, T value) {
  if (server_id.length() == server_id_len_) {
    server_id_table_[server_id] = value;
  } else {
    QUIC_BUG(quic_bug_434893339_03)
        << "Server ID of " << static_cast<int>(server_id.length())
        << " bytes; this map requires " << static_cast<int>(server_id_len_);
  }
}
}  
#endif  