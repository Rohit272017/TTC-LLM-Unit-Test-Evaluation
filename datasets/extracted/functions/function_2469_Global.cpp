#include "xla/service/custom_call_target_registry.h"
#include <cstdlib>
#include <iostream>
#include <mutex>  
#include <string>
#include <unordered_map>
#include <utility>
namespace xla {
CustomCallTargetRegistry* CustomCallTargetRegistry::Global() {
  static auto* registry = new CustomCallTargetRegistry;
  return registry;
}
void CustomCallTargetRegistry::Register(const std::string& symbol,
                                        void* address,
                                        const std::string& platform) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto [it, inserted] =
      registered_symbols_.insert({{symbol, platform}, address});
  if (!inserted && it->second != address) {
    std::cerr << "Duplicate custom call registration detected for symbol \""
              << symbol << "\" with different addresses " << address
              << "(current) and " << it->second << " (previous) on platform "
              << platform
              << "Rejecting the registration to avoid confusion about which "
                 "symbol would actually get used at runtime.\n";
    std::exit(1);
  }
}
void* CustomCallTargetRegistry::Lookup(const std::string& symbol,
                                       const std::string& platform) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registered_symbols_.find(std::make_pair(symbol, platform));
  return it == registered_symbols_.end() ? nullptr : it->second;
}
std::unordered_map<std::string, void*>
CustomCallTargetRegistry::registered_symbols(
    const std::string& platform) const {
  std::unordered_map<std::string, void*> calls;
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& [metadata, address] : registered_symbols_) {
    if (metadata.second == platform) {
      calls[metadata.first] = address;
    }
  }
  return calls;
}
}  