#include "xla/service/xla_debug_info_manager.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
namespace xla {
void XlaDebugInfoManager::RegisterModule(
    std::shared_ptr<const HloModule> hlo_module,
    BufferAssignmentProto buffer_assignment) {
  CHECK(hlo_module != nullptr);
  absl::MutexLock lock(&mutex_);
  auto result = modules_.try_emplace(hlo_module->unique_id());
  CHECK(result.second);
  XlaModuleEntry& m = result.first->second;
  m.hlo_module = std::move(hlo_module);
  m.buffer_assignment = std::move(buffer_assignment);
  m.active = true;
}
void XlaDebugInfoManager::UnregisterModule(ModuleIdentifier module_id) {
  absl::MutexLock lock(&mutex_);
  auto it = modules_.find(module_id);
  CHECK(it != modules_.end());
  if (!tracing_active_) {
    modules_.erase(it);
  } else {
    XlaModuleEntry& m = it->second;
    m.active = false;
  }
}
void XlaDebugInfoManager::StartTracing() {
  absl::MutexLock lock(&mutex_);
  tracing_active_ = true;
}
void XlaDebugInfoManager::StopTracing(
    std::vector<std::unique_ptr<HloProto>>* module_debug_info) {
  std::vector<XlaModuleEntry> modules_to_serialize;
  {
    absl::MutexLock lock(&mutex_);
    if (!tracing_active_) return;
    tracing_active_ = false;
    modules_to_serialize.reserve(modules_.size());
    for (auto it = modules_.begin(); it != modules_.end();) {
      auto& m = it->second;
      auto cur_it = it++;
      if (!m.active) {
        modules_to_serialize.emplace_back(std::move(m));
        modules_.erase(cur_it);
      } else {
        modules_to_serialize.emplace_back(m);
      }
    }
  }
  if (module_debug_info) {
    module_debug_info->clear();
    for (const auto& m : modules_to_serialize) {
      auto hlo_proto = std::make_unique<HloProto>(MakeHloProto(*m.hlo_module));
      *hlo_proto->mutable_buffer_assignment() = m.buffer_assignment;
      module_debug_info->emplace_back(std::move(hlo_proto));
    }
  }
}
bool XlaDebugInfoManager::TracksModule(ModuleIdentifier module_id) const {
  absl::MutexLock lock(&mutex_);
  return modules_.find(module_id) != modules_.end();
}
}  