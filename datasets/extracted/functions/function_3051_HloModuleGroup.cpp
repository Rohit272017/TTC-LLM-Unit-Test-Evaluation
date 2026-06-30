#include "xla/hlo/ir/hlo_module_group.h"
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
namespace xla {
HloModuleGroup::HloModuleGroup(std::unique_ptr<HloModule> module)
    : name_(module->name()) {
  push_back(std::move(module));
}
HloModuleGroup::HloModuleGroup(absl::string_view name,
                               absl::Span<std::unique_ptr<HloModule>> modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}
HloModuleGroup::HloModuleGroup(
    absl::string_view name, std::vector<std::unique_ptr<HloModule>>&& modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}
std::vector<std::unique_ptr<HloModule>> HloModuleGroup::ConsumeModules() {
  std::vector<std::unique_ptr<HloModule>> ret_modules = std::move(modules_);
  modules_.clear();
  module_ptrs_.clear();
  return ret_modules;
}
std::string HloModuleGroup::ToString() const {
  std::ostringstream s;
  s << "HloModuleGroup " << name() << "\n\n";
  for (const HloModule* module : modules()) {
    s << module->ToString() << "\n";
  }
  return s.str();
}
HloModuleGroupProto HloModuleGroup::ToProto() const {
  HloModuleGroupProto proto;
  proto.set_name(name());
  for (const HloModule* module : modules()) {
    *proto.add_hlo_modules() = module->ToProto();
  }
  return proto;
}
 absl::StatusOr<HloModuleGroup> HloModuleGroup::CreateFromProto(
    const HloModuleGroupProto& proto,
    absl::Span<const HloModuleConfig> module_configs) {
  TF_RET_CHECK(!proto.name().empty()) << "Module group name cannot be empty";
  TF_RET_CHECK(proto.hlo_modules_size() > 0)
      << "Module group must have at least one HLO module";
  TF_RET_CHECK(proto.hlo_modules_size() == module_configs.size());
  std::vector<std::unique_ptr<HloModule>> modules;
  for (int i = 0; i < proto.hlo_modules_size(); ++i) {
    const HloModuleProto& module_proto = proto.hlo_modules(i);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProto(module_proto, module_configs[i]));
    modules.push_back(std::move(module));
  }
  return HloModuleGroup(proto.name(), absl::MakeSpan(modules));
}
void HloModuleGroup::push_back(std::unique_ptr<HloModule> module) {
  module->metadata()->set_module_group_name(name());
  modules_.push_back(std::move(module));
  module_ptrs_.push_back(modules_.back().get());
}
void HloModuleGroup::ReplaceModule(int index,
                                   std::unique_ptr<HloModule> module) {
  modules_.at(index)->MoveMetadataToModule(module.get());
  modules_.at(index) = std::move(module);
  module_ptrs_.at(index) = modules_.at(index).get();
}
std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group) {
  out << group.ToString();
  return out;
}
}  