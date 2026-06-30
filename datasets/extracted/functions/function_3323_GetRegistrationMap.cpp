#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include <string>
#include <unordered_map>
#include "absl/base/call_once.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace grappler {
namespace {
typedef std::unordered_map<string, CustomGraphOptimizerRegistry::Creator>
    RegistrationMap;
RegistrationMap* registered_optimizers = nullptr;
RegistrationMap* GetRegistrationMap() {
  if (registered_optimizers == nullptr)
    registered_optimizers = new RegistrationMap;
  return registered_optimizers;
}
typedef std::unordered_map<string, PluginGraphOptimizerRegistry::Creator>
    PluginRegistrationMap;
PluginRegistrationMap* GetPluginRegistrationMap() {
  static PluginRegistrationMap* registered_plugin_optimizers =
      new PluginRegistrationMap;
  return registered_plugin_optimizers;
}
typedef std::unordered_map<string, ConfigList> PluginConfigMap;
PluginConfigMap* GetPluginConfigMap() {
  static PluginConfigMap* plugin_config_map = new PluginConfigMap;
  return plugin_config_map;
}
const ConfigList& DefaultPluginConfigs() {
  static ConfigList* default_plugin_configs = new ConfigList(
      false,
      {{"implementation_selector", RewriterConfig::ON},
       {"function_optimization", RewriterConfig::ON},
       {"common_subgraph_elimination", RewriterConfig::ON},
       {"arithmetic_optimization", RewriterConfig::ON},
       {"debug_stripper", RewriterConfig::ON},
       {"constant_folding", RewriterConfig::ON},
       {"shape_optimization", RewriterConfig::ON},
       {"auto_mixed_precision", RewriterConfig::ON},
       {"auto_mixed_precision_onednn_bfloat16", RewriterConfig::ON},
       {"auto_mixed_precision_mkl", RewriterConfig::ON},
       {"auto_mixed_precision_cpu", RewriterConfig::ON},
       {"pin_to_host_optimization", RewriterConfig::ON},
       {"layout_optimizer", RewriterConfig::ON},
       {"remapping", RewriterConfig::ON},
       {"loop_optimization", RewriterConfig::ON},
       {"dependency_optimization", RewriterConfig::ON},
       {"auto_parallel", RewriterConfig::ON},
       {"memory_optimization", RewriterConfig::ON},
       {"scoped_allocator_optimization", RewriterConfig::ON}});
  return *default_plugin_configs;
}
}  
std::unique_ptr<CustomGraphOptimizer>
CustomGraphOptimizerRegistry::CreateByNameOrNull(const string& name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) return nullptr;
  return std::unique_ptr<CustomGraphOptimizer>(it->second());
}
std::vector<string> CustomGraphOptimizerRegistry::GetRegisteredOptimizers() {
  std::vector<string> optimizer_names;
  optimizer_names.reserve(GetRegistrationMap()->size());
  for (const auto& opt : *GetRegistrationMap())
    optimizer_names.emplace_back(opt.first);
  return optimizer_names;
}
void CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
    const Creator& optimizer_creator, const string& name) {
  const auto it = GetRegistrationMap()->find(name);
  if (it != GetRegistrationMap()->end()) {
    LOG(FATAL) << "CustomGraphOptimizer is registered twice: " << name;
  }
  GetRegistrationMap()->insert({name, optimizer_creator});
}
std::vector<std::unique_ptr<CustomGraphOptimizer>>
PluginGraphOptimizerRegistry::CreateOptimizers(
    const std::set<string>& device_types) {
  std::vector<std::unique_ptr<CustomGraphOptimizer>> optimizer_list;
  for (auto it = GetPluginRegistrationMap()->begin();
       it != GetPluginRegistrationMap()->end(); ++it) {
    if (device_types.find(it->first) == device_types.end()) continue;
    static absl::once_flag plugin_optimizer_flag;
    absl::call_once(plugin_optimizer_flag, [&]() {
      LOG(INFO) << "Plugin optimizer for device_type " << it->first
                << " is enabled.";
    });
    optimizer_list.emplace_back(
        std::unique_ptr<CustomGraphOptimizer>(it->second()));
  }
  return optimizer_list;
}
void PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(
    const Creator& optimizer_creator, const std::string& device_type,
    ConfigList& configs) {
  auto ret = GetPluginConfigMap()->insert({device_type, configs});
  if (!ret.second) {
    LOG(FATAL) << "PluginGraphOptimizer with device_type "  
               << device_type << " is registered twice.";
  }
  GetPluginRegistrationMap()->insert({device_type, optimizer_creator});
}
void PluginGraphOptimizerRegistry::PrintPluginConfigsIfConflict(
    const std::set<string>& device_types) {
  bool init = false, conflict = false;
  ConfigList plugin_configs;
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;
    if (!init) {
      plugin_configs = cur_plugin_configs;
      init = true;
    } else {
      if (!(plugin_configs == cur_plugin_configs)) {
        conflict = true;
        break;
      }
    }
  }
  if (!conflict) return;
  LOG(WARNING) << "Plugins have conflicting configs. Potential performance "
                  "regression may happen.";
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;
    string logs = "";
    strings::StrAppend(&logs, "disable_model_pruning\t\t",
                       cur_plugin_configs.disable_model_pruning, "\n");
    for (auto const& pair : cur_plugin_configs.toggle_config) {
      strings::StrAppend(&logs, pair.first, string(32 - pair.first.size(), ' '),
                         (pair.second != RewriterConfig::OFF), "\n");
    }
    LOG(WARNING) << "Plugin's configs for device_type " << device_type << ":\n"
                 << logs;
  }
}
ConfigList PluginGraphOptimizerRegistry::GetPluginConfigs(
    bool use_plugin_optimizers, const std::set<string>& device_types) {
  if (!use_plugin_optimizers) return DefaultPluginConfigs();
  ConfigList ret_plugin_configs = DefaultPluginConfigs();
  for (const auto& device_type : device_types) {
    const auto it = GetPluginConfigMap()->find(device_type);
    if (it == GetPluginConfigMap()->end()) continue;
    auto cur_plugin_configs = it->second;
    if (cur_plugin_configs.disable_model_pruning == true)
      ret_plugin_configs.disable_model_pruning = true;
    for (auto& pair : cur_plugin_configs.toggle_config) {
      if (cur_plugin_configs.toggle_config[pair.first] == RewriterConfig::OFF)
        ret_plugin_configs.toggle_config[pair.first] = RewriterConfig::OFF;
    }
  }
  return ret_plugin_configs;
}
bool PluginGraphOptimizerRegistry::IsConfigsConflict(
    ConfigList& user_config, ConfigList& plugin_config) {
  if (plugin_config == DefaultPluginConfigs()) return false;
  if (user_config.disable_model_pruning != plugin_config.disable_model_pruning)
    return true;
  for (auto& pair : user_config.toggle_config) {
    if ((user_config.toggle_config[pair.first] == RewriterConfig::ON) &&
        (plugin_config.toggle_config[pair.first] == RewriterConfig::OFF))
      return true;
  }
  return false;
}
}  
}  