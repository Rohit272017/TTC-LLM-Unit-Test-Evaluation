#include "tsl/profiler/lib/profiler_factory.h"
#include <memory>
#include <utility>
#include <vector>
#include "tsl/platform/mutex.h"
#include "tsl/profiler/lib/profiler_controller.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
namespace tsl {
namespace profiler {
namespace {
mutex mu(LINKER_INITIALIZED);
std::vector<ProfilerFactory>* GetFactories() {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}
}  
void RegisterProfilerFactory(ProfilerFactory factory) {
  mutex_lock lock(mu);
  GetFactories()->push_back(std::move(factory));
}
std::vector<std::unique_ptr<profiler::ProfilerInterface>> CreateProfilers(
    const tensorflow::ProfileOptions& options) {
  std::vector<std::unique_ptr<profiler::ProfilerInterface>> result;
  mutex_lock lock(mu);
  for (const auto& factory : *GetFactories()) {
    auto profiler = factory(options);
    if (profiler == nullptr) continue;
    result.emplace_back(
        std::make_unique<ProfilerController>(std::move(profiler)));
  }
  return result;
}
void ClearRegisteredProfilersForTest() {
  mutex_lock lock(mu);
  GetFactories()->clear();
}
}  
}  