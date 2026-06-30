#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CACHE_H_
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
namespace device_compilation_cache_internal {
template <typename ExecutableType>
int64_t ExecutableSize(const ExecutableType* executable) {
  return 0;
}
template <>
inline int64_t ExecutableSize<xla::LocalExecutable>(
    const xla::LocalExecutable* executable) {
  if (executable != nullptr && executable->executable() != nullptr) {
    return executable->executable()->SizeOfGeneratedCodeInBytes();
  }
  return 0;
}
template <>
inline int64_t ExecutableSize<xla::PjRtLoadedExecutable>(
    const xla::PjRtLoadedExecutable* executable) {
  if (executable != nullptr) {
    return executable->SizeOfGeneratedCodeInBytes();
  }
  return 0;
}
}  
template <typename ExecutableType>
class DeviceCompilationCache {
 public:
  DeviceCompilationCache() = default;
  ~DeviceCompilationCache() = default;
  using Key = DeviceCompilationClusterSignature;
  struct Value {
    DeviceCompileState compile_state = DeviceCompileState::kUncompiled;
    Status compilation_status;
    int64_t request_count = 0;
    const XlaCompiler::CompilationResult* compilation_result = nullptr;
    ExecutableType* executable = nullptr;
  };
  std::optional<Value> Lookup(const Key& key) const;
  Value LookupOrCreate(const Key& key);
  void Store(const Key& key, std::optional<DeviceCompileState> compile_state,
             std::optional<Status> compilation_status,
             std::optional<std::unique_ptr<XlaCompiler::CompilationResult>>
                 compilation_result,
             std::optional<std::unique_ptr<ExecutableType>> executable);
  std::string DebugString() const;
 private:
  struct Entry {
    mutable mutex mu;
    DeviceCompileState compile_state TF_GUARDED_BY(mu) =
        DeviceCompileState::kUncompiled;
    int64_t request_count TF_GUARDED_BY(mu) = 0;
    Status compilation_status TF_GUARDED_BY(mu);
    std::unique_ptr<XlaCompiler::CompilationResult> compilation_result
        TF_GUARDED_BY(mu);
    std::unique_ptr<ExecutableType> executable TF_GUARDED_BY(mu);
    std::string DebugString() const {
      mutex_lock lock(mu);
      int64_t executable_size =
          device_compilation_cache_internal::ExecutableSize<ExecutableType>(
              executable.get());
      int64_t hlo_module_size = 0;
      if (compilation_result != nullptr &&
          compilation_result->computation != nullptr) {
        hlo_module_size =
            compilation_result->computation->proto().ByteSizeLong();
      }
      return absl::StrCat(
          "{compile_state: ", compile_state, ", request_count: ", request_count,
          ", compilation_status: ", compilation_status.ToString(),
          ", compilation_result?: ", compilation_result != nullptr,
          ", hlo_module_size: ", hlo_module_size, " bytes",
          ", executable?: ", executable != nullptr,
          ", executable_size: ", executable_size, " bytes}");
    }
  };
  mutable mutex compile_cache_mu_;
  absl::flat_hash_map<Key, std::unique_ptr<Entry>, Key::Hash> cache_
      TF_GUARDED_BY(compile_cache_mu_);
  DeviceCompilationCache(const DeviceCompilationCache&) = delete;
  void operator=(const DeviceCompilationCache&) = delete;
};
template <typename ExecutableType>
std::optional<typename DeviceCompilationCache<ExecutableType>::Value>
DeviceCompilationCache<ExecutableType>::Lookup(const Key& key) const {
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    auto it = cache_.find(key);
    if (it == cache_.cend()) {
      return std::nullopt;
    }
    entry = it->second.get();
  }
  mutex_lock lock(entry->mu);
  Value value = {entry->compile_state,
                 entry->compilation_status,
                 ++entry->request_count,
                 entry->compilation_result.get(),
                 entry->executable.get()};
  return value;
}
template <typename ExecutableType>
typename DeviceCompilationCache<ExecutableType>::Value
DeviceCompilationCache<ExecutableType>::LookupOrCreate(const Key& key) {
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    auto it = cache_.emplace(key, std::make_unique<Entry>()).first;
    entry = it->second.get();
  }
  mutex_lock lock(entry->mu);
  Value value = {entry->compile_state,
                 entry->compilation_status,
                 ++entry->request_count,
                 entry->compilation_result.get(),
                 entry->executable.get()};
  return value;
}
template <typename ExecutableType>
void DeviceCompilationCache<ExecutableType>::Store(
    const Key& key, std::optional<DeviceCompileState> compile_state,
    std::optional<Status> compilation_status,
    std::optional<std::unique_ptr<XlaCompiler::CompilationResult>>
        compilation_result,
    std::optional<std::unique_ptr<ExecutableType>> executable) {
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    auto it = cache_.emplace(key, std::make_unique<Entry>()).first;
    entry = it->second.get();
  }
  {
    mutex_lock lock(entry->mu);
    if (compile_state.has_value()) {
      entry->compile_state = *compile_state;
    }
    if (compilation_status.has_value()) {
      entry->compilation_status = *compilation_status;
    }
    if (compilation_result.has_value()) {
      entry->compilation_result = std::move(*compilation_result);
    }
    if (executable.has_value()) {
      entry->executable = std::move(*executable);
    }
  }
  VLOG(4) << "Added/updated cache entry: key=" << key.HumanString()
          << ", entry=" << entry->DebugString();
}
template <typename ExecutableType>
std::string DeviceCompilationCache<ExecutableType>::DebugString() const {
  std::string s = "DeviceCompilationCache<ExecutableType> {\n";
  {
    mutex_lock lock(compile_cache_mu_);
    for (const auto& [key, entry] : cache_) {
      absl::StrAppend(&s, key.HumanString(), " : ", entry->DebugString(),
                      ",\n");
    }
  }
  absl::StrAppend(&s, "}");
  return s;
}
}  
#endif  