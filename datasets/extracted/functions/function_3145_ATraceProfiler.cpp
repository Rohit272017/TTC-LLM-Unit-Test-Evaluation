#include "tensorflow/lite/profiling/atrace_profiler.h"
#include <dlfcn.h>
#include "tensorflow/lite/core/api/profiler.h"
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#include <string>
#include <type_traits>
namespace tflite {
namespace profiling {
class ATraceProfiler : public tflite::Profiler {
 public:
  using FpIsEnabled = std::add_pointer<bool()>::type;
  using FpBeginSection = std::add_pointer<void(const char*)>::type;
  using FpEndSection = std::add_pointer<void()>::type;
  ATraceProfiler() {
    handle_ = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (handle_) {
      atrace_is_enabled_ =
          reinterpret_cast<FpIsEnabled>(dlsym(handle_, "ATrace_isEnabled"));
      atrace_begin_section_ = reinterpret_cast<FpBeginSection>(
          dlsym(handle_, "ATrace_beginSection"));
      atrace_end_section_ =
          reinterpret_cast<FpEndSection>(dlsym(handle_, "ATrace_endSection"));
      if (!atrace_is_enabled_ || !atrace_begin_section_ ||
          !atrace_end_section_) {
        dlclose(handle_);
        handle_ = nullptr;
      }
    }
  }
  ~ATraceProfiler() override {
    if (handle_) {
      dlclose(handle_);
    }
  }
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    if (handle_ && atrace_is_enabled_()) {
      std::string trace_event_tag = tag;
      trace_event_tag += "@";
      trace_event_tag += std::to_string(event_metadata1) + "/" +
                         std::to_string(event_metadata2);
      atrace_begin_section_(trace_event_tag.c_str());
    }
    return 0;
  }
  void EndEvent(uint32_t event_handle) override {
    if (handle_) {
      atrace_end_section_();
    }
  }
 private:
  void* handle_;
  FpIsEnabled atrace_is_enabled_;
  FpBeginSection atrace_begin_section_;
  FpEndSection atrace_end_section_;
};
std::unique_ptr<tflite::Profiler> MaybeCreateATraceProfiler() {
#if defined(TFLITE_ENABLE_DEFAULT_PROFILER)
  return std::unique_ptr<tflite::Profiler>(new ATraceProfiler());
#else  
#if defined(__ANDROID__)
  constexpr char kTraceProp[] = "debug.tflite.trace";
  char trace_enabled[PROP_VALUE_MAX] = "";
  int length = __system_property_get(kTraceProp, trace_enabled);
  if (length == 1 && trace_enabled[0] == '1') {
    return std::unique_ptr<tflite::Profiler>(new ATraceProfiler());
  }
#endif  
  return nullptr;
#endif  
}
}  
}  