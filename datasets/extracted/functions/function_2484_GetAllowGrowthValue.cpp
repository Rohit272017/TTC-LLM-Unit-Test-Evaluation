#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include "xla/tsl/framework/bfc_allocator.h"
#include "tsl/platform/logging.h"
namespace tensorflow {
namespace {
bool GetAllowGrowthValue(bool orig_value) {
  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    return orig_value;
  }
  if (strcmp("false", force_allow_growth_string) == 0) {
    if (orig_value) {
      LOG(WARNING)
          << "Overriding orig_value setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << orig_value << ".";
    }
    return false;
  } else if (strcmp("true", force_allow_growth_string) == 0) {
    if (!orig_value) {
      LOG(WARNING)
          << "Overriding orig_value setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << orig_value << ".";
    }
    return true;
  }
  LOG(ERROR)
      << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
      << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
      << " values are \"true\" or \"false\". Using original config value"
      << " of " << orig_value << ".";
  return orig_value;
}
bool GetGarbageCollectionValue() {
  const char* enable_gpu_garbage_collection =
      std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
  if (enable_gpu_garbage_collection == nullptr) {
    return true;
  }
  if (strcmp("false", enable_gpu_garbage_collection) == 0) {
    return false;
  } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
    return true;
  }
  LOG(ERROR)
      << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
      << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
      << " Valid values are \"true\" or \"false\"."
      << " Using the default value \"true\".";
  return true;
}
}  
GPUBFCAllocator::GPUBFCAllocator(
    std::unique_ptr<tsl::SubAllocator> sub_allocator, size_t total_memory,
    const std::string& name, const Options& opts)
    : BFCAllocator(std::move(sub_allocator), total_memory, name, [&] {
        BFCAllocator::Options o;
        o.allow_growth = GetAllowGrowthValue(opts.allow_growth);
        o.allow_retry_on_failure = opts.allow_retry_on_failure;
        if (opts.garbage_collection.has_value()) {
          o.garbage_collection = *opts.garbage_collection;
        } else {
          o.garbage_collection = GetGarbageCollectionValue();
        }
        o.fragmentation_fraction = opts.fragmentation_fraction;
        return o;
      }()) {}
}  