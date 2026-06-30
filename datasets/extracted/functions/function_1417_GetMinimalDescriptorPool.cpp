#include "internal/minimal_descriptor_pool.h"
#include <cstdint>
#include "google/protobuf/descriptor.pb.h"
#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "google/protobuf/descriptor.h"
namespace cel::internal {
namespace {
ABSL_CONST_INIT const uint8_t kMinimalDescriptorSet[] = {
#include "internal/minimal_descriptor_set_embed.inc"
};
}  
absl::Nonnull<const google::protobuf::DescriptorPool*> GetMinimalDescriptorPool() {
  static absl::Nonnull<const google::protobuf::DescriptorPool* const> pool = []() {
    google::protobuf::FileDescriptorSet file_desc_set;
    ABSL_CHECK(file_desc_set.ParseFromArray(  
       kMinimalDescriptorSet, ABSL_ARRAYSIZE(kMinimalDescriptorSet)));
    auto* pool = new google::protobuf::DescriptorPool();
    for (const auto& file_desc : file_desc_set.file()) {
      ABSL_CHECK(pool->BuildFile(file_desc) != nullptr);  
    }
    return pool;
  }();
  return pool;
}
}  