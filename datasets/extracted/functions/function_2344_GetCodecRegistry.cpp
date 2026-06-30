#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include <stddef.h>
#include "absl/base/no_destructor.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
namespace tensorstore {
namespace internal_zarr3 {
ZarrCodecSpec::~ZarrCodecSpec() = default;
ZarrCodecKind ZarrArrayToArrayCodecSpec::kind() const {
  return ZarrCodecKind::kArrayToArray;
}
ZarrCodecKind ZarrArrayToBytesCodecSpec::kind() const {
  return ZarrCodecKind::kArrayToBytes;
}
size_t ZarrArrayToBytesCodecSpec::sharding_height() const { return 0; }
ZarrCodecKind ZarrBytesToBytesCodecSpec::kind() const {
  return ZarrCodecKind::kBytesToBytes;
}
CodecRegistry& GetCodecRegistry() {
  static absl::NoDestructor<CodecRegistry> registry;
  return *registry;
}
}  
}  