#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_READER_H_
#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_image {
class ImageReader {
 public:
  virtual ~ImageReader() = default;
  virtual absl::Status Initialize(riegeli::Reader* reader) = 0;
  virtual ImageInfo GetImageInfo() = 0;
  virtual absl::Status Decode(tensorstore::span<unsigned char> dest) = 0;
};
}  
}  
#endif  