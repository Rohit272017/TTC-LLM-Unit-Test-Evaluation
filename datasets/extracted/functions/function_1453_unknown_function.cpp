#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_WRITER_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_WRITER_H_
#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_image {
class ImageWriter {
 public:
  virtual ~ImageWriter() = default;
  virtual absl::Status Initialize(riegeli::Writer*) = 0;
  virtual absl::Status Encode(
      const ImageInfo& image,
      tensorstore::span<const unsigned char> source) = 0;
  virtual absl::Status Done() = 0;
};
}  
}  
#endif  