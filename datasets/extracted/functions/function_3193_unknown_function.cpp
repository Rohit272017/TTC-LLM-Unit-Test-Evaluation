#include "tensorstore/internal/compression/zstd_compressor.h"
#include <cstddef>
#include <memory>
#include <utility>
#include "riegeli/bytes/writer.h"
#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"
namespace tensorstore {
namespace internal {
std::unique_ptr<riegeli::Writer> ZstdCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  using Writer = riegeli::ZstdWriter<std::unique_ptr<riegeli::Writer>>;
  Writer::Options options;
  options.set_compression_level(level);
  return std::make_unique<Writer>(std::move(base_writer), options);
}
std::unique_ptr<riegeli::Reader> ZstdCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  using Reader = riegeli::ZstdReader<std::unique_ptr<riegeli::Reader>>;
  return std::make_unique<Reader>(std::move(base_reader));
}
}  
}  