#include "tensorstore/internal/compression/bzip2_compressor.h"
#include <cstddef>
#include <memory>
#include <utility>
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/bzip2/bzip2_reader.h"
#include "riegeli/bzip2/bzip2_writer.h"
namespace tensorstore {
namespace internal {
std::unique_ptr<riegeli::Writer> Bzip2Compressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  using Writer = riegeli::Bzip2Writer<std::unique_ptr<riegeli::Writer>>;
  Writer::Options options;
  options.set_compression_level(level);
  return std::make_unique<Writer>(std::move(base_writer), options);
}
std::unique_ptr<riegeli::Reader> Bzip2Compressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  using Reader = riegeli::Bzip2Reader<std::unique_ptr<riegeli::Reader>>;
  return std::make_unique<Reader>(std::move(base_reader));
}
}  
}  