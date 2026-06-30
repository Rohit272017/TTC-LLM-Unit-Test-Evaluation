#include "tensorstore/internal/compression/zlib_compressor.h"
#include <cstddef>
#include <memory>
#include <utility>
#include "riegeli/bytes/writer.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"
namespace tensorstore {
namespace internal {
std::unique_ptr<riegeli::Writer> ZlibCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  using Writer = riegeli::ZlibWriter<std::unique_ptr<riegeli::Writer>>;
  Writer::Options options;
  if (level != -1) options.set_compression_level(level);
  options.set_header(use_gzip_header ? Writer::Header::kGzip
                                     : Writer::Header::kZlib);
  return std::make_unique<Writer>(std::move(base_writer), options);
}
std::unique_ptr<riegeli::Reader> ZlibCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  using Reader = riegeli::ZlibReader<std::unique_ptr<riegeli::Reader>>;
  Reader::Options options;
  options.set_header(use_gzip_header ? Reader::Header::kGzip
                                     : Reader::Header::kZlib);
  return std::make_unique<Reader>(std::move(base_reader), options);
}
}  
}  