#include "tensorstore/internal/compression/blosc.h"
#include <cstddef>
#include <string>
#include <string_view>
#include "absl/status/status.h"
#include <blosc.h>
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace blosc {
Result<std::string> Encode(std::string_view input, const Options& options) {
  if (input.size() > BLOSC_MAX_BUFFERSIZE) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Blosc compression input of ", input.size(),
        " bytes exceeds maximum size of ", BLOSC_MAX_BUFFERSIZE));
  }
  std::string output(input.size() + BLOSC_MAX_OVERHEAD, '\0');
  int shuffle = options.shuffle;
  if (shuffle == -1) {
    shuffle = options.element_size == 1 ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE;
  }
  const int n = blosc_compress_ctx(
      options.clevel, shuffle, options.element_size, input.size(), input.data(),
      output.data(), output.size(), options.compressor, options.blocksize,
      1);
  if (n < 0) {
    return absl::InternalError(
        tensorstore::StrCat("Internal blosc error: ", n));
  }
  output.erase(n);
  return output;
}
Result<std::string> Decode(std::string_view input) {
  size_t nbytes;
  if (blosc_cbuffer_validate(input.data(), input.size(), &nbytes) != 0) {
    return absl::InvalidArgumentError("Invalid blosc-compressed data");
  }
  std::string output(nbytes, '\0');
  if (nbytes > 0) {
    const int n =
        blosc_decompress_ctx(input.data(), output.data(), output.size(),
                             1);
    if (n <= 0) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Blosc error: ", n));
    }
  }
  return output;
}
}  
}  