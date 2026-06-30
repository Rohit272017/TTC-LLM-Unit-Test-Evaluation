#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/driver/n5/compressor_registry.h"
#include "tensorstore/internal/compression/zlib_compressor.h"
#include "tensorstore/internal/json_binding/json_binding.h"
namespace tensorstore {
namespace internal_n5 {
namespace {
struct Registration {
  Registration() {
    using internal::ZlibCompressor;
    namespace jb = tensorstore::internal_json_binding;
    RegisterCompressor<ZlibCompressor>(
        "gzip",
        jb::Object(
            jb::Member(
                "level",
                jb::Projection(
                    &ZlibCompressor::level,
                    jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                        [](auto* v) { *v = -1; }, jb::Integer<int>(-1, 9)))),
            jb::Member(
                "useZlib",
                jb::Projection(
                    &ZlibCompressor::use_gzip_header,
                    jb::GetterSetter(
                        [](bool use_gzip) { return !use_gzip; },
                        [](bool& use_gzip, bool use_zlib) {
                          use_gzip = !use_zlib;
                        },
                        jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                            [](bool* use_zlib) { *use_zlib = false; }))))));
  }
} registration;
}  
}  
}  