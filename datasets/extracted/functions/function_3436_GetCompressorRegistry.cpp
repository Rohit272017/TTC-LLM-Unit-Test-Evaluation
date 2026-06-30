#include "tensorstore/driver/n5/compressor.h"
#include <utility>
#include "absl/base/no_destructor.h"
#include "tensorstore/driver/n5/compressor_registry.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_registry.h"
namespace tensorstore {
namespace internal_n5 {
using CompressorRegistry = internal::JsonSpecifiedCompressor::Registry;
CompressorRegistry& GetCompressorRegistry() {
  static absl::NoDestructor<CompressorRegistry> registry;
  return *registry;
}
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Compressor, [](auto is_loading,
                                                      const auto& options,
                                                      auto* obj,
                                                      ::nlohmann::json* j) {
  namespace jb = tensorstore::internal_json_binding;
  auto& registry = GetCompressorRegistry();
  return jb::Object(
      jb::Member("type",
                 jb::MapValue(registry.KeyBinder(),
                              std::make_pair(Compressor{}, "raw"))),
      registry.RegisteredObjectBinder())(is_loading, options, obj, j);
})
}  
}  