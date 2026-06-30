#ifndef TENSORFLOW_LITE_CORE_API_OP_RESOLVER_INTERNAL_H_
#define TENSORFLOW_LITE_CORE_API_OP_RESOLVER_INTERNAL_H_
#include <memory>
#include "tensorflow/lite/core/api/op_resolver.h"
namespace tflite {
class OpResolverInternal {
 public:
  OpResolverInternal() = delete;
  static bool MayContainUserDefinedOps(const OpResolver& op_resolver) {
    return op_resolver.MayContainUserDefinedOps();
  }
  static std::shared_ptr<::tflite::internal::OperatorsCache> GetSharedCache(
      const ::tflite::OpResolver& op_resolver) {
    return op_resolver.registration_externals_cache_;
  }
};
}  
#endif  