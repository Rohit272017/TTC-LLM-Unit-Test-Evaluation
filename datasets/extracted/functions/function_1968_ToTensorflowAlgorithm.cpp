#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/rng_alg.h"
namespace tensorflow {
Algorithm ToTensorflowAlgorithm(xla::RandomAlgorithm alg) {
  switch (alg) {
    case xla::RandomAlgorithm::RNG_PHILOX:
      return RNG_ALG_PHILOX;
    case xla::RandomAlgorithm::RNG_THREE_FRY:
      return RNG_ALG_THREEFRY;
    case xla::RandomAlgorithm::RNG_DEFAULT:  
    default:
      return RNG_ALG_AUTO_SELECT;
  }
}
xla::RandomAlgorithm DefaultRngAlgForDeviceType(
    absl::string_view device_type_string) {
  if (device_type_string == DEVICE_GPU_XLA_JIT ||
      device_type_string == DEVICE_CPU_XLA_JIT) {
    return xla::RandomAlgorithm::RNG_PHILOX;
  } else {
    return xla::RandomAlgorithm::RNG_DEFAULT;
  }
}
}  