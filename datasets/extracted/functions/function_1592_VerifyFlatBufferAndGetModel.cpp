#include "tensorflow/lite/core/tools/verifier_internal.h"
#include <stddef.h>
#include <stdint.h>
#include "flatbuffers/verifier.h"  
#include "tensorflow/lite/schema/schema_generated.h"
namespace tflite {
namespace internal {
const Model* VerifyFlatBufferAndGetModel(const void* buf, size_t len) {
  ::flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  if (VerifyModelBuffer(verifier)) {
    return ::tflite::GetModel(buf);
  } else {
    return nullptr;
  }
}
}  
}  