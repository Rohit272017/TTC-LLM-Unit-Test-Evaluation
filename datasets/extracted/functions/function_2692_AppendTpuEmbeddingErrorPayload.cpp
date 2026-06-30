#include "tensorflow/core/tpu/tpu_embedding_errors.h"
#include <string>
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
namespace tensorflow::tpu {
Status AppendTpuEmbeddingErrorPayload(Status obj) {
  if (obj.ok()) {
    return absl::OkStatus();
  } else {
    const std::string error_message =
        absl::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.message());
    Status status(obj.code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl,
                      absl::Cord(error_payload.SerializeAsString()));
    return status;
  }
}
bool HasTpuEmbeddingErrorPayload(const Status& status) {
  return status.GetPayload(kTpuEmbeddingErrorUrl).has_value();
}
bool HasTpuEmbeddingErrorMessage(const Status& status) {
  return absl::StrContains(status.message(), kTpuEmbeddingErrorMessage);
}
}  