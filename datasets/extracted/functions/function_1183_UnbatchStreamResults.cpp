#include "tensorflow/core/tfrt/kernels/stream_ops_util.h"
#include <cstdint>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/kernels/stream_ops_util_constants.h"
namespace tensorflow {
namespace tfrt_stub {
absl::StatusOr<std::vector<std::pair<int64_t, std::vector<tensorflow::Tensor>>>>
UnbatchStreamResults(const tensorflow::Tensor& step_ids,
                     absl::Span<const tensorflow::Tensor> tensors) {
  std::vector<std::pair<int64_t, std::vector<tensorflow::Tensor>>> responses;
  if (step_ids.dims() > 0) {
    if (step_ids.dtype() != tensorflow::DT_INT64 || step_ids.dims() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected a 1-D int64 tensor for batched step ids but got dtype=",
          tensorflow::DataTypeString(step_ids.dtype()),
          " shape=", step_ids.shape().DebugString()));
    }
    const int batch_size = step_ids.dim_size(0);
    for (int i = 0; i < tensors.size(); ++i) {
      const tensorflow::TensorShape& shape = tensors[i].shape();
      if (shape.dims() < 1 || shape.dim_size(0) != batch_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "All inputs to PwStreamResults inside tf.batch_function are "
            "required to be batched (batch_size=",
            batch_size, ") but input #", i, " has shape ",
            shape.DebugString()));
      }
    }
    std::vector<int> sizes;
    absl::flat_hash_set<int64_t> unique_step_ids;
    for (int i = 0; i < step_ids.NumElements(); ++i) {
      const int64_t request_id = step_ids.flat<int64_t>()(i);
      const int64_t step_id =
          static_cast<uint64_t>(request_id) >> (64 - kStepIdBitSize);
      VLOG(1) << "PwStreamResults op is unbatching request_id=" << request_id
              << ", step_id=" << step_id;
      if (step_id <= 0) {
        return absl::InternalError(
            absl::StrCat("Invalid step id=", step_id,
                         "; this usually indicates that `PwStreamResults` "
                         "was called from an unsupported nested context"));
      }
      if (i != 0 && request_id == step_ids.flat<int64_t>()(0)) {
        break;
      }
      if (!responses.empty() && responses.back().first == step_id) {
        sizes.back()++;
      } else {
        responses.push_back({step_id, {}});
        sizes.push_back(1);
        const bool inserted = unique_step_ids.insert(step_id).second;
        if (!inserted) {
          return absl::InternalError(absl::StrCat(
              "Non-contiguous step ids found in the step id batch: ",
              step_ids.DebugString(batch_size)));
        }
      }
    }
    int offset = 0;
    for (int i = 0; i < responses.size(); ++i) {
      auto& outputs = responses[i].second;
      outputs.resize(tensors.size());
      const int limit = offset + sizes[i];
      for (int j = 0; j < tensors.size(); ++j) {
        outputs[j] = tensors[j].Slice(offset, limit);
      }
      offset = limit;
    }
  } else {
    const int64_t step_id = step_ids.flat<int64_t>()(0);
    if (step_id <= 0) {
      return absl::InternalError(
          "Invalid step id; this usually indicates that `PwStreamResults` was "
          "called from an unsupported nested context");
    }
    responses.push_back({step_id, std::vector<tensorflow::Tensor>(
                                      tensors.begin(), tensors.end())});
  }
  return responses;
}
}  
}  