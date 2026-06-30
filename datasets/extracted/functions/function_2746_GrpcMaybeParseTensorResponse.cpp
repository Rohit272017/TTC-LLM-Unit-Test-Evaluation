#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
namespace tensorflow {
bool GrpcMaybeParseTensorResponse(::grpc::ByteBuffer* src,
                                  TensorResponse* dst) {
  ::tensorflow::GrpcByteSource byte_source(src);
  auto s = dst->ParseFrom(&byte_source);
  return s.ok();
}
}  