#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "grpcpp/support/byte_buffer.h"
#include "grpcpp/support/slice.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/proto_encode_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/worker.pb.h"
namespace tensorflow {
namespace grpc {
void EncodeRecvTensorResponseToByteBuffer(const RecvTensorResponse& proto,
                                          ::grpc::ByteBuffer* result) {
  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(
      const_cast<uint8*>(reinterpret_cast<const uint8*>(slice.begin())));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}
static int VarLengthEncodingSize(uint32 tag, size_t bytes) {
  return core::VarintLength(tag << 3) + core::VarintLength(bytes) + bytes;
}
static int SkeletonEncodingSizeUpperBound(const Tensor& val) {
  static const int kVarintMax64 = 10;  
  const int ndims = val.shape().dims();
  return (2 * kVarintMax64) +           
         (ndims * (4 * kVarintMax64));  
}
static void EncodeSkeleton(const Tensor& val, io::ProtoEncodeHelper* e) {
  e->WriteUint64(TensorProto::kDtypeFieldNumber, val.dtype());
  const int ndims = val.shape().dims();
  int tensor_shape_bytes = 0;
  for (int d = 0; d < ndims; d++) {
    int64_t dim_size = val.shape().dim_size(d);
    tensor_shape_bytes +=
        2 +  
        1 +  
        core::VarintLength(dim_size);
  }
  if (tensor_shape_bytes > 0) {
    e->WriteVarlengthBeginning(TensorProto::kTensorShapeFieldNumber,
                               tensor_shape_bytes);
    for (int d = 0; d < ndims; d++) {
      int64_t dim_size = val.shape().dim_size(d);
      int64_t dim_varlen = 1 +  
                           core::VarintLength(dim_size);
      e->WriteVarlengthBeginning(TensorShapeProto::kDimFieldNumber, dim_varlen);
      e->WriteUint64(TensorShapeProto_Dim::kSizeFieldNumber, dim_size);
    }
  }
#ifndef NDEBUG
  {
    TensorProto skeleton;
    skeleton.set_dtype(val.dtype());
    val.shape().AsProto(skeleton.mutable_tensor_shape());
    string tensor_except_contents;  
    skeleton.AppendToString(&tensor_except_contents);
    TensorProto skeleton2;
    skeleton2.ParseFromString(string(e->data(), e->size()));
    string out;
    skeleton.AppendToString(&out);
    DCHECK_EQ(tensor_except_contents, out) << skeleton.DebugString() << " vs\n"
                                           << skeleton2.DebugString();
  }
#endif
}
void EncodeTensorToByteBuffer(bool is_dead, const Tensor& val, bool require_ack,
                              ::grpc::ByteBuffer* result) {
  const int kLargeTensorBytes = 1024;
  const int64_t kProtoBufLimitBytes = 1LL << 31;
  if (val.TotalBytes() > kProtoBufLimitBytes) {
    size_t exceeded_bytes = val.TotalBytes() - kProtoBufLimitBytes;
    LOG(FATAL) << "Cannot encode a Tensor that exceeds the 2GB protobuf limit. "
                  "Exceeded bytes: "
               << exceeded_bytes
               << ", tensor shape: " << val.shape().AsProto().DebugString();
  }
  RecvTensorResponse response;
  if (is_dead) {
    response.set_is_dead(is_dead);
  }
  response.set_require_ack(require_ack);
  response.set_send_start_micros(Env::Default()->NowMicros());
  if (!DataTypeCanUseMemcpy(val.dtype())) {
    val.AsProtoTensorContent(response.mutable_tensor());
    EncodeRecvTensorResponseToByteBuffer(response, result);
  } else {
    absl::InlinedVector<char, 128UL> skeleton(
        SkeletonEncodingSizeUpperBound(val));
    io::ProtoEncodeHelper e_skeleton(skeleton.data(), skeleton.size());
    EncodeSkeleton(val, &e_skeleton);
    StringPiece tdata = val.tensor_data();
    uint32 overall_tensor_proto_bytesize =
        (e_skeleton.size() +
         VarLengthEncodingSize(TensorProto::kTensorContentFieldNumber,
                               tdata.size()));
    string header;  
    response.AppendToString(&header);
    size_t expected_size =
        (header.size() +
         VarLengthEncodingSize(RecvTensorResponse::kTensorFieldNumber,
                               overall_tensor_proto_bytesize));
    bool share_tensor_slice_memory = (tdata.size() > kLargeTensorBytes);
    size_t encoder_size = expected_size - tdata.size();
    absl::InlinedVector<char, 1024UL> space(encoder_size);
    io::ProtoEncodeHelper e(space.data(), space.size());
    e.WriteRawBytes(header);
    e.WriteVarlengthBeginning(RecvTensorResponse::kTensorFieldNumber,
                              overall_tensor_proto_bytesize);
    e.WriteRawBytes(StringPiece(e_skeleton.data(), e_skeleton.size()));
    e.WriteVarlengthBeginning(TensorProto::kTensorContentFieldNumber,
                              tdata.size());
    ::grpc::Slice slices[2];
    int num_slices = 0;
    {
      size_t slice_len =
          e.size() + (share_tensor_slice_memory ? 0 : tdata.size());
      slices[0] = ::grpc::Slice(slice_len);
      memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
      if (!share_tensor_slice_memory) {
        memcpy(const_cast<uint8_t*>(slices[0].begin()) + e.size(), tdata.data(),
               tdata.size());
      }
      num_slices += 1;
    }
    if (share_tensor_slice_memory) {
      const TensorBuffer* buf = DMAHelper::buffer(&val);
      buf->Ref();
      slices[1] = ::grpc::Slice(
          const_cast<void*>(static_cast<const void*>(tdata.data())),
          tdata.size(),
          [](void* backing) { static_cast<TensorBuffer*>(backing)->Unref(); },
          const_cast<TensorBuffer*>(buf));
      num_slices += 1;
    }
    size_t total_bytes = 0;
    for (int i = 0; i < num_slices; i++) {
      total_bytes += slices[i].size();
    }
    CHECK_EQ(total_bytes, expected_size);
    ::grpc::ByteBuffer tmp(&slices[0], num_slices);
    result->Swap(&tmp);
  }
}
}  
}  