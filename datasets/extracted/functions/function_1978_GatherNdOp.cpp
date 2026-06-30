#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/gather_nd_op.h"
#include <string>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bad_indices_policy.h"
namespace tensorflow {
namespace {
constexpr char kBadIndicesPolicyAtrr[] = "bad_indices_policy";
}  
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Device, typename T, typename Index>
class GatherNdOp : public OpKernel {
 public:
  explicit GatherNdOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
    if (c->HasAttr(kBadIndicesPolicyAtrr)) {
      std::string bad_indices_policy_str;
      OP_REQUIRES_OK(
          c, c->GetAttr(kBadIndicesPolicyAtrr, &bad_indices_policy_str));
      absl::StatusOr<BadIndicesPolicy> bad_indices_policy =
          BadIndicesPolicyFromString(bad_indices_policy_str);
      OP_REQUIRES_OK(c, bad_indices_policy.status());
      bad_indices_policy_ = *bad_indices_policy;
    }
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);
    Tensor out;
    OP_REQUIRES_OK(c, functor::DoGatherNd<Device, T, Index>(
                          c, params, indices, &out, bad_indices_policy_));
    c->set_output(0, out);
  }
 private:
  BadIndicesPolicy bad_indices_policy_ = BadIndicesPolicy::kDefault;
};
#define REGISTER_GATHER_ND_FULL(dev, type, index_type)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("GatherNd")                                         \
          .Device(DEVICE_##dev)                                \
          .TypeConstraint<type>("Tparams")                     \
          .TypeConstraint<index_type>("Tindices")              \
          .AttrConstraint<std::string>(                        \
              "bad_indices_policy",                            \
              {"", "DEFAULT", "ERROR", "IGNORE"}), \
      GatherNdOp<dev##Device, type, index_type>)
#define REGISTER_GATHER_ND_CPU(type)         \
  REGISTER_GATHER_ND_FULL(CPU, type, int16); \
  REGISTER_GATHER_ND_FULL(CPU, type, int32); \
  REGISTER_GATHER_ND_FULL(CPU, type, int64_t)
TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_float8_e5m2(REGISTER_GATHER_ND_CPU);
TF_CALL_float8_e4m3fn(REGISTER_GATHER_ND_CPU);
#undef REGISTER_GATHER_ND_CPU
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {
#define DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM)          \
  template <>                                                 \
  Index GatherNdSlice<GPUDevice, T, Index, NDIM>::operator()( \
      const GPUDevice& d, const Index slice_size,             \
      typename TTypes<int32>::Scalar Tscratch,                \
      typename TTypes<T, NDIM + 1>::ConstTensor Tparams,      \
      typename TTypes<Index>::ConstMatrix Tindices,           \
      typename TTypes<T>::Matrix Tout);                       \
  extern template struct GatherNdSlice<GPUDevice, T, Index, NDIM>;
#define DECLARE_GPU_SPECS_INDEX(T, Index)    \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 0); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 5); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 6); \
  DECLARE_GPU_SPECS_INDEX_NDIM(T, Index, 7);
#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64_t)
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
}  
#undef REGISTER_GATHER_ND_FULL
#define REGISTER_GATHER_ND_FULL(dev, type, index_type)                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("GatherNd")                                                         \
          .Device(DEVICE_##dev)                                                \
          .TypeConstraint<type>("Tparams")                                     \
          .TypeConstraint<index_type>("Tindices")                              \
          .AttrConstraint<std::string>("bad_indices_policy",                   \
                                       {"", "DEFAULT", "IGNORE"}), \
      GatherNdOp<dev##Device, type, index_type>)
#define REGISTER_GATHER_ND_GPU(type)         \
  REGISTER_GATHER_ND_FULL(GPU, type, int32); \
  REGISTER_GATHER_ND_FULL(GPU, type, int64_t)
TF_CALL_int32(REGISTER_GATHER_ND_GPU);
TF_CALL_int64(REGISTER_GATHER_ND_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_ND_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GATHER_ND_GPU);
#undef REGISTER_GATHER_ND_GPU
#endif  
#undef REGISTER_GATHER_ND_FULL
}  