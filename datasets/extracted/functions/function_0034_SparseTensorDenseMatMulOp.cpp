#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"
#include "Eigen/Core"  
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/bfloat16.h"
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Device, typename T, typename Tindices>
class SparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit SparseTensorDenseMatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_b", &adjoint_b_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor* a_indices;
    const Tensor* a_values;
    const Tensor* a_shape;
    const Tensor* b;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape));
    OP_REQUIRES_OK(ctx, ctx->input("b", &b));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b->shape()),
                errors::InvalidArgument("Tensor 'b' is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_shape->shape()),
                errors::InvalidArgument("Tensor 'a_shape' is not a vector"));
    OP_REQUIRES(
        ctx, a_shape->NumElements() == 2,
        errors::InvalidArgument("Tensor 'a_shape' must have 2 elements"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_values->shape()),
                errors::InvalidArgument("Tensor 'a_values' is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a_indices->shape()),
                errors::InvalidArgument("Tensor 'a_indices' is not a matrix"));
    const int64_t nnz = a_indices->shape().dim_size(0);
    OP_REQUIRES(ctx, nnz == a_values->NumElements(),
                errors::InvalidArgument("Number of rows of a_indices does not "
                                        "match number of entries in a_values"));
    OP_REQUIRES(
        ctx, a_indices->shape().dim_size(1) == a_shape->NumElements(),
        errors::InvalidArgument("Number of columns of a_indices does not match "
                                "number of entries in a_shape"));
    auto a_shape_t = a_shape->vec<int64_t>();
    const int64_t outer_left = (adjoint_a_) ? a_shape_t(1) : a_shape_t(0);
    const int64_t outer_right =
        (adjoint_b_) ? b->shape().dim_size(0) : b->shape().dim_size(1);
    const int64_t inner_left = (adjoint_a_) ? a_shape_t(0) : a_shape_t(1);
    const int64_t inner_right =
        (adjoint_b_) ? b->shape().dim_size(1) : b->shape().dim_size(0);
    OP_REQUIRES(
        ctx, inner_right == inner_left,
        errors::InvalidArgument(
            "Cannot multiply A and B because inner dimension does not match: ",
            inner_left, " vs. ", inner_right,
            ".  Did you forget a transpose?  "
            "Dimensions of A: [",
            a_shape_t(0), ", ", a_shape_t(1),
            ").  Dimensions of B: ", b->shape().DebugString()));
    if (std::is_same<Device, GPUDevice>::value) {
      const int int32max = std::numeric_limits<int>::max();
      OP_REQUIRES(
          ctx,
          (FastBoundsCheck(inner_left, int32max) &&
           FastBoundsCheck(inner_right, int32max) &&
           FastBoundsCheck(outer_left, int32max) &&
           FastBoundsCheck(outer_right, int32max) &&
           FastBoundsCheck(b->NumElements(), int32max) &&
           FastBoundsCheck(outer_left * outer_right, int32max) &&
           FastBoundsCheck(a_values->NumElements(), int32max)),
          errors::InvalidArgument("Cannot use GPU for > 2^31 entry inputs"));
      OP_REQUIRES(ctx, FastBoundsCheck(nnz * outer_right, int32max),
                  errors::InvalidArgument(
                      "Cannot use GPU when output.shape[1] * nnz(a) > 2^31"));
    }
    TensorShape out_shape({outer_left, outer_right});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a_values->NumElements() == 0 || b->NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }
#define MAYBE_ADJOINT(ADJ_A, ADJ_B)                                           \
  if (adjoint_a_ == ADJ_A && adjoint_b_ == ADJ_B) {                           \
    Status functor_status = functor::SparseTensorDenseMatMulFunctor<          \
        Device, T, Tindices, ADJ_A,                                           \
        ADJ_B>::Compute(ctx, out->matrix<T>(), a_indices->matrix<Tindices>(), \
                        a_values->vec<T>(), b->matrix<T>());                  \
    OP_REQUIRES_OK(ctx, functor_status);                                      \
  }
    MAYBE_ADJOINT(false, false);
    MAYBE_ADJOINT(false, true);
    MAYBE_ADJOINT(true, false);
    MAYBE_ADJOINT(true, true);
#undef MAYBE_ADJOINT
  }
 private:
  bool adjoint_a_;
  bool adjoint_b_;
};
#define REGISTER_CPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<CPUDevice, TypeT, TypeIndex>);
#define REGISTER_KERNELS_CPU(T) \
  REGISTER_CPU(T, int64_t);     \
  REGISTER_CPU(T, int32)
REGISTER_KERNELS_CPU(Eigen::half);
REGISTER_KERNELS_CPU(float);
REGISTER_KERNELS_CPU(double);
REGISTER_KERNELS_CPU(int32);
REGISTER_KERNELS_CPU(complex64);
REGISTER_KERNELS_CPU(complex128);
REGISTER_KERNELS_CPU(bfloat16);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindices, ADJ_A, ADJ_B)                         \
  template <>                                                               \
  Status SparseTensorDenseMatMulFunctor<                                    \
      GPUDevice, T, Tindices, ADJ_A,                                        \
      ADJ_B>::Compute(OpKernelContext* ctx, typename TTypes<T>::Matrix out, \
                      TTypes<Tindices>::ConstMatrix a_indices,              \
                      typename TTypes<T>::ConstVec a_values,                \
                      typename TTypes<T>::ConstMatrix b);                   \
  extern template struct SparseTensorDenseMatMulFunctor<                    \
      GPUDevice, T, Tindices, ADJ_A, ADJ_B>;
#define REGISTER_GPU_SPEC(T, ADJ_A, ADJ_B)  \
  DECLARE_GPU_SPEC(T, int32, ADJ_A, ADJ_B); \
  DECLARE_GPU_SPEC(T, int64_t, ADJ_A, ADJ_B)
#define DECLARE_ADJOINT_GPU_SPEC(T)  \
  REGISTER_GPU_SPEC(T, false, false) \
  REGISTER_GPU_SPEC(T, false, true)  \
  REGISTER_GPU_SPEC(T, true, false)  \
  REGISTER_GPU_SPEC(T, true, true)
DECLARE_ADJOINT_GPU_SPEC(Eigen::half);
DECLARE_ADJOINT_GPU_SPEC(float);
DECLARE_ADJOINT_GPU_SPEC(double);
DECLARE_ADJOINT_GPU_SPEC(complex64);
DECLARE_ADJOINT_GPU_SPEC(complex128);
#undef DECLARE_ADJOINT_GPU_SPEC
#undef DECLARE_GPU_SPEC
#undef REGISTER_GPU_SPEC
}  
#define REGISTER_GPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_GPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<GPUDevice, TypeT, TypeIndex>);
#define REGISTER_KERNELS_GPU(T) \
  REGISTER_GPU(T, int64_t);     \
  REGISTER_GPU(T, int32)
REGISTER_KERNELS_GPU(Eigen::half);
REGISTER_KERNELS_GPU(float);
REGISTER_KERNELS_GPU(double);
REGISTER_KERNELS_GPU(complex64);
REGISTER_KERNELS_GPU(complex128);
#undef REGISTER_GPU
#undef REGISTER_KERNELS_GPU
#endif  
namespace functor {
namespace {
Status KOutOfBoundsError(int64_t k, std::size_t i, int rhs_index_a,
                         std::size_t lhs_right) {
  return errors::InvalidArgument("k (", k, ") from index[", i, ",", rhs_index_a,
                                 "] out of bounds (>=", lhs_right, ")");
}
Status MOutOfBoundsError(int64_t m, std::size_t i, int lhs_index_a,
                         int64_t out_dim0) {
  return errors::InvalidArgument("m (", m, ") from index[", i, ",", lhs_index_a,
                                 "] out of bounds (>=", out_dim0, ")");
}
template <typename T, typename Tsum, typename Tindices, bool ADJ_A, bool ADJ_B>
Status SparseTensorDenseMatMulImpl(
    typename TTypes<Tsum>::Matrix out,
    typename TTypes<Tindices>::ConstMatrix a_indices,
    typename TTypes<T>::ConstVec a_values, typename TTypes<T>::ConstMatrix b) {
  static constexpr std::size_t kNumVectorize = 32;
  const std::size_t nnz = a_values.size();
  const std::size_t rhs_right = (ADJ_B ? b.dimension(0) : b.dimension(1));
  const std::size_t lhs_right = (ADJ_B ? b.dimension(1) : b.dimension(0));
  const int lhs_index_a = ADJ_A ? 1 : 0;
  const int rhs_index_a = ADJ_A ? 0 : 1;
  if (rhs_right < kNumVectorize) {
    auto maybe_adjoint_b = MaybeAdjoint<decltype(b), ADJ_B>(b);
    for (std::size_t i = 0; i < nnz; ++i) {
      const Tindices m = internal::SubtleMustCopy(a_indices(i, lhs_index_a));
      const Tindices k = internal::SubtleMustCopy(a_indices(i, rhs_index_a));
      if (!FastBoundsCheck(k, lhs_right)) {
        return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);
      }
      if (!FastBoundsCheck(m, out.dimension(0))) {
        return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(0));
      }
      const T a_value = ADJ_A ? MaybeConj(a_values(i)) : a_values(i);
      for (std::size_t n = 0; n < rhs_right; ++n) {
        const T b_value = maybe_adjoint_b(k, n);
        out(m, n) += static_cast<Tsum>(a_value) * static_cast<Tsum>(b_value);
      }
    }
  } else {
    const int b_chip_index = ADJ_B ? 1 : 0;
#define LOOP_NNZ(b_passed)                                                  \
  for (std::size_t i = 0; i < nnz; ++i) {                                   \
    const Tindices m = internal::SubtleMustCopy(a_indices(i, lhs_index_a)); \
    const Tindices k = internal::SubtleMustCopy(a_indices(i, rhs_index_a)); \
    const T a_value = (ADJ_A) ? MaybeConj(a_values(i)) : a_values(i);       \
    if (!FastBoundsCheck(k, lhs_right)) {                                   \
      return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);               \
    }                                                                       \
    if (!FastBoundsCheck(m, out.dimension(0))) {                            \
      return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(0));        \
    }                                                                       \
    out.template chip<0>(m) +=                                              \
        b_passed.template chip<b_chip_index>(k).template cast<Tsum>() *     \
        static_cast<Tsum>(a_value);                                         \
  }
    if (ADJ_B) {
      Eigen::array<int, 2> shuffle{1, 0};  
      Eigen::Tensor<T, 2, Eigen::ColMajor> col_major_conj_b =
          b.swap_layout().shuffle(shuffle).conjugate();
      LOOP_NNZ(col_major_conj_b);
    } else {
      LOOP_NNZ(b);
    }
#undef LOOP_NNZ
  }
  return absl::OkStatus();
}
}  
template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<CPUDevice, T, Tindices, ADJ_A, ADJ_B> {
  static Status Compute(OpKernelContext* ctx, typename TTypes<T>::Matrix out,
                        typename TTypes<Tindices>::ConstMatrix a_indices,
                        typename TTypes<T>::ConstVec a_values,
                        typename TTypes<T>::ConstMatrix b) {
    using Tsum = typename SumType<T>::type;
    Tensor temp_out_t;
    if (!std::is_same<T, Tsum>::value) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DataTypeToEnum<Tsum>::value,
          TensorShape({out.dimension(0), out.dimension(1)}), &temp_out_t));
      auto temp_out = temp_out_t.matrix<Tsum>();
      temp_out.setZero();
      TF_RETURN_IF_ERROR(
          SparseTensorDenseMatMulImpl<T, Tsum, Tindices, ADJ_A, ADJ_B>(
              temp_out, a_indices, a_values, b));
      out = temp_out.template cast<T>();
    } else {
      out.setZero();
      auto out_workaround =
          *reinterpret_cast<typename TTypes<Tsum>::Matrix*>(&out);
      TF_RETURN_IF_ERROR(
          SparseTensorDenseMatMulImpl<T, Tsum, Tindices, ADJ_A, ADJ_B>(
              out_workaround, a_indices, a_values, b));
    }
    return absl::OkStatus();
  }
};
}  
}  