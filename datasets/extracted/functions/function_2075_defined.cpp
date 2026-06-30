#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define EIGEN_USE_CUSTOM_THREAD_POOL
#define EIGEN_USE_THREADS
#define Eigen EigenForTFLite
#define TFLITE_REDUCE_INSTANTIATIONS
#if defined(TFLITE_REDUCE_INSTANTIATIONS)
#define TENSOR_CONTRACTION_DISPATCH(METHOD, ALIGNMENT, ARGS)                  \
  if (this->m_lhs_inner_dim_contiguous && this->m_rhs_inner_dim_contiguous && \
      !this->m_rhs_inner_dim_reordered) {                                     \
    METHOD<true, true, false, ALIGNMENT> ARGS;                                \
  } else {                                                                    \
    eigen_assert(false && "Unsupported contraction formats");                 \
  }
#endif
#include "unsupported/Eigen/CXX11/Tensor"  
#include "xla/tsl/framework/convolution/eigen_spatial_convolutions-inl.h"
#endif  