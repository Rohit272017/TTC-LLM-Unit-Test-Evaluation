#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_
#include <cstdint>
#include "ruy/profiler/instrumentation.h"  
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_custom_gemv.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"
#ifndef TFLITE_WITH_RUY
#include "tensorflow/lite/kernels/cpu_backend_gemm_eigen.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_gemmlowp.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_x86.h"
#endif
namespace tflite {
namespace cpu_backend_gemm {
#if !defined(TFLITE_WITH_RUY) && defined(TFLITE_X86_PLATFORM)
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplX86<LhsScalar, RhsScalar, AccumScalar,
                                      DstScalar, quantization_flavor> {};
#else
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                           DstScalar, quantization_flavor> {};
#if !defined(TFLITE_WITH_RUY)
template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingGemmlowp<SrcScalar, SrcScalar, std::int32_t,
                                    DstScalar, quantization_flavor> {};
#if !defined(GEMMLOWP_NEON)
template <typename SrcScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                               quantization_flavor> {};
template <typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               DstScalar, quantization_flavor> {};
template <QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               std::int8_t, quantization_flavor> {};
#endif  
template <>
struct GemmImpl<float, float, float, float, QuantizationFlavor::kFloatingPoint>
    : detail::GemmImplUsingEigen {};
#endif  
#endif  
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
          const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    TFLITE_DCHECK(false);
    return;
  }
  bool must_use_ruy = false;
  if (context->use_caching()) {
    must_use_ruy = true;
  }
  if (lhs_params.order != Order::kRowMajor ||
      rhs_params.order != Order::kColMajor ||
      dst_params.order != Order::kColMajor) {
    must_use_ruy = true;
  }
  if (must_use_ruy) {
    detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             quantization_flavor>::Run(lhs_params, lhs_data,
                                                       rhs_params, rhs_data,
                                                       dst_params, dst_data,
                                                       params, context);
    return;
  }
  const bool try_custom_gemv = (dst_params.cols == 1);
  if (try_custom_gemv) {
    if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
                           dst_params, dst_data, params, context)) {
      return;
    }
  }
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
           quantization_flavor>::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                                     dst_params, dst_data, params, context);
}
template <QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<int8_t>& lhs_params, const int8_t* lhs_data,
          const MatrixParams<int16_t>& rhs_params, const int16_t* rhs_data,
          const MatrixParams<int16_t>& dst_params, int16_t* dst_data,
          const GemmParams<int32_t, int16_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    TFLITE_DCHECK(false);
    return;
  }
  detail::GemmImplUsingRuy<int8_t, int16_t, int32_t, int16_t,
                           quantization_flavor>::Run(lhs_params, lhs_data,
                                                     rhs_params, rhs_data,
                                                     dst_params, dst_data,
                                                     params, context);
}
template <typename LhsScalar, typename RhsScalar,
          QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<int32_t>& dst_params, int32_t* dst_data,
          const GemmParams<int32_t, int32_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  ruy::profiler::ScopeLabel label2("cpu_backend_gemm::Gemm: general GEMM");
  detail::GemmImplUsingRuy<LhsScalar, RhsScalar, int32_t, int32_t,
                           quantization_flavor>::Run(lhs_params, lhs_data,
                                                     rhs_params, rhs_data,
                                                     dst_params, dst_data,
                                                     params, context);
}
}  
}  
#endif  