#define EIGEN_USE_THREADS
#include <algorithm>
#include <memory>
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/image_resizer_state.h"
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
namespace {
struct CachedInterpolation {
  int64_t start;
  int64_t end;
  float start_scale;
  float end_minus_one_scale;
  bool needs_bounding;
};
}  
template <typename Device, typename T>
class ResizeAreaOp : public OpKernel {
 public:
  explicit ResizeAreaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }
  template <bool NeedsXBounding>
  static void ComputePatchSumOf3Channels(float scale,
                                         const ImageResizerState& st,
                                         const std::vector<const T*>& y_ptrs,
                                         const std::vector<float>& y_scales,
                                         const CachedInterpolation& x_interp,
                                         float* output_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))
    float sum_0 = 0;
    float sum_1 = 0;
    float sum_2 = 0;
    for (int i = 0; i < y_ptrs.size(); ++i) {
      const T* ptr = y_ptrs[i];
      float scale_x = x_interp.start_scale;
      int64_t offset = 3 * BOUND_IF_NEEDED(x_interp.start, st.in_width);
      float sum_y_0 = static_cast<float>(ptr[offset + 0]) * scale_x;
      float sum_y_1 = static_cast<float>(ptr[offset + 1]) * scale_x;
      float sum_y_2 = static_cast<float>(ptr[offset + 2]) * scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (int64_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          int64_t offset = 3 * BOUND_IF_NEEDED(x, st.in_width);
          sum_y_0 += static_cast<float>(ptr[offset + 0]);
          sum_y_1 += static_cast<float>(ptr[offset + 1]);
          sum_y_2 += static_cast<float>(ptr[offset + 2]);
        }
        scale_x = x_interp.end_minus_one_scale;
        offset = 3 * BOUND_IF_NEEDED(x_interp.end - 1, st.in_width);
        sum_y_0 += static_cast<float>(ptr[offset + 0]) * scale_x;
        sum_y_1 += static_cast<float>(ptr[offset + 1]) * scale_x;
        sum_y_2 += static_cast<float>(ptr[offset + 2]) * scale_x;
      }
      sum_0 += sum_y_0 * y_scales[i];
      sum_1 += sum_y_1 * y_scales[i];
      sum_2 += sum_y_2 * y_scales[i];
    }
    output_ptr[0] = sum_0 * scale;
    output_ptr[1] = sum_1 * scale;
    output_ptr[2] = sum_2 * scale;
#undef BOUND_IF_NEEDED
  }
  template <bool NeedsXBounding>
  static void ComputePatchSum(float scale, const ImageResizerState& st,
                              const std::vector<const T*>& y_ptrs,
                              const std::vector<float>& y_scales,
                              const CachedInterpolation& x_interp,
                              float* output_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))
    const auto num_channels = st.channels;
    for (int64_t c = 0; c < num_channels; ++c) {
      float sum = 0;
      for (int i = 0; i < y_ptrs.size(); ++i) {
        const T* ptr = y_ptrs[i];
        float scale_x = x_interp.start_scale;
        float sum_y = static_cast<float>(
                          ptr[num_channels *
                                  BOUND_IF_NEEDED(x_interp.start, st.in_width) +
                              c]) *
                      scale_x;
        if (x_interp.start + 1 != x_interp.end) {
          for (int64_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
            sum_y += static_cast<float>(
                ptr[num_channels * BOUND_IF_NEEDED(x, st.in_width) + c]);
          }
          scale_x = x_interp.end_minus_one_scale;
          sum_y += static_cast<float>(
                       ptr[num_channels *
                               BOUND_IF_NEEDED(x_interp.end - 1, st.in_width) +
                           c]) *
                   scale_x;
        }
        sum += sum_y * y_scales[i];
      }
      output_ptr[c] = sum * scale;
    }
#undef BOUND_IF_NEEDED
  }
  void Compute(OpKernelContext* context) override {
    ImageResizerState st(align_corners_, false);
    st.ValidateAndCreateOutput(context);
    if (!context->status().ok()) return;
    typename TTypes<T, 4>::ConstTensor input_data(
        context->input(0).tensor<T, 4>());
    std::vector<CachedInterpolation> x_interps(st.out_width);
    for (int64_t x = 0; x < st.out_width; ++x) {
      auto& x_interp = x_interps[x];
      const float in_x = x * st.width_scale;
      const float in_x1 = (x + 1) * st.width_scale;
      int64_t v = std::floor(in_x);
      x_interp.start = v;
      x_interp.start_scale =
          v < in_x ? (v + 1 > in_x1 ? st.width_scale : v + 1 - in_x)
                   : (v + 1 > in_x1 ? in_x1 - v : 1.0);
      v = std::ceil(in_x1);
      x_interp.end = v;
      v = x_interp.end - 1;
      x_interp.end_minus_one_scale =
          v < in_x ? (v + 1 > in_x1 ? st.width_scale : v + 1 - in_x)
                   : (v + 1 > in_x1 ? in_x1 - v : 1.0);
      x_interp.needs_bounding =
          Bound(x_interp.start, st.in_width) != x_interp.start ||
          Bound(x_interp.end - 1, st.in_width) != (x_interp.end - 1);
    }
    if (st.channels == 3) {
      ComputeLoop<3>(st, x_interps, input_data);
    } else {
      ComputeLoop<-1>(st, x_interps, input_data);
    }
  }
  template <int64_t kKnownNumChannels>
  void ComputeLoop(const ImageResizerState& st,
                   const std::vector<CachedInterpolation>& x_interps,
                   typename TTypes<T, 4>::ConstTensor input_data) {
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();
    const T* const input_ptr = input_data.data();
    std::vector<float> y_scales;
    std::vector<const T*> y_ptrs;
    float scale = 1.0 / (st.height_scale * st.width_scale);
    float* output_ptr = output_data.data();
    for (int64_t b = 0; b < st.batch_size; ++b) {
      for (int64_t y = 0; y < st.out_height; ++y) {
        const float in_y = y * st.height_scale;
        const float in_y1 = (y + 1) * st.height_scale;
        const int64_t y_start = std::floor(in_y);
        const int64_t y_end = std::ceil(in_y1);
        y_scales.clear();
        y_ptrs.clear();
        for (int64_t i = y_start; i < y_end; ++i) {
          float scale_y;
          if (i < in_y) {
            scale_y = (i + 1 > in_y1 ? st.height_scale : i + 1 - in_y);
          } else {
            scale_y = (i + 1 > in_y1 ? in_y1 - i : 1.0);
          }
          y_scales.push_back(scale_y);
          y_ptrs.push_back(
              input_ptr + (b * st.in_height * st.in_width * st.channels +
                           Bound(i, st.in_height) * st.in_width * st.channels));
        }
        if (kKnownNumChannels == 3) {
          for (int64_t x = 0; x < st.out_width; ++x) {
            const CachedInterpolation& x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSumOf3Channels<true>(scale, st, y_ptrs, y_scales,
                                               x_interp, output_ptr);
            } else {
              ComputePatchSumOf3Channels<false>(scale, st, y_ptrs, y_scales,
                                                x_interp, output_ptr);
            }
            output_ptr += 3;
          }
        } else {
          for (int64_t x = 0; x < st.out_width; ++x) {
            const CachedInterpolation& x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSum<true>(scale, st, y_ptrs, y_scales, x_interp,
                                    output_ptr);
            } else {
              ComputePatchSum<false>(scale, st, y_ptrs, y_scales, x_interp,
                                     output_ptr);
            }
            output_ptr += st.channels;
          }
        }
      }
    }
  }
 private:
  static EIGEN_ALWAYS_INLINE int64_t Bound(int64_t val, int64_t limit) {
    return std::min(limit - 1, std::max(int64_t{0}, val));
  }
  bool align_corners_;
};
#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeArea")          \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeAreaOp<CPUDevice, T>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
}  