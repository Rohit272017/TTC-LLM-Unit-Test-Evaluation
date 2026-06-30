#ifndef TENSORFLOW_CORE_KERNELS_CONV_2D_H_
#define TENSORFLOW_CORE_KERNELS_CONV_2D_H_
#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/Tensor"  
#include "xla/tsl/framework/convolution/eigen_spatial_convolutions.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"
static bool Conv2dUseFp16Accumulate() {
  static bool use_fp16_accumulate = []() {
    const char* env = std::getenv("TF_CONV2D_USE_FP16_ACCUMULATE");
    return (env != nullptr) && (absl::string_view(env) == "1");
  }();
  return use_fp16_accumulate;
}
namespace tensorflow {
namespace functor {
template <typename Device, typename Input, typename Filter, typename Output,
          typename OutputKernel>
void SpatialConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int row_stride, int col_stride,
                            int row_dilation, int col_dilation,
                            const Eigen::PaddingType& padding,
                            const OutputKernel& output_kernel,
                            int padding_top = 0, int padding_bottom = 0,
                            int padding_left = 0, int padding_right = 0) {
  output.device(d) = Eigen::SpatialConvolution(
      input, filter, col_stride, row_stride, padding, col_dilation,
      row_dilation, output_kernel, padding_left, padding_right, padding_top,
      padding_bottom);
}
template <typename Device, typename T,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct SpatialConvolution {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, int row_dilation, int col_dilation,
                  const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           row_dilation, col_dilation, padding, output_kernel);
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           row_dilation, col_dilation, padding, output_kernel);
  }
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, int row_dilation, int col_dilation,
                  int padding_top, int padding_bottom, int padding_left,
                  int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(
        d, output, input, filter, row_stride, col_stride, row_dilation,
        col_dilation, Eigen::PaddingType::PADDING_VALID, output_kernel,
        padding_top, padding_bottom, padding_left, padding_right);
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(
        d, output, input, filter, row_stride, col_stride, row_dilation,
        col_dilation, Eigen::PaddingType::PADDING_VALID, output_kernel,
        padding_top, padding_bottom, padding_left, padding_right);
  }
};
template <typename Device, typename OutputKernel>
struct SpatialConvolution<Device, Eigen::half, OutputKernel> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride, padding, col_dilation,
          row_dilation, output_kernel);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                    col_stride, row_stride, padding,
                                    col_dilation, row_dilation, output_kernel)
              .template cast<Eigen::half>();
    }
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride, padding, col_dilation,
          row_dilation, output_kernel);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(input.template cast<float>(),
                                    filter.template cast<float>(), col_stride,
                                    row_stride, padding, col_dilation,
                                    row_dilation, output_kernel)
              .template cast<Eigen::half>();
    }
  }
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride,
          Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
          output_kernel, padding_left, padding_right, padding_top,
          padding_bottom);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(
              input.cast<float>(), filter.cast<float>(), col_stride, row_stride,
              Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
              output_kernel, padding_left, padding_right, padding_top,
              padding_bottom)
              .template cast<Eigen::half>();
    }
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride,
          Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
          output_kernel, padding_left, padding_right, padding_top,
          padding_bottom);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(
              input.template cast<float>(), filter.template cast<float>(),
              col_stride, row_stride, Eigen::PaddingType::PADDING_VALID,
              col_dilation, row_dilation, output_kernel, padding_left,
              padding_right, padding_top, padding_bottom)
              .template cast<Eigen::half>();
    }
  }
};
template <typename Device, typename OutputKernel>
struct SpatialConvolution<Device, Eigen::bfloat16, OutputKernel> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::bfloat16, 4>::Tensor output,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor input,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                  col_stride, row_stride, padding, col_dilation,
                                  row_dilation, output_kernel)
            .template cast<Eigen::bfloat16>();
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(input.template cast<float>(),
                                  filter.template cast<float>(), col_stride,
                                  row_stride, padding, col_dilation,
                                  row_dilation, output_kernel)
            .template cast<Eigen::bfloat16>();
  }
  void operator()(const Device& d,
                  typename TTypes<Eigen::bfloat16, 4>::Tensor output,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor input,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(
            input.cast<float>(), filter.cast<float>(), col_stride, row_stride,
            Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
            output_kernel, padding_left, padding_right, padding_top,
            padding_bottom)
            .template cast<Eigen::bfloat16>();
  }
  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(
            input.template cast<float>(), filter.template cast<float>(),
            col_stride, row_stride, Eigen::PaddingType::PADDING_VALID,
            col_dilation, row_dilation, output_kernel, padding_left,
            padding_right, padding_top, padding_bottom)
            .template cast<Eigen::bfloat16>();
  }
};
template <typename Device, typename T>
struct SpatialConvolutionBackwardInputFunc {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation) {
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
        filter, output_backward, input_backward.dimension(2),
        input_backward.dimension(1), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};
template <typename T>
struct SpatialConvolutionBackwardInputFunc<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation) {
    To32Bit(input_backward).device(d) = Eigen::SpatialConvolutionBackwardInput(
        To32Bit(filter), To32Bit(output_backward), input_backward.dimension(2),
        input_backward.dimension(1), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};
template <typename Device, typename T>
struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex padded_cols, Eigen::DenseIndex padded_rows,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation, Eigen::DenseIndex pad_left,
                  Eigen::DenseIndex pad_top) {
    input_backward.device(d) =
        Eigen::SpatialConvolutionBackwardInput(
            filter, output_backward, padded_cols, padded_rows, col_stride,
            row_stride, col_dilation, row_dilation)
            .eval()
            .slice(Eigen::DSizes<Eigen::DenseIndex, 4>{0, pad_left, pad_top, 0},
                   input_backward.dimensions());
  }
};
template <typename T>
struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc<Eigen::GpuDevice,
                                                              T> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex padded_cols, Eigen::DenseIndex padded_rows,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation, Eigen::DenseIndex pad_left,
                  Eigen::DenseIndex pad_top) {
    To32Bit(input_backward).device(d) =
        Eigen::SpatialConvolutionBackwardInput(
            To32Bit(filter), To32Bit(output_backward), padded_cols, padded_rows,
            col_stride, row_stride, col_dilation, row_dilation)
            .eval()
            .slice(Eigen::DSizes<Eigen::DenseIndex, 4>{0, pad_left, pad_top, 0},
                   input_backward.dimensions());
  }
};
template <typename Device, typename T,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct MatMulConvFunctor {
  void operator()(
      const Device& d, typename TTypes<T, 2>::Tensor out,
      typename TTypes<T, 2>::ConstTensor in0,
      typename TTypes<T, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    out.device(d) = in0.contract(in1, dim_pair, output_kernel);
  }
};
template <typename Device, typename OutputKernel>
struct MatMulConvFunctor<Device, Eigen::half, OutputKernel> {
  void operator()(
      const Device& d, typename TTypes<Eigen::half, 2>::Tensor out,
      typename TTypes<Eigen::half, 2>::ConstTensor in0,
      typename TTypes<Eigen::half, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      out.device(d) = in0.contract(in1, dim_pair, output_kernel);
    } else {
      out.device(d) =
          in0.cast<float>()
              .contract(in1.template cast<float>(), dim_pair, output_kernel)
              .template cast<Eigen::half>();
    }
  }
};
template <typename Device, typename OutputKernel>
struct MatMulConvFunctor<Device, Eigen::bfloat16, OutputKernel> {
  void operator()(
      const Device& d, typename TTypes<Eigen::bfloat16, 2>::Tensor out,
      typename TTypes<Eigen::bfloat16, 2>::ConstTensor in0,
      typename TTypes<Eigen::bfloat16, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    out.device(d) = in0.cast<float>()
                        .contract(in1.cast<float>(), dim_pair, output_kernel)
                        .template cast<Eigen::bfloat16>();
  }
};
template <typename Device, typename T, typename IndexType, int NDIMS>
struct TransformFilter {
  void operator()(const Device& d, FilterTensorFormat dst_filter_format,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out) {
    Eigen::DSizes<IndexType, NDIMS - 2> spatial_dims;
    for (int i = 0; i < spatial_dims.rank(); ++i) {
      spatial_dims[i] = in.dimension(i);
    }
    Eigen::DSizes<IndexType, 3> merged_dims;
    merged_dims[0] = spatial_dims.TotalSize();  
    merged_dims[1] = in.dimension(NDIMS - 2);   
    merged_dims[2] = in.dimension(NDIMS - 1);   
    Eigen::DSizes<IndexType, 3> shuffling_perm;
    Eigen::DSizes<IndexType, NDIMS> expanded_dims;
    if (dst_filter_format == FORMAT_OIHW) {
      shuffling_perm = Eigen::DSizes<IndexType, 3>(2, 1, 0);
      expanded_dims[0] = merged_dims[2];  
      expanded_dims[1] = merged_dims[1];  
      for (int i = 0; i < spatial_dims.rank(); ++i) {
        expanded_dims[2 + i] = spatial_dims[i];
      }
    } else if (dst_filter_format == FORMAT_OHWI) {
      shuffling_perm = Eigen::DSizes<IndexType, 3>(2, 0, 1);
      expanded_dims[0] = merged_dims[2];          
      expanded_dims[NDIMS - 1] = merged_dims[1];  
      for (int i = 0; i < spatial_dims.rank(); ++i) {
        expanded_dims[1 + i] = spatial_dims[i];
      }
    } else {
      DCHECK(false) << "Unsupported destination filter format: "
                    << ToString(dst_filter_format);
    }
    out.device(d) =
        in.reshape(merged_dims).shuffle(shuffling_perm).reshape(expanded_dims);
  }
};
template <typename Device, typename T, typename IndexType>
struct TransformDepth {
  void operator()(const Device& d,
                  typename TTypes<T, 4, IndexType>::ConstTensor in,
                  const Eigen::DSizes<IndexType, 4>& shuffle,
                  typename TTypes<T, 4, IndexType>::Tensor out) {
    Eigen::DSizes<IndexType, 3> merged_dims;
    Eigen::DSizes<IndexType, 4> expanded_dims;
    Eigen::DSizes<IndexType, 3> new_shuffle;
    if (shuffle[1] == 2 && shuffle[2] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = shuffle[0];
      new_shuffle[1] = 2;
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(shuffle[0]);
      expanded_dims[1] = in.dimension(2);
      expanded_dims[2] = in.dimension(3);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 2 && shuffle[1] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = 2;
      new_shuffle[1] = shuffle[2];
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(2);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(shuffle[2]);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 0 && shuffle[1] == 3 && shuffle[2] == 1 &&
               shuffle[3] == 2) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1) * in.dimension(2);
      merged_dims[2] = in.dimension(3);
      new_shuffle[0] = 0;
      new_shuffle[1] = 2;
      new_shuffle[2] = 1;
      expanded_dims[0] = in.dimension(0);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(1);
      expanded_dims[3] = in.dimension(2);
    } else {
      assert(false && "unexpected shuffle");
    }
    out.device(d) =
        in.reshape(merged_dims).shuffle(new_shuffle).reshape(expanded_dims);
  }
};
template <typename Device, typename T, typename IndexType, int NDIMS>
struct PadInput {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  const std::array<int, NDIMS - 2>& padding_left,
                  const std::array<int, NDIMS - 2>& padding_right,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out,
                  TensorFormat format, const T& padding_value) {
    Eigen::array<Eigen::IndexPair<IndexType>, NDIMS> padding;
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'N')] = {0, 0};
    for (int i = 0; i < NDIMS - 2; ++i) {
      padding[GetTensorDimIndex<NDIMS - 2>(format, '0' + i)] = {
          padding_left[i], padding_right[i]};
    }
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'C')] = {0, 0};
    out.device(d) = in.pad(padding, padding_value);
  }
};
template <typename Device, typename T, int NDIMS>
struct NHWCToNCHW {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};
template <typename Device, typename T, int NDIMS>
struct NCHWToNHWC {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};
template <typename Device, typename T, bool conjugate = false>
struct SwapDimension1And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const absl::Span<const int64_t>& input_dims, T* out);
};
template <typename Device, typename T, bool conjugate = false>
struct SwapDimension0And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const absl::Span<const int64_t>& input_dims, T* out);
};
template <typename Device, typename T, int NDIMS>
struct ReverseTransformFilter {
  void operator()(const Device& d, FilterTensorFormat src_filter_format,
                  typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};
}  
template <class T>
class ConvAlgorithmMap;
template <>
class ConvAlgorithmMap<Eigen::ThreadPoolDevice> {};
}  
#endif  