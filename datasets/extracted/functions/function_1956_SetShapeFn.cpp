#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
REGISTER_OP("FFT")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
REGISTER_OP("IFFT")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
REGISTER_OP("FFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });
REGISTER_OP("IFFT2D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    });
REGISTER_OP("FFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });
REGISTER_OP("IFFT3D")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    });
REGISTER_OP("FFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
REGISTER_OP("IFFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
Status RFFTShape(InferenceContext* c, const bool forward, const int rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  ShapeHandle unused_shape;
  DimensionHandle unused_dim;
  ShapeHandle fft_length_input = c->input(1);
  TF_RETURN_IF_ERROR(c->WithRank(fft_length_input, 1, &unused_shape));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(fft_length_input, 0), rank, &unused_dim));
  const Tensor* fft_length_tensor = c->input_tensor(1);
  if (fft_length_tensor == nullptr) {
    for (int i = 0; i < rank; ++i) {
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->UnknownDim(), &out));
    }
  } else {
    auto fft_length_as_vec = fft_length_tensor->vec<int32>();
    for (int i = 0; i < rank; ++i) {
      auto dim = forward && i == rank - 1 && fft_length_as_vec(i) != 0
                     ? fft_length_as_vec(i) / 2 + 1
                     : fft_length_as_vec(i);
      TF_RETURN_IF_ERROR(c->ReplaceDim(out, -rank + i, c->MakeDim(dim), &out));
    }
  }
  c->set_output(0, out);
  return absl::OkStatus();
}
REGISTER_OP("RFFT")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 1); });
REGISTER_OP("IRFFT")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 1); });
REGISTER_OP("RFFT2D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 2); });
REGISTER_OP("IRFFT2D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 2); });
REGISTER_OP("RFFT3D")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, true, 3); });
REGISTER_OP("IRFFT3D")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) { return RFFTShape(c, false, 3); });
REGISTER_OP("RFFTND")
    .Input("input: Treal")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Tcomplex")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
REGISTER_OP("IRFFTND")
    .Input("input: Tcomplex")
    .Input("fft_length: int32")
    .Input("axes: int32")
    .Output("output: Treal")
    .Attr("Treal: {float32, float64} = DT_FLOAT")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });
REGISTER_OP("BatchFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT");
REGISTER_OP("BatchIFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT");
REGISTER_OP("BatchFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT2D");
REGISTER_OP("BatchIFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT2D");
REGISTER_OP("BatchFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use FFT3D");
REGISTER_OP("BatchIFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(15, "Use IFFT3D");
}  