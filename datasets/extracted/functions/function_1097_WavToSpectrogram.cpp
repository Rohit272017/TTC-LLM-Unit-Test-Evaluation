#include "tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram.h"
#include <vector>
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
using tensorflow::DT_FLOAT;
using tensorflow::DT_UINT8;
using tensorflow::Output;
using tensorflow::TensorShape;
tensorflow::Status WavToSpectrogram(const tensorflow::string& input_wav,
                                    int32_t window_size, int32_t stride,
                                    float brightness,
                                    const tensorflow::string& output_image) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace tensorflow::ops;  
  Output file_reader =
      tensorflow::ops::ReadFile(root.WithOpName("input_wav"), input_wav);
  DecodeWav wav_decoder =
      DecodeWav(root.WithOpName("wav_decoder"), file_reader);
  Output spectrogram = AudioSpectrogram(root.WithOpName("spectrogram"),
                                        wav_decoder.audio, window_size, stride);
  Output brightness_placeholder =
      Placeholder(root.WithOpName("brightness_placeholder"), DT_FLOAT,
                  Placeholder::Attrs().Shape(TensorShape({})));
  Output mul = Mul(root.WithOpName("mul"), spectrogram, brightness_placeholder);
  Output min_const = Const(root.WithOpName("min_const"), 255.0f);
  Output min = Minimum(root.WithOpName("min"), mul, min_const);
  Output cast = Cast(root.WithOpName("cast"), min, DT_UINT8);
  Output expand_dims_const = Const(root.WithOpName("expand_dims_const"), -1);
  Output expand_dims =
      ExpandDims(root.WithOpName("expand_dims"), cast, expand_dims_const);
  Output squeeze = Squeeze(root.WithOpName("squeeze"), expand_dims,
                           Squeeze::Attrs().Axis({0}));
  Output png_encoder = EncodePng(root.WithOpName("png_encoder"), squeeze);
  tensorflow::ops::WriteFile file_writer = tensorflow::ops::WriteFile(
      root.WithOpName("output_image"), output_image, png_encoder);
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  tensorflow::Tensor brightness_tensor(DT_FLOAT, TensorShape({}));
  brightness_tensor.scalar<float>()() = brightness;
  TF_RETURN_IF_ERROR(
      session->Run({{"brightness_placeholder", brightness_tensor}}, {},
                   {"output_image"}, nullptr));
  return absl::OkStatus();
}