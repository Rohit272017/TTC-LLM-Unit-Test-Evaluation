#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
REGISTER_OP("CTCLoss")
    .Input("inputs: T")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Attr("preprocess_collapse_repeated: bool = false")
    .Attr("ctc_merge_repeated: bool = true")
    .Attr("ignore_longer_outputs_than_inputs: bool = false")
    .Output("loss: T")
    .Output("gradient: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle labels_indices;
      ShapeHandle labels_values;
      ShapeHandle sequence_length;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &labels_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &labels_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sequence_length));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(labels_indices, 0),
                                  c->Dim(labels_values, 0), &unused));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));
      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, inputs);
      return absl::OkStatus();
    });
REGISTER_OP("CTCLossV2")
    .Input("inputs: float")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Attr("preprocess_collapse_repeated: bool = false")
    .Attr("ctc_merge_repeated: bool = true")
    .Attr("ignore_longer_outputs_than_inputs: bool = false")
    .Output("loss: float")
    .Output("gradient: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle labels_indices;
      ShapeHandle labels_values;
      ShapeHandle sequence_length;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &labels_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &labels_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sequence_length));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(labels_indices, 0),
                                  c->Dim(labels_values, 0), &unused));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));
      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, inputs);
      return absl::OkStatus();
    });
REGISTER_OP("CTCGreedyDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("merge_repeated: bool = false")
    .Attr("blank_index: int = -1")
    .Output("decoded_indices: int64")
    .Output("decoded_values: int64")
    .Output("decoded_shape: int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      DimensionHandle total_decoded_outputs = c->UnknownDim();
      c->set_output(0, c->Matrix(total_decoded_outputs, 2));
      c->set_output(1, c->Vector(total_decoded_outputs));
      c->set_output(2, c->Vector(2));
      c->set_output(3, c->Matrix(batch_size, 1));
      return absl::OkStatus();
    });
REGISTER_OP("CTCBeamSearchDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("top_paths: int >= 1")
    .Attr("merge_repeated: bool = true")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      int32_t top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));
      int out_idx = 0;
      for (int i = 0; i < top_paths; ++i) {  
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      ShapeHandle shape_v = c->Vector(2);
      for (int i = 0; i < top_paths; ++i) {  
        c->set_output(out_idx++, shape_v);
      }
      c->set_output(out_idx++, c->Matrix(batch_size, top_paths));
      return absl::OkStatus();
    });
}  