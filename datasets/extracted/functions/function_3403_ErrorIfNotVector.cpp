#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
namespace tensorflow {
namespace graph_transforms {
namespace {
Status ErrorIfNotVector(const Tensor& input, const string& input_name,
                        int expected_width) {
  if ((input.shape().dims() != 1) ||
      (input.shape().dim_size(0) != expected_width)) {
    return errors::InvalidArgument(
        input_name,
        " input to batch norm has bad shape: ", input.shape().DebugString());
  }
  return OkStatus();
}
Status GetScaleAndOffsetValues(const NodeMatch& match,
                               std::vector<float>* scale_values,
                               std::vector<float>* offset_values) {
  const NodeDef& batch_norm_node = match.node;
  CHECK(batch_norm_node.op() == "BatchNormWithGlobalNormalization" ||
        batch_norm_node.op() == "FusedBatchNorm");
  const bool is_fused = batch_norm_node.op() == "FusedBatchNorm";
  const int mean_idx = is_fused ? 3 : 1;
  const int var_idx = is_fused ? 4 : 2;
  const int beta_idx = is_fused ? 2 : 3;
  const int gamma_idx = is_fused ? 1 : 4;
  const string epsilon_attr = is_fused ? "epsilon" : "variance_epsilon";
  const bool scale_after_normalization =
      is_fused || batch_norm_node.attr().at("scale_after_normalization").b();
  const NodeDef& mean_node = match.inputs[mean_idx].node;
  CHECK_EQ("Const", mean_node.op());
  const NodeDef& variance_node = match.inputs[var_idx].node;
  CHECK_EQ("Const", variance_node.op());
  const NodeDef& beta_node = match.inputs[beta_idx].node;
  CHECK_EQ("Const", beta_node.op());
  const NodeDef& gamma_node = match.inputs[gamma_idx].node;
  CHECK_EQ("Const", gamma_node.op());
  Tensor mean = GetNodeTensorAttr(mean_node, "value");
  Tensor variance = GetNodeTensorAttr(variance_node, "value");
  Tensor beta = GetNodeTensorAttr(beta_node, "value");
  Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
  const float variance_epsilon = batch_norm_node.attr().at(epsilon_attr).f();
  const int64_t num_cols = mean.shape().dim_size(0);
  TF_RETURN_IF_ERROR(ErrorIfNotVector(variance, "Variance", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(beta, "Beta", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(gamma, "gamma", num_cols));
  scale_values->resize(num_cols);
  offset_values->resize(num_cols);
  if (scale_after_normalization) {
    for (int i = 0; i < num_cols; ++i) {
      (*scale_values)[i] =
          (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
          gamma.flat<float>()(i);
    }
  } else {
    for (int i = 0; i < num_cols; ++i) {
      (*scale_values)[i] =
          (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
    }
  }
  for (int i = 0; i < num_cols; ++i) {
    (*offset_values)[i] =
        (-mean.flat<float>()(i) * (*scale_values)[i]) + beta.flat<float>()(i);
  }
  return OkStatus();
}
Status FuseScaleOffsetToConvWeights(const std::vector<float>& scale_values,
                                    const std::vector<float>& offset_values,
                                    const NodeMatch& conv_node_match,
                                    const string& conv_output_name,
                                    std::vector<NodeDef>* new_nodes) {
  const NodeDef& conv_node = conv_node_match.node;
  const NodeDef& input_node = conv_node_match.inputs[0].node;
  const NodeDef& weights_node = conv_node_match.inputs[1].node;
  CHECK_EQ("Const", weights_node.op());
  Tensor weights = GetNodeTensorAttr(weights_node, "value");
  int64_t weights_cols;
  if (conv_node.op() == "Conv2D") {
    weights_cols = weights.shape().dim_size(3);
  } else if (conv_node.op() == "DepthwiseConv2dNative") {
    weights_cols = weights.shape().dim_size(2) * weights.shape().dim_size(3);
  } else {
    weights_cols = weights.shape().dim_size(1);
  }
  CHECK_EQ(weights_cols, scale_values.size());
  auto weights_vector = weights.flat<float>();
  Tensor scaled_weights(DT_FLOAT, weights.shape());
  auto scaled_weights_vector = scaled_weights.flat<float>();
  for (int64_t row = 0; row < weights_vector.dimension(0); ++row) {
    scaled_weights_vector(row) =
        weights_vector(row) * scale_values[row % weights_cols];
  }
  Tensor bias_offset(DT_FLOAT, {weights_cols});
  auto bias_offset_vector = bias_offset.flat<float>();
  for (int64_t col = 0; col < weights_cols; ++col) {
    bias_offset_vector(col) = offset_values[col];
  }
  NodeDef scaled_weights_node;
  scaled_weights_node.set_op("Const");
  scaled_weights_node.set_name(weights_node.name());
  SetNodeAttr("dtype", DT_FLOAT, &scaled_weights_node);
  SetNodeTensorAttr<float>("value", scaled_weights, &scaled_weights_node);
  new_nodes->push_back(scaled_weights_node);
  new_nodes->push_back(input_node);
  new_nodes->push_back(conv_node);
  NodeDef bias_offset_node;
  bias_offset_node.set_op("Const");
  bias_offset_node.set_name(conv_node.name() + "_bn_offset");
  SetNodeAttr("dtype", DT_FLOAT, &bias_offset_node);
  SetNodeTensorAttr<float>("value", bias_offset, &bias_offset_node);
  new_nodes->push_back(bias_offset_node);
  NodeDef bias_add_node;
  bias_add_node.set_op("BiasAdd");
  bias_add_node.set_name(conv_output_name);
  if (conv_node.attr().count("data_format")) {
    CopyNodeAttr(conv_node, "data_format", "data_format", &bias_add_node);
  }
  CopyNodeAttr(conv_node, "T", "T", &bias_add_node);
  AddNodeInput(conv_node.name(), &bias_add_node);
  AddNodeInput(bias_offset_node.name(), &bias_add_node);
  new_nodes->push_back(bias_add_node);
  return OkStatus();
}
Status FuseBatchNormWithConv(const NodeMatch& match,
                             std::vector<NodeDef>* new_nodes) {
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));
  const NodeDef& batch_norm_node = match.node;
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale_values, offset_values, match.inputs[0],
                                   batch_norm_node.name(), new_nodes));
  return OkStatus();
}
Status FuseBatchNormWithBatchToSpace(const NodeMatch& match,
                                     std::vector<NodeDef>* new_nodes) {
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));
  const NodeDef& batch_norm_node = match.node;
  const NodeMatch& batch_to_space_node_match = match.inputs[0];
  const NodeMatch& conv_node_match = batch_to_space_node_match.inputs[0];
  const NodeDef& batch_to_space_node = batch_to_space_node_match.node;
  const NodeDef& conv_node = conv_node_match.node;
  string biasadd_name = conv_node.name() + "/biasadd";
  TF_RETURN_IF_ERROR(FuseScaleOffsetToConvWeights(
      scale_values, offset_values, conv_node_match, biasadd_name, new_nodes));
  NodeDef new_batch_to_space_node = batch_to_space_node;
  new_batch_to_space_node.set_name(batch_norm_node.name());
  new_batch_to_space_node.set_input(0, biasadd_name);
  new_nodes->push_back(batch_to_space_node_match.inputs[1].node);
  new_nodes->push_back(batch_to_space_node_match.inputs[2].node);
  new_nodes->push_back(new_batch_to_space_node);
  return OkStatus();
}
Status FuseBatchNormWithConvConcat(const NodeMatch& match,
                                   std::vector<NodeDef>* new_nodes) {
  std::vector<float> scale_values;
  std::vector<float> offset_values;
  TF_RETURN_IF_ERROR(
      GetScaleAndOffsetValues(match, &scale_values, &offset_values));
  const NodeDef& batch_norm_node = match.node;
  const NodeMatch& concat_node_match = match.inputs[0];
  NodeDef concat_node = concat_node_match.node;
  CHECK_EQ("ConcatV2", concat_node.op());
  NodeDef axis_node = concat_node_match.inputs[2].node;
  CHECK_EQ("Const", axis_node.op());
  Tensor axis = GetNodeTensorAttr(axis_node, "value");
  int32_t axis_scalar = (axis.scalar<int32>())();
  std::vector<float> scale0(scale_values);
  std::vector<float> offset0(offset_values);
  std::vector<float> scale1(scale_values);
  std::vector<float> offset1(offset_values);
  if (axis_scalar == 3) {
    const NodeDef& weights0_node = concat_node_match.inputs[0].inputs[1].node;
    Tensor weights0 = GetNodeTensorAttr(weights0_node, "value");
    const int64_t split_cols = weights0.shape().dim_size(3);
    scale0.erase(scale0.begin() + split_cols, scale0.end());
    offset0.erase(offset0.begin() + split_cols, offset0.end());
    scale1.erase(scale1.begin(), scale1.begin() + split_cols);
    offset1.erase(offset1.begin(), offset1.begin() + split_cols);
  }
  const string concat0_output_name = concat_node.name() + "_bn_in0";
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale0, offset0, concat_node_match.inputs[0],
                                   concat0_output_name, new_nodes));
  const string concat1_output_name = concat_node.name() + "_bn_in1";
  TF_RETURN_IF_ERROR(
      FuseScaleOffsetToConvWeights(scale1, offset1, concat_node_match.inputs[1],
                                   concat1_output_name, new_nodes));
  new_nodes->push_back(concat_node_match.inputs[2].node);
  concat_node.set_name(batch_norm_node.name());
  concat_node.set_input(0, concat0_output_name);
  concat_node.set_input(1, concat1_output_name);
  new_nodes->push_back(concat_node);
  return OkStatus();
}
}  
Status FoldOldBatchNorms(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  GraphDef current_graph_def = input_graph_def;
  bool did_graph_change;
  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  
      {"BatchNormWithGlobalNormalization|FusedBatchNorm",    
        {
          {"Conv2D|DepthwiseConv2dNative",                          
            {
              {"*"},                          
              {"Const"},                      
            }
          },
          {"Const"},                          
          {"Const"},                          
          {"Const"},                          
          {"Const"},                          
        }
      },  
        [&did_graph_change](const NodeMatch& match,
                            const std::set<string>& input_nodes,
                            const std::set<string>& output_nodes,
                            std::vector<NodeDef>* new_nodes) {
          TF_RETURN_IF_ERROR(FuseBatchNormWithConv(match, new_nodes));
          did_graph_change = true;
          return OkStatus();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);
  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  
        {"BatchNormWithGlobalNormalization|FusedBatchNorm",    
         {
             {"BatchToSpaceND",                  
              {
                  {"Conv2D|DepthwiseConv2dNative",                     
                   {
                       {"*"},                    
                       {"Const"},                
                   }
                  },
                  {"Const"},                     
                  {"Const"},                     
              }
             },
             {"Const"},                          
             {"Const"},                          
             {"Const"},                          
             {"Const"},                          
         }
        },  
        [&did_graph_change](const NodeMatch& match,
                            const std::set<string>& input_nodes,
                            const std::set<string>& output_nodes,
                            std::vector<NodeDef>* new_nodes) {
          TF_RETURN_IF_ERROR(FuseBatchNormWithBatchToSpace(match, new_nodes));
          did_graph_change = true;
          return OkStatus();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);
  do {
    did_graph_change = false;
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  
      {"BatchNormWithGlobalNormalization|FusedBatchNorm",    
        {
          {"ConcatV2|Concat",                     
            {
              {"Conv2D|DepthwiseConv2dNative",                          
                {
                  {"*"},                          
                  {"Const"},                      
                }
              },
              {"Conv2D|DepthwiseConv2dNative",                          
                {
                  {"*"},                          
                  {"Const"},                      
                }
              },
              {"Const"},                          
            },
          },
          {"Const"},                          
          {"Const"},                          
          {"Const"},                          
          {"Const"},                          
        }
      },  
        [&did_graph_change](const NodeMatch& match,
                            const std::set<string>& input_nodes,
                            const std::set<string>& output_nodes,
                            std::vector<NodeDef>* new_nodes) {
          TF_RETURN_IF_ERROR(FuseBatchNormWithConvConcat(match, new_nodes));
          did_graph_change = true;
          return OkStatus();
        },
        {}, &replaced_graph_def));
    current_graph_def = replaced_graph_def;
  } while (did_graph_change);
  *output_graph_def = current_graph_def;
  return OkStatus();
}
REGISTER_GRAPH_TRANSFORM("fold_old_batch_norms", FoldOldBatchNorms);
}  
}  