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
Status FuseResizePadAndConv(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  
      {"Conv2D",
          {
              {"MirrorPad",
                  {
                      {"ResizeBilinear"},
                      {"*"}
                  }
              },
              {"*"}
          }
      },  
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& conv_node = match.node;
        const NodeDef& mirror_pad_node = match.inputs[0].node;
        const NodeDef& weights_node = match.inputs[1].node;
        const NodeDef& resize_node = match.inputs[0].inputs[0].node;
        const NodeDef& pad_dims_node = match.inputs[0].inputs[1].node;
        new_nodes->push_back(weights_node);
        new_nodes->push_back(pad_dims_node);
        NodeDef fused_conv;
        fused_conv.set_op("FusedResizeAndPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(resize_node.input(0), &fused_conv);
        AddNodeInput(resize_node.input(1), &fused_conv);
        AddNodeInput(mirror_pad_node.input(1), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(resize_node, "align_corners", "resize_align_corners",
                     &fused_conv);
        CopyNodeAttr(mirror_pad_node, "mode", "mode", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);
        return OkStatus();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return OkStatus();
}
Status FuseResizeAndConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  
      {"Conv2D",
          {
              {"ResizeBilinear"},
              {"*"}
          }
      },  
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& conv_node = match.node;
        const NodeDef& resize_node = match.inputs[0].node;
        const NodeDef& weights_node = match.inputs[1].node;
        new_nodes->push_back(weights_node);
        NodeDef pad_dims_node;
        pad_dims_node.set_op("Const");
        pad_dims_node.set_name(conv_node.name() + "_dummy_paddings");
        SetNodeAttr("dtype", DT_INT32, &pad_dims_node);
        SetNodeTensorAttr<int32>("value", {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0},
                                 &pad_dims_node);
        new_nodes->push_back(pad_dims_node);
        NodeDef fused_conv;
        fused_conv.set_op("FusedResizeAndPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(resize_node.input(0), &fused_conv);
        AddNodeInput(resize_node.input(1), &fused_conv);
        AddNodeInput(pad_dims_node.name(), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(resize_node, "align_corners", "resize_align_corners",
                     &fused_conv);
        SetNodeAttr("mode", "REFLECT", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);
        return OkStatus();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return OkStatus();
}
Status FusePadAndConv(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  
      {"Conv2D",
          {
              {"MirrorPad",
                  {
                      {"*"},
                      {"*"},
                  }
              },
              {"*"}
          }
      },  
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& conv_node = match.node;
        CHECK_EQ("Conv2D", conv_node.op());
        const NodeDef& mirror_pad_node = match.inputs[0].node;
        CHECK_EQ("MirrorPad", mirror_pad_node.op());
        const NodeDef& weights_node = match.inputs[1].node;
        const NodeDef& input_node = match.inputs[0].inputs[0].node;
        const NodeDef& pad_dims_node = match.inputs[0].inputs[1].node;
        new_nodes->push_back(weights_node);
        new_nodes->push_back(input_node);
        new_nodes->push_back(pad_dims_node);
        NodeDef fused_conv;
        fused_conv.set_op("FusedPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(mirror_pad_node.input(0), &fused_conv);
        AddNodeInput(mirror_pad_node.input(1), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(mirror_pad_node, "mode", "mode", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);
        return OkStatus();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return OkStatus();
}
REGISTER_GRAPH_TRANSFORM("fuse_resize_pad_and_conv", FuseResizePadAndConv);
REGISTER_GRAPH_TRANSFORM("fuse_resize_and_conv", FuseResizeAndConv);
REGISTER_GRAPH_TRANSFORM("fuse_pad_and_conv", FusePadAndConv);
}  
}  