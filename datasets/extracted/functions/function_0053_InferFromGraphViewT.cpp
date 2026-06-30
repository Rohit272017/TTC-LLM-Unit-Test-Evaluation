#include "tensorflow/core/grappler/utils/frame.h"
#include <deque>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace grappler {
namespace {}  
template <typename GraphViewT>
inline Status FrameView::InferFromGraphViewT(const GraphViewT& graph_view) {
  if (is_inferred_) {
    return errors::Internal("FrameView was already inferred from the graph");
  }
  is_inferred_ = true;
  std::deque<int> ready_node_indices;
  for (const auto& node : graph_view.GetNodes()) {
    if (node.NumRegularFanins() + node.NumControllingFanins() == 0) {
      ready_node_indices.push_back(node.node_index());
      node_to_frames_[node.node()] = node_has_no_frames_;
    }
  }
  const auto* graph = graph_view.graph();
  absl::flat_hash_map<string, int> frame_name_to_id;
  auto process_fanout = [this, graph](
                            absl::flat_hash_map<string, int>* frame_name_to_id,
                            std::deque<int>* ready_node_indices,
                            const NodeDef* ready_node, int fanout_node_index) {
    const NodeDef* fanout_node = &graph->node(fanout_node_index);
    if (!node_to_frames_.contains(fanout_node)) {
      std::vector<int> frame_ids = node_to_frames_[ready_node];
      if (IsExit(*ready_node)) {
        frame_ids.pop_back();
      }
      if (IsEnter(*fanout_node)) {
        const AttrValue* frame_name_attr =
            AttrSlice(*fanout_node).Find("frame_name");
        if (!frame_name_attr) {
          return errors::InvalidArgument(
              "Missing frame name for the Enter node: ",
              SummarizeNodeDef(*fanout_node));
        }
        const string& frame_name = frame_name_attr->s();
        int frame_id;
        if (frame_name_to_id->contains(frame_name)) {
          frame_id = (*frame_name_to_id)[frame_name];
        } else {
          frame_id = static_cast<int>(frame_name_to_id->size());
          (*frame_name_to_id)[frame_name] = frame_id;
        }
        frame_ids.push_back(frame_id);
      }
      ready_node_indices->push_back(fanout_node_index);
      node_to_frames_[fanout_node] = std::move(frame_ids);
    } else {
      std::vector<int> frame_ids_fanout = node_to_frames_[fanout_node];
      std::vector<int> frame_ids_node = node_to_frames_[ready_node];
      if (IsEnter(*fanout_node)) {
        frame_ids_fanout.pop_back();
      }
      if (IsExit(*ready_node)) {
        frame_ids_node.pop_back();
      }
      if (frame_ids_node != frame_ids_fanout) {
        return errors::InvalidArgument(
            "Invalid graph: Frame ids for node ", ready_node->name(),
            " does not match frame ids for it's fanout ", fanout_node->name());
      }
    }
    return absl::OkStatus();
  };
  while (!ready_node_indices.empty()) {
    const int ready_node_index = ready_node_indices.front();
    ready_node_indices.pop_front();
    const auto* ready_node_view = graph_view.GetNode(ready_node_index);
    const NodeDef* ready_node_def = ready_node_view->node();
    for (const auto& regular_fanouts_port_i :
         ready_node_view->GetRegularFanouts()) {
      for (const auto& regular_fanout : regular_fanouts_port_i) {
        TF_RETURN_IF_ERROR(process_fanout(&frame_name_to_id,
                                          &ready_node_indices, ready_node_def,
                                          regular_fanout.node_index()));
      }
    }
    for (const auto& controlled_fanout :
         ready_node_view->GetControlledFanouts()) {
      TF_RETURN_IF_ERROR(process_fanout(&frame_name_to_id, &ready_node_indices,
                                        ready_node_def,
                                        controlled_fanout.node_index()));
    }
  }
  num_frames_ = static_cast<int>(frame_name_to_id.size());
  return absl::OkStatus();
}
Status FrameView::InferFromGraphView(const utils::GraphView& graph_view) {
  return InferFromGraphViewT(graph_view);
}
Status FrameView::InferFromGraphView(
    const utils::MutableGraphView& graph_view) {
  return InferFromGraphViewT(graph_view);
}
Status FrameView::InferFromGraph(const GraphDef& graph) {
  Status status;
  utils::GraphView graph_view(&graph, &status);
  TF_RETURN_IF_ERROR(status);
  return InferFromGraphViewT(graph_view);
}
const std::vector<int>& FrameView::Frames(const NodeDef& node) const {
  DCHECK(is_inferred_) << "FrameView is not initialized";
  auto frames = node_to_frames_.find(&node);
  if (frames == node_to_frames_.end()) {
    LOG(WARNING) << "Node '" << node.name()
                 << "' doesn't belong to the graph used for initialization";
    return node_has_no_frames_;
  } else {
    return frames->second;
  }
}
bool FrameView::IsInFrame(const NodeDef& node) const {
  return !Frames(node).empty();
}
}  
}  