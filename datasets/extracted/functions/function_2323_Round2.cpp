#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
namespace grappler {
const char kAttrInputSrc[] = "input_source_";
const char kAttrSrcDevice[] = "send_device";
const char kAttrDstDevice[] = "recv_device";
const char kAttrTensorName[] = "tensor_name";
const char kChannelDevice[] = "Channel";
const char kStreaming[] = "_streaming";
namespace {
using ::tensorflow::strings::HumanReadableNumBytes;
float Round2(const float x) {
  return ::round(100.0 * x) / 100.0;
}
Costs& FindOrCreateZero(const string& op_name,
                        std::map<string, Costs>* op_cost) {
  auto it = op_cost->find(op_name);
  if (it == op_cost->end()) {
    it = op_cost->emplace(op_name, Costs::ZeroCosts()).first;
  }
  return it->second;
}
struct RecvNodeDescriptor {
  const NodeDef* node;
  const int port_num;
  const string device;
  RecvNodeDescriptor(const NodeDef* node_, const int port_num_,
                     const string& device_)
      : node(node_), port_num(port_num_), device(device_) {}
};
struct RecvNodeDescriptorHash {
  std::size_t operator()(const RecvNodeDescriptor& recv_node) const {
    return std::hash<const NodeDef*>()(recv_node.node) ^
           std::hash<int>()(recv_node.port_num) ^
           std::hash<string>()(recv_node.device);
  }
};
struct RecvNodeDescriptorEqual {
  bool operator()(const RecvNodeDescriptor& a,
                  const RecvNodeDescriptor& b) const {
    return a.node == b.node && a.port_num == b.port_num && a.device == b.device;
  }
};
void UpdateDeviceAnnotationState(const NodeDef* node,
                                 const NodeState& node_state,
                                 DeviceState* device) {
  if (node->attr().count(kOutputShapes) == 0) return;
  int64_t execution_count = node->attr().count(kExecutionCount) == 0
                                ? 1
                                : node->attr().at(kExecutionCount).i();
  auto& shape_annotation_stats = device->shape_annotation_stats;
  shape_annotation_stats.num_ops_annotated += 1;
  shape_annotation_stats.num_ops_executed += execution_count;
  shape_annotation_stats.num_ops_executed_more_than_once +=
      execution_count > 1 ? 1 : 0;
  shape_annotation_stats.num_ops_with_incompatible_shapes +=
      node_state.shape_incompatible ? 1 : 0;
  shape_annotation_stats.num_ops_with_dynamic_shapes +=
      (execution_count > 1 && node->attr().count(kOutputSame) == 0) ? 1 : 0;
}
bool IsStreamingPort(const NodeDef& node, const int port) {
  if (!node.attr().contains(kStreaming)) return false;
  auto& attr_list = node.attr().at(kStreaming).list();
  bool is_streaming_port = false;
  if (port >= 0 && port < attr_list.b().size()) {
    is_streaming_port = attr_list.b(port);
  }
  return is_streaming_port;
}
}  
void LIFOManager::AddNode(const NodeDef* node) {
  if (IsMerge(*node)) {
    nodes_.push_front(node);
  } else {
    nodes_.push_back(node);
  }
}
const NodeDef* LIFOManager::GetCurrNode() {
  CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
  if (curr_pos_ == nodes_.end()) {
    curr_pos_ = --(nodes_.rbegin().base());  
  }
  return *curr_pos_;
}
void LIFOManager::RemoveCurrNode() {
  GetCurrNode();
  nodes_.erase(curr_pos_);
  curr_pos_ = nodes_.end();  
}
HeapReadyManager::HeapReadyManager() : ReadyNodeManager() {
  std::make_heap(nodes_.begin(), nodes_.end());
}
Status HeapReadyManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_map) {
  node_map_ = node_map;
  nodes_.clear();
  curr_node_ = nullptr;
  greater_ = Greater();
  return absl::OkStatus();
}
void HeapReadyManager::AddNode(const NodeDef* node) {
  nodes_.push_back(node);
  std::push_heap(nodes_.begin(), nodes_.end(), greater_);
}
const NodeDef* HeapReadyManager::GetCurrNode() {
  if (curr_node_) return curr_node_;
  if (nodes_.empty()) {
    CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
  }
  const std::string node_name = nodes_.front()->name();
  curr_node_ = nodes_.front();
  std::pop_heap(nodes_.begin(), nodes_.end(), greater_);
  nodes_.pop_back();
  return curr_node_;
}
void HeapReadyManager::RemoveCurrNode() {
  if (curr_node_) {
    curr_node_ = nullptr;
  } else {
    std::pop_heap(nodes_.begin(), nodes_.end(), greater_);
    nodes_.pop_back();
  }
}
bool HeapReadyManager::Empty() const {
  return nodes_.empty() && curr_node_ == nullptr;
}
bool FirstReadyCmp(
    const std::unordered_map<const NodeDef*, NodeState>* node_map,
    const NodeDef* a, const NodeDef* b) {
  if (node_map->at(a).time_ready == node_map->at(b).time_ready) {
    return a->name().compare(b->name()) > 0;
  } else {
    return node_map->at(a).time_ready > node_map->at(b).time_ready;
  }
}
std::function<bool(const NodeDef*, const NodeDef*)>
FirstReadyManager::Greater() {
  auto greater = [this](const NodeDef* a, const NodeDef* b) -> bool {
    return FirstReadyCmp(node_map_, a, b);
  };
  return greater;
}
std::function<bool(const NodeDef*, const NodeDef*)>
PriorityReadyManager::Greater() {
  auto greater = [this](const NodeDef* a, const NodeDef* b) -> bool {
    auto pri_a = node_priority_.at(a->name());
    auto pri_b = node_priority_.at(b->name());
    if (pri_a == pri_b) {
      return FirstReadyCmp(node_map_, a, b);
    }
    return pri_a > pri_b;
  };
  return greater;
}
void PriorityReadyManager::AddNode(const NodeDef* node) {
  if (node_priority_.count(node->name()) == 0) {
    VLOG(3) << "Priority of node " << node->name() << " not found.";
    node_priority_[node->name()] = 0;
  }
  HeapReadyManager::AddNode(node);
}
Status PriorityReadyManager::SetPriority(
    const std::unordered_map<string, int>& node_priority) {
  node_priority_ = node_priority;
  return absl::OkStatus();
}
CompositeNodeManager::CompositeNodeManager()
    : ReadyNodeManager(), send_manager_(), recv_manager_() {}
Status CompositeNodeManager::Init(
    const std::unordered_map<const NodeDef*, NodeState>* node_map) {
  node_map_ = node_map;
  TF_RETURN_IF_ERROR(send_manager_.Init(node_map));
  TF_RETURN_IF_ERROR(recv_manager_.Init(node_map));
  curr_node_ = nullptr;
  return absl::OkStatus();
}
void CompositeNodeManager::AddNode(const NodeDef* node) {
  if (IsSend(*node)) {
    send_manager_.AddNode(node);
  } else if (IsRecv(*node)) {
    recv_manager_.AddNode(node);
  } else {
    const auto& device = node_map_->at(node).device_name;
    ops_lifo_map_[device].AddNode(node);
  }
}
const NodeDef* CompositeNodeManager::GetCurrNode() {
  if (curr_node_) return curr_node_;
  std::vector<std::pair<const NodeDef*, Costs::Duration>> candidates;
  for (auto& ops_lifo : ops_lifo_map_) {
    if (!ops_lifo.second.Empty()) {
      const auto* op = ops_lifo.second.GetCurrNode();
      candidates.emplace_back(op, node_map_->at(op).time_ready);
    }
  }
  if (!send_manager_.Empty()) {
    const auto* send = send_manager_.GetCurrNode();
    candidates.emplace_back(send, node_map_->at(send).time_ready);
  }
  if (!recv_manager_.Empty()) {
    const auto* recv = recv_manager_.GetCurrNode();
    candidates.emplace_back(recv, node_map_->at(recv).time_ready);
  }
  CHECK(!candidates.empty());
  auto first_ready = std::min_element(
      candidates.begin(), candidates.end(),
      [](const std::pair<const NodeDef*, Costs::Duration>& a,
         const std::pair<const NodeDef*, Costs::Duration>& b) {
        if (a.second == b.second) {
          int a_score = 2 * IsSend(*a.first) + IsRecv(*a.first);
          int b_score = 2 * IsSend(*b.first) + IsRecv(*b.first);
          if (a_score == b_score) {
            return a.first->name().compare(b.first->name()) < 0;
          } else {
            return a_score > b_score;
          }
        } else {
          return a.second < b.second;
        }
      });
  curr_node_ = first_ready->first;
  return curr_node_;
}
void CompositeNodeManager::RemoveCurrNode() {
  const auto* node = GetCurrNode();
  if (IsSend(*node)) {
    send_manager_.RemoveCurrNode();
  } else if (IsRecv(*node)) {
    recv_manager_.RemoveCurrNode();
  } else {
    const auto device = node_map_->at(node).device_name;
    ops_lifo_map_[device].RemoveCurrNode();
  }
  curr_node_ = nullptr;
}
bool CompositeNodeManager::Empty() const {
  bool empty = true;
  for (const auto& ops_lifo : ops_lifo_map_) {
    empty &= ops_lifo.second.Empty();
  }
  return empty && send_manager_.Empty() && recv_manager_.Empty();
}
std::unique_ptr<ReadyNodeManager> ReadyNodeManagerFactory(
    const string& ready_node_manager) {
  if (ready_node_manager == "FIFO") {
    return std::make_unique<FIFOManager>();
  } else if (ready_node_manager == "LIFO") {
    return std::make_unique<LIFOManager>();
  } else if (ready_node_manager == "FirstReady") {
    return std::make_unique<FirstReadyManager>();
  } else if (ready_node_manager == "Composite") {
    return std::make_unique<CompositeNodeManager>();
  }
  LOG(FATAL) << "Not a valid ready node manager: " << ready_node_manager;
  return nullptr;
}
SchedulerState::~SchedulerState() {}
SchedulerState::SchedulerState(const bool use_static_shapes,
                               const bool use_aggressive_shape_inference,
                               Cluster* cluster,
                               std::unique_ptr<VirtualPlacer> placer)
    : graph_costs_(Costs::ZeroCosts()),
      cluster_(cluster),
      use_static_shapes_(use_static_shapes),
      use_aggressive_shape_inference_(use_aggressive_shape_inference),
      placer_(std::move(placer)) {
  DCHECK(placer_);  
  graph_costs_.num_ops_total = 0;
  initialized_ = false;
  track_mem_usage_snapshot_ = VLOG_IS_ON(1);
}
Status SchedulerState::Init(const GrapplerItem* item,
                            std::vector<const NodeDef*>* initial_nodes,
                            bool create_explicit_channel_device) {
  initialized_ = false;
  node_map_.clear();
  device_.clear();
  additional_nodes_.clear();
  graph_costs_ = Costs::ZeroCosts();
  graph_costs_.num_ops_total = 0;
  op_to_cost_.clear();
  op_counts_.clear();
  op_costs_.clear();
  initial_nodes->clear();
  graph_properties_ = std::make_unique<GraphProperties>(*item);
  if (use_static_shapes_) {
    TF_RETURN_IF_ERROR(graph_properties_->InferStatically(
        true, use_aggressive_shape_inference_, true));
  } else {
    TF_RETURN_IF_ERROR(graph_properties_->InferDynamically(cluster_));
  }
  grappler_item_ = item;
  const auto& graph = grappler_item_->graph;
  const auto& fetch_nodes = grappler_item_->fetch;
  std::set<string> feed_nodes;
  for (const auto& f : grappler_item_->feed) {
    auto iter_and_inserted_flag = feed_nodes.insert(f.first);
    QCHECK(iter_and_inserted_flag.second)
        << "Duplicate feed node found: " << f.first;
  }
  std::unordered_map<string, const NodeDef*> name_to_node;
  std::vector<const NodeDef*> fetch_fanin_nodes;
  TF_RETURN_IF_ERROR(ComputeTransitiveFanin(graph, fetch_nodes, &name_to_node,
                                            &fetch_fanin_nodes));
  std::unordered_map<string, const NodeDef*> name_to_send;
  for (const auto& node : graph.node()) {
    if (IsSend(node)) {
      const auto& attr = node.attr();
      name_to_send[attr.at("tensor_name").s()] = &node;
    }
  }
  std::unordered_map<RecvNodeDescriptor, const NodeDef*, RecvNodeDescriptorHash,
                     RecvNodeDescriptorEqual>
      cached_recv_nodes;
  for (const auto* curr_node : fetch_fanin_nodes) {
    auto& curr_node_state = GetNodeStateOrCreateIt(curr_node);
    const string curr_node_device = DeviceName(curr_node);
    std::vector<string> inputs;
    if (IsRecv(*curr_node)) {
      const auto& attr = curr_node->attr();
      if (attr.count("tensor_name")) {
        const auto& send_node_name = attr.at("tensor_name").s();
        auto it = name_to_send.find(send_node_name);
        if (it != name_to_send.end()) {
          const NodeDef* send = it->second;
          inputs = {send->name()};
        }
      }
    } else {
      for (const string& input : curr_node->input()) {
        inputs.push_back(input);
      }
    }
    for (const string& input_node_name : inputs) {
      const string node_name = NodeName(input_node_name);
      const NodeDef* input_node = name_to_node[node_name];
      if (input_node == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unknown node: ", node_name));
      }
      const string in_device = DeviceName(input_node);
      const auto input_node_port_num = NodePosition(input_node_name);
      if (curr_node_device == in_device || IsControlInput(input_node_name)) {
        curr_node_state.inputs.push_back(
            std::make_pair(input_node, input_node_port_num));
        auto& input_node_state = GetNodeStateOrCreateIt(input_node);
        input_node_state.outputs[input_node_port_num].push_back(curr_node);
      } else {
        RecvNodeDescriptor recv_node(input_node, input_node_port_num,
                                     curr_node_device);
        auto it = cached_recv_nodes.find(recv_node);
        if (it != cached_recv_nodes.end()) {
          const NodeDef* recv_op = it->second;
          curr_node_state.inputs.push_back(std::make_pair(recv_op, 0));
          auto& input_node_state = node_map_.at(recv_op);
          input_node_state.outputs[0].push_back(curr_node);
        } else {
          auto send_and_recv =
              CreateSendRecv(input_node, curr_node, input_node, input_node_name,
                             create_explicit_channel_device);
          const auto* send = send_and_recv.first;
          const auto* recv = send_and_recv.second;
          curr_node_state.inputs.push_back(std::make_pair(recv, 0));
          auto& input_node_state = GetNodeStateOrCreateIt(input_node);
          input_node_state.outputs[input_node_port_num].push_back(send);
          cached_recv_nodes[recv_node] = recv;
        }
      }
    }
    const bool given_as_feed =
        feed_nodes.find(curr_node->name()) != feed_nodes.end();
    const bool has_no_inputs = inputs.empty();
    if (given_as_feed || has_no_inputs) {
      curr_node_state.time_ready = Costs::Duration();
      initial_nodes->push_back(curr_node);
      VLOG(3) << "Added ready node: " << curr_node->name();
    }
    feed_nodes.erase(curr_node->name());
    if (IsPersistent(*curr_node)) {
      auto& device_state = device_[curr_node_device];
      for (int port_num = 0,
               port_num_end = curr_node_state.output_properties.size();
           port_num < port_num_end; ++port_num) {
        device_state.persistent_nodes.insert(
            std::make_pair(curr_node, port_num));
      }
    }
  }
  if (initial_nodes->empty()) {
    return errors::InvalidArgument("No ready nodes in the graph.");
  }
  if (!feed_nodes.empty()) {
    VLOG(1) << "Some feed nodes were not consumed by the fetch fanin: "
            << absl::StrJoin(feed_nodes, ",");
  }
  initialized_ = true;
  return absl::OkStatus();
}
void SchedulerState::MaybeUpdateInputOutput(const NodeDef* node) {
  CHECK(!initialized_) << "MaybeUpdateInputOutput is called after Init().";
  if ((IsSend(*node) || IsRecv(*node)) && node->attr().count(kAttrInputSrc)) {
    auto& node_state = node_map_[node];
    auto& inputs = node_state.input_properties;
    auto& outputs = node_state.output_properties;
    CHECK(inputs.empty());
    CHECK(outputs.empty());
    const auto& attr = node->attr();
    const auto& input_source_name = attr.at(kAttrInputSrc).s();
    if (IsControlInput(input_source_name)) {
      OpInfo::TensorProperties control_message;
      control_message.set_dtype(DT_FLOAT);
      control_message.mutable_shape()->add_dim()->set_size(1);
      auto* value = control_message.mutable_value();
      value->add_float_val(1);
      inputs.push_back(control_message);
      outputs.push_back(control_message);
    } else {
      const auto& output_properties =
          graph_properties_->GetOutputProperties(NodeName(input_source_name));
      if (!output_properties.empty()) {
        const auto input_node_port_num = NodePosition(input_source_name);
        CHECK_GT(output_properties.size(), input_node_port_num);
        inputs.push_back(output_properties[input_node_port_num]);
        outputs.push_back(output_properties[input_node_port_num]);
      }
    }
  }
}
string SchedulerState::DeviceName(const NodeDef* node) const {
  return placer_->get_canonical_device_name(*node);
}
string SchedulerState::SanitizedDeviceName(const NodeDef* node) const {
  return absl::StrReplaceAll(placer_->get_canonical_device_name(*node),
                             {{":", "_"}});
}
string SchedulerState::ChannelDeviceName(const NodeDef* from,
                                         const NodeDef* to) const {
  CHECK(!initialized_) << "ChannelDeviceName is called after Init().";
  return absl::StrCat(kChannelDevice, "_from_", SanitizedDeviceName(from),
                      "_to_", SanitizedDeviceName(to));
}
std::pair<const NodeDef*, const NodeDef*> SchedulerState::CreateSendRecv(
    const NodeDef* from, const NodeDef* to, const NodeDef* input_node,
    const string& input_name, bool create_channel_device) {
  CHECK(!initialized_) << "CreateSendRecv is called after Init().";
  auto input_node_port_num = NodePosition(input_name);
  string src_name;
  bool control_input = false;
  if (input_node_port_num >= 0) {
    src_name = absl::StrCat(from->name(), "_", input_node_port_num);
  } else {
    src_name = absl::StrCat(from->name(), "_minus1");
    control_input = true;
  }
  auto* send = new NodeDef();
  send->set_name("Send_" + src_name + "_from_" + SanitizedDeviceName(from) +
                 "_to_" + SanitizedDeviceName(to));
  send->set_op("_Send");
  send->add_input(from->name());
  auto send_device =
      create_channel_device ? ChannelDeviceName(from, to) : DeviceName(from);
  send->set_device(send_device);
  auto& send_attr = *(send->mutable_attr());
  send_attr[kAttrInputSrc].set_s(input_name);
  send_attr[kAttrSrcDevice].set_s(DeviceName(from));
  send_attr[kAttrDstDevice].set_s(DeviceName(to));
  if (input_node->attr().count(kAttrTensorName)) {
    send_attr[kAttrTensorName].set_s(
        input_node->attr().at(kAttrTensorName).s());
  }
  auto* recv = new NodeDef();
  recv->set_name("Recv_" + src_name + "_on_" + SanitizedDeviceName(to));
  recv->set_op("_Recv");
  recv->add_input(send->name());
  recv->set_device(DeviceName(to));
  auto& recv_attr = *(recv->mutable_attr());
  recv_attr[kAttrInputSrc].set_s(input_name);
  if (input_node->attr().count(kAttrTensorName)) {
    recv_attr[kAttrTensorName].set_s(
        input_node->attr().at(kAttrTensorName).s());
  }
  if (from->attr().contains(kStreaming) && !control_input) {
    if (input_node_port_num >= from->attr().at(kStreaming).list().b_size()) {
      LOG(ERROR)
          << from->name()
          << " port index larger than length of _streaming attribute list.";
    } else if (from->attr().at(kStreaming).list().b(input_node_port_num)) {
      send_attr[kStreaming].mutable_list()->add_b(true);
      recv_attr[kStreaming].mutable_list()->add_b(true);
    }
  }
  auto& send_node_state = GetNodeStateOrCreateIt(send);
  send_node_state.device_name = send->device();  
  send_node_state.inputs.push_back(std::make_pair(from, input_node_port_num));
  send_node_state.outputs[0].push_back(recv);
  auto& recv_node_state = GetNodeStateOrCreateIt(recv);
  recv_node_state.inputs.push_back(std::make_pair(send, 0));
  recv_node_state.outputs[0].push_back(to);
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(send));
  additional_nodes_.emplace_back(std::unique_ptr<NodeDef>(recv));
  return std::make_pair(send, recv);
}
OpContext SchedulerState::CreateOpContext(const NodeDef* node) const {
  DeviceProperties device;
  device = placer_->get_device(*node);
  if (IsSend(*node)) {
    device.set_type(kChannelDevice);
  }
  OpContext op_context;
  const auto& node_state = node_map_.at(node);
  op_context.name = node->name();
  op_context.device_name = node_state.device_name;
  auto& op_info = op_context.op_info;
  op_info.set_op(node->op());
  *op_info.mutable_attr() = node->attr();
  for (auto& input : node_state.input_properties) {
    *op_info.add_inputs() = input;
  }
  for (auto& output : node_state.output_properties) {
    *op_info.add_outputs() = output;
  }
  op_info.mutable_device()->Swap(&device);
  if (grappler_item_->graph.has_library()) {
    op_context.function_library = &grappler_item_->graph.library();
  }
  return op_context;
}
NodeState& SchedulerState::GetNodeStateOrCreateIt(const NodeDef* node) {
  CHECK(!initialized_) << "GetNodeStateOrCreateIt is called after Init().";
  auto it = node_map_.find(node);
  if (it != node_map_.end()) {
    return it->second;
  }
  it = node_map_.emplace(node, NodeState()).first;
  auto& node_state = it->second;
  node_state.input_properties =
      graph_properties_->GetInputProperties(node->name());
  node_state.output_properties =
      graph_properties_->GetOutputProperties(node->name());
  node_state.shape_incompatible =
      graph_properties_->CheckShapeIncompatible(node->name());
  MaybeUpdateInputOutput(node);
  if (!IsSend(*node)) {
    node_state.device_name = DeviceName(node);
  }
  for (size_t i = 0; i < node_state.output_properties.size(); ++i) {
    node_state.time_no_references[i] = Costs::Duration::max();
    node_state.num_outputs_executed[i] = 0;
    node_state.outputs[i] = {};
  }
  node_state.time_no_references[-1] = Costs::Duration::max();
  node_state.num_outputs_executed[-1] = 0;
  node_state.outputs[-1] = {};
  node_state.time_scheduled = Costs::Duration().infinity();
  return it->second;
}
void SchedulerState::GetOutputNodes(const NodeDef* node,
                                    const Costs::Duration& curr_time,
                                    std::vector<const NodeDef*>* output_nodes) {
  int slot = -1;
  if (IsSwitch(*node) && node->attr().count(kOutputSlots) > 0 &&
      node->attr().at(kOutputSlots).list().i_size() > 0) {
    slot = node->attr().at(kOutputSlots).list().i(0);
    for (int i = 1; i < node->attr().at(kOutputSlots).list().i_size(); ++i) {
      if (slot != node->attr().at(kOutputSlots).list().i(i)) {
        slot = -1;
        break;
      }
    }
  }
  auto& node_state = node_map_[node];
  for (const auto& port_num_output_pair : node_state.outputs) {
    if (slot >= 0 && port_num_output_pair.first != slot) continue;
    for (auto* output_node : port_num_output_pair.second) {
      auto& output_state = node_map_[output_node];
      output_state.num_inputs_ready++;
      int output_state_inputs_size = output_state.inputs.size();
      if (output_state.num_inputs_ready == output_state_inputs_size ||
          IsMerge(*output_node)) {
        output_state.time_ready = curr_time;
        output_nodes->push_back(output_node);
        VLOG(3) << "  Add output: " << output_node->name();
      }
    }
  }
}
std::vector<const NodeDef*> SchedulerState::MarkNodeExecuted(
    const NodeDef* node, const Costs& node_costs, const OpContext& op_context,
    bool extract_execution_count_attr,
    const std::string& override_device_name) {
  auto& node_state = node_map_[node];
  bool previously_executed_merge =
      IsMerge(*node) && (node_state.time_finished != Costs::Duration::max());
  node_state.execution_count = 1;
  if (extract_execution_count_attr && node->attr().count(kExecutionCount) > 0) {
    node_state.execution_count = node->attr().at(kExecutionCount).i();
  }
  node_state.node_costs = node_costs;
  Costs total_node_costs = node_state.TotalNodeCosts();
  graph_costs_ = CombineCosts(graph_costs_, total_node_costs);
  const string& op_name = node->op();
  auto& op_cost = FindOrCreateZero(op_name, &op_to_cost_);
  op_cost = CombineCosts(op_cost, total_node_costs);
  if (VLOG_IS_ON(2)) {
    string node_description = GetOpDescription(op_context.op_info);
    op_counts_[node_description] += 1;
    op_costs_[node_description] =
        std::make_pair(total_node_costs.execution_time.asMicroSeconds().count(),
                       !node_costs.inaccurate);
  }
  std::string device_name = node_state.device_name;
  if (!override_device_name.empty()) {
    device_name = override_device_name;
  }
  auto& device = device_[device_name];
  device.nodes_executed.push_back(node);
  if (node_state.time_scheduled == Costs::Duration().infinity()) {
    node_state.time_scheduled =
        std::max(device.GetCurrTime(), node_state.time_ready);
    device.device_costs.execution_time = node_state.time_scheduled;
  }
  device.device_costs = CombineCosts(device.device_costs, total_node_costs);
  auto curr_time = device.GetCurrTime();
  node_state.time_finished = curr_time;
  UpdateDeviceAnnotationState(node, node_state, &device);
  if (!IsPersistent(*node)) {
    for (const auto& port_num_output_pair : node_state.outputs) {
      int port_num = port_num_output_pair.first;
      if (node_state.outputs[port_num].empty()) {
        node_state.time_no_references[port_num] = curr_time;
      } else {
        if (node_state.node_costs.persistent_output_ports.contains(port_num)) {
          continue;
        }
        if (!IsStreamingPort(*node, port_num)) {
          device.memory_usage += GetOrCalculateOutputSize(node_state, port_num);
        }
        device.nodes_in_memory.insert(std::make_pair(node, port_num));
      }
    }
  }
  for (const auto& port : node_costs.persistent_output_ports) {
    device.persistent_nodes.insert({node, port});
  }
  auto& device_op_cost = FindOrCreateZero(op_name, &device.op_to_cost);
  device_op_cost = CombineCosts(device_op_cost, total_node_costs);
  VLOG(3) << "Op scheduled -- name: " << node->name() << ", op: " << node->op()
          << ", device: " << node->device()
          << ", execution_count: " << node_state.execution_count
          << ", ready: " << node_state.time_ready.count()
          << ", scheduled: " << node_state.time_scheduled.count()
          << ", finished: " << node_state.time_finished.count();
  VLOG(5) << "  Current device memory usage (before deallocation): "
          << device.memory_usage;
  std::vector<const NodeDef*> new_nodes;
  if (previously_executed_merge) {
    VLOG(1) << "node [ " << node->name() << ", " << node->op() << " ] "
            << "is executed more than once. "
            << "Skip scheduling its output nodes.";
  } else {
    GetOutputNodes(node, curr_time, &new_nodes);
  }
  if (!IsPersistent(*node)) {
    if (device.memory_usage > device.max_memory_usage) {
      device.max_memory_usage = device.memory_usage;
      if (track_mem_usage_snapshot_) {
        device.mem_usage_snapshot_at_peak = device.nodes_in_memory;
      }
    }
  }
  if (track_mem_usage_snapshot_) {
    device.temporary_memory_usage_trace.push_back(
        {node->name(), device.memory_usage});
  }
  for (const auto& input_port : node_state.inputs) {
    auto* input = input_port.first;
    auto port = input_port.second;
    auto& input_state = node_map_[input];
    input_state.num_outputs_executed[port]++;
    int input_state_outputs_size_ = input_state.outputs[port].size();
    if (input_state.node_costs.persistent_output_ports.contains(port)) continue;
    if (input_state.num_outputs_executed[port] == input_state_outputs_size_ &&
        !IsPersistent(*input)) {
      input_state.time_no_references[port] = curr_time;
      auto& input_device = device_[input_state.device_name];
      if (!IsStreamingPort(*input, port)) {
        input_device.memory_usage -=
            GetOrCalculateOutputSize(input_state, port);
      }
      input_device.nodes_in_memory.erase(std::make_pair(input, port));
    }
  }
  return new_nodes;
}
Costs SchedulerState::Summary() const {
  VLOG(1) << graph_costs_.num_ops_total << " ops processed in total, with "
          << graph_costs_.num_ops_with_unknown_shapes
          << " having unknown shapes";
  VLOG(1) << "Expected execution time: " << graph_costs_.execution_time.count();
  VLOG(1) << "Expected compute time: " << graph_costs_.compute_time.count();
  VLOG(1) << "Expected memory time: " << graph_costs_.memory_time.count();
  VLOG(1) << "Expected intermediate memory time: "
          << graph_costs_.intermediate_memory_time.count();
  VLOG(1) << "Expected max memory: " << graph_costs_.max_memory;
  VLOG(1) << "Expected max per-op buffers: " << graph_costs_.max_per_op_buffers;
  VLOG(1) << "Expected max per-op streaming buffers: "
          << graph_costs_.max_per_op_streaming;
  VLOG(1) << "Per-op execution time / compute time / memory time"
          << " / intermediate memory time:";
  for (const auto& op_cost_pair : op_to_cost_) {
    const auto& op = op_cost_pair.first;
    const auto& cost = op_cost_pair.second.execution_time.count();
    const auto& compute_cost = op_cost_pair.second.compute_time.count();
    const auto& memory_cost = op_cost_pair.second.memory_time.count();
    const auto& intermediate_memory_cost =
        op_cost_pair.second.intermediate_memory_time.count();
    const bool is_op_cost_accurate = !op_cost_pair.second.inaccurate;
    if (cost) {  
      VLOG(1) << absl::StrFormat(" + %30s : %c %10d / %10d / %10d / %10d", op,
                                 (is_op_cost_accurate ? ' ' : '~'), cost,
                                 compute_cost, memory_cost,
                                 intermediate_memory_cost);
    }
  }
  VLOG(1) << "Devices:";
  Costs critical_path_costs = Costs::ZeroCosts();
  std::vector<string> device_names;
  device_names.reserve(device_.size());
  for (auto& it : device_) {
    device_names.push_back(it.first);
  }
  std::sort(device_names.begin(), device_names.end());
  for (const auto& name : device_names) {
    const auto& state = device_.at(name);
    std::map<string, int64_t> op_to_memory;
    int64_t persistent_memory_usage = 0;
    std::set<string> persistent_ops;
    for (const auto& node_port : state.persistent_nodes) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      int64_t output_size = 0;
      auto it = node_map_.find(node);
      if (it != node_map_.end()) {
        output_size = GetOrCalculateOutputSize(it->second, port);
      }
      persistent_memory_usage += output_size;
      op_to_memory[node->op()] += output_size;
      persistent_ops.insert(node->op());
    }
    int64_t max_memory_usage = persistent_memory_usage + state.max_memory_usage;
    critical_path_costs.estimated_max_memory_per_device[name] =
        max_memory_usage;
    const Costs::NanoSeconds wall_time_ns = state.GetCurrTime();
    VLOG(1) << "Device = " << name
            << ", num_nodes = " << state.nodes_executed.size()
            << ", wall_time_ns = " << wall_time_ns.count() << ", memory usage: "
            << "persistent = " << HumanReadableNumBytes(persistent_memory_usage)
            << ", peak = " << HumanReadableNumBytes(state.max_memory_usage)
            << ", total = " << HumanReadableNumBytes(max_memory_usage)
            << ", at the end: " << HumanReadableNumBytes(state.memory_usage);
    VLOG(1) << state.device_costs.num_ops_total
            << " ops processed in total, with "
            << state.device_costs.num_ops_with_unknown_shapes
            << " having unknown shapes";
    const auto& device_annotation_stats = state.shape_annotation_stats;
    if (device_annotation_stats.num_ops_annotated > 0) {
      VLOG(1) << device_annotation_stats.num_ops_annotated
              << " ops with shape annotation, with "
              << device_annotation_stats.num_ops_executed_more_than_once
              << " executed more than once, "
              << device_annotation_stats.num_ops_with_dynamic_shapes
              << " with dynamic shapes, "
              << device_annotation_stats.num_ops_with_incompatible_shapes
              << " with incompatible shapes, "
              << device_annotation_stats.num_ops_executed
              << " ops executed in total.";
    }
    VLOG(1) << "Per-op execution time / compute time / memory time "
            << " / intermediate memory time"
            << " (and memory usage at peak memory usage):";
    for (const auto& node_port : state.mem_usage_snapshot_at_peak) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      auto it = node_map_.find(node);
      if (it != node_map_.end()) {
        op_to_memory[node->op()] += GetOrCalculateOutputSize(it->second, port);
      }
    }
    Costs::NanoSeconds total_compute_time_ns;
    bool is_total_cost_accurate = true;
    for (const auto& op_cost_pair : state.op_to_cost) {
      const auto& op = op_cost_pair.first;
      const auto& cost = op_cost_pair.second.execution_time.count();
      const auto& compute_cost = op_cost_pair.second.compute_time.count();
      const auto& memory_cost = op_cost_pair.second.memory_time.count();
      const auto& intermediate_memory_cost =
          op_cost_pair.second.intermediate_memory_time.count();
      total_compute_time_ns += op_cost_pair.second.execution_time;
      const bool is_op_cost_accurate = !op_cost_pair.second.inaccurate;
      if (!is_op_cost_accurate) {
        is_total_cost_accurate = false;
      }
      int64_t op_mem_usage = 0;
      auto it = op_to_memory.find(op);
      if (it != op_to_memory.end()) {
        op_mem_usage = it->second;
      }
      const float mem_usage_percent =
          max_memory_usage > 0 ? Round2(100.0 * op_mem_usage / max_memory_usage)
                               : 0.0;
      if (cost || mem_usage_percent > 1.0) {
        VLOG(1) << absl::StrFormat(
                       " + %30s : %c %10d / %10d / %10d / %10d", op.c_str(),
                       (is_op_cost_accurate ? ' ' : '~'), cost, compute_cost,
                       memory_cost, intermediate_memory_cost)
                << " (" << HumanReadableNumBytes(op_mem_usage) << " ["
                << mem_usage_percent << "%] "
                << (persistent_ops.count(op) > 0 ? ": persistent op)" : ")");
      }
    }
    int utilization = 0;
    if (wall_time_ns.count() > 0) {
      utilization = total_compute_time_ns.count() * 100 / wall_time_ns.count();
    }
    VLOG(1) << "Device = " << name << ", total_compute_time_ns = "
            << (is_total_cost_accurate ? "" : "~")
            << total_compute_time_ns.count()
            << ", utilization = " << utilization << "%";
    if (critical_path_costs.execution_time <= state.GetCurrTime()) {
      critical_path_costs = state.device_costs;
      critical_path_costs.persistent_memory = persistent_memory_usage;
      critical_path_costs.temporary_memory = state.max_memory_usage;
      critical_path_costs.max_memory = max_memory_usage;
    }
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Node description, counts, cost:";
    for (const auto& item : op_counts_) {
      int cost;
      bool is_cost_accurate;
      std::tie(cost, is_cost_accurate) = op_costs_.at(item.first);
      VLOG(2) << "Node: " << item.first << ", Count: " << item.second
              << ", Individual Cost: " << (is_cost_accurate ? "" : "~") << cost
              << " us";
    }
  }
  VLOG(1) << "Critical path execution time: "
          << critical_path_costs.execution_time.count();
  return critical_path_costs;
}
Costs SchedulerState::Summary(RunMetadata* metadata) {
  if (metadata) GenerateRunMetadata(metadata);
  return Summary();
}
void SchedulerState::GenerateRunMetadata(RunMetadata* metadata) {
  StepStats* stepstats = metadata->mutable_step_stats();
  for (const auto& device : device_) {
    GraphDef* device_partition_graph = metadata->add_partition_graphs();
    DeviceStepStats* device_stepstats = stepstats->add_dev_stats();
    device_stepstats->set_device(device.first);
    for (const auto& node_def : device.second.nodes_executed) {
      if (node_map_.find(node_def) == node_map_.end()) {
        continue;
      }
      const NodeState& nodestate = node_map_.at(node_def);
      NodeExecStats* node_stats = device_stepstats->add_node_stats();
      uint64 total_output_size = 0;
      uint64_t persistent_output_size = 0;
      for (int slot = 0, slot_end = nodestate.output_properties.size();
           slot < slot_end; slot++) {
        const auto& properties = nodestate.output_properties[slot];
        NodeOutput* no = node_stats->add_output();
        no->set_slot(slot);
        TensorDescription* tensor_descr = no->mutable_tensor_description();
        tensor_descr->set_dtype(properties.dtype());
        *tensor_descr->mutable_shape() = properties.shape();
        const int64_t tensor_size_requested =
            CalculateOutputSize(nodestate.output_properties, slot);
        const int64_t tensor_size_allocated =
            GetOrCalculateOutputSize(nodestate, slot);
        total_output_size += tensor_size_allocated;
        if (nodestate.node_costs.persistent_output_ports.contains(slot)) {
          persistent_output_size += tensor_size_allocated;
        }
        tensor_descr->mutable_allocation_description()->set_requested_bytes(
            tensor_size_requested);
        tensor_descr->mutable_allocation_description()->set_allocated_bytes(
            tensor_size_allocated);
      }
      if (node_def->op() != "HloGenericOp") {
        node_stats->set_timeline_label(node_def->op());
      } else {
        string timeline_label;
        if (node_def->attr().count("hlo_opcode") > 0) {
          absl::StrAppend(&timeline_label,
                          node_def->attr().at("hlo_opcode").s());
        }
        if (node_def->attr().count("_hlo_metadata_op_type") > 0) {
          absl::StrAppend(&timeline_label, "/",
                          node_def->attr().at("_hlo_metadata_op_type").s());
        }
        node_stats->set_timeline_label(timeline_label);
      }
      node_stats->set_node_name(node_def->name());
      node_stats->set_op_start_rel_micros(0);
      node_stats->set_all_start_micros(
          nodestate.time_scheduled.asMicroSeconds().count());
      node_stats->set_op_end_rel_micros(
          nodestate.time_finished.asMicroSeconds().count() -
          nodestate.time_scheduled.asMicroSeconds().count());
      node_stats->set_all_end_rel_micros(
          nodestate.time_finished.asMicroSeconds().count() -
          nodestate.time_scheduled.asMicroSeconds().count());
      node_stats->set_op_start_rel_nanos(0);
      node_stats->set_all_start_nanos(nodestate.time_scheduled.count());
      node_stats->set_op_end_rel_nanos(nodestate.time_finished.count() -
                                       nodestate.time_scheduled.count());
      node_stats->set_all_end_rel_nanos(nodestate.time_finished.count() -
                                        nodestate.time_scheduled.count());
      auto* mem_stats = node_stats->mutable_memory_stats();
      mem_stats->set_temp_memory_size(0);
      int64_t persistent_memory_size = 0;
      if (IsPersistent(*node_def)) {
        persistent_memory_size = total_output_size;
      } else {
        persistent_memory_size = persistent_output_size;
      }
      mem_stats->set_persistent_memory_size(persistent_memory_size);
      *device_partition_graph->add_node() = *node_def;
    }
  }
}
const std::unordered_map<string, int64_t> SchedulerState::GetPeakMemoryUsage()
    const {
  std::unordered_map<string, int64_t> result;
  for (const auto& device : device_) {
    const string& name = device.first;
    const DeviceState& state = device.second;
    result[name] = state.max_memory_usage;
  }
  return result;
}
const std::unordered_map<string, int64_t>
SchedulerState::GetPersistentMemoryUsage() const {
  std::unordered_map<string, int64_t> result;
  for (const auto& device : device_) {
    const string& name = device.first;
    const DeviceState& state = device.second;
    int64_t persistent_memory_usage = 0;
    for (const auto& node_port : state.persistent_nodes) {
      const auto* node = node_port.first;
      const auto port = node_port.second;
      const auto& node_state = node_map_.at(node);
      persistent_memory_usage += GetOrCalculateOutputSize(node_state, port);
    }
    result[name] = persistent_memory_usage;
  }
  return result;
}
void SchedulerState::SetNodeStateTimeScheduled(const NodeDef* node) {
  auto& node_state = node_map_.at(node);
  auto& device = device_[node_state.device_name];
  node_state.time_scheduled = device.GetCurrTime();
}
int64_t SchedulerState::GetOrCalculateOutputSize(const NodeState& node_state,
                                                 int port_num) const {
  auto& node_costs = node_state.node_costs;
  auto it = node_costs.output_tensor_size_bytes.find(port_num);
  if (it != node_costs.output_tensor_size_bytes.end()) {
    return it->second;
  }
  return CalculateOutputSize(node_state.output_properties, port_num);
}
VirtualScheduler::~VirtualScheduler() {}
VirtualScheduler::VirtualScheduler(const bool use_static_shapes,
                                   const bool use_aggressive_shape_inference,
                                   Cluster* cluster,
                                   ReadyNodeManager* ready_nodes,
                                   std::unique_ptr<VirtualPlacer> placer)
    : scheduler_state_(std::make_unique<SchedulerState>(
          use_static_shapes, use_aggressive_shape_inference, cluster,
          std::move(placer))),
      ready_nodes_(ready_nodes) {}
VirtualScheduler::VirtualScheduler(
    ReadyNodeManager* ready_nodes,
    std::unique_ptr<SchedulerState> scheduler_state)
    : scheduler_state_(std::move(scheduler_state)), ready_nodes_(ready_nodes) {}
Status VirtualScheduler::Init(const GrapplerItem* item) {
  TF_RETURN_IF_ERROR(ready_nodes_->Init(GetNodeStates()));
  std::vector<const NodeDef*> initial_nodes;
  auto status = scheduler_state_->Init(item, &initial_nodes);
  if (status.ok()) {
    for (auto node : initial_nodes) {
      ready_nodes_->AddNode(node);
    }
  }
  return status;
}
OpContext VirtualScheduler::GetCurrNode() {
  const NodeDef* node = ready_nodes_->GetCurrNode();
  return scheduler_state_->CreateOpContext(node);
}
bool VirtualScheduler::MarkCurrNodeExecuted(const Costs& node_costs) {
  const NodeDef* node = ready_nodes_->GetCurrNode();
  auto new_nodes = scheduler_state_->MarkNodeExecuted(
      node, node_costs,
      scheduler_state_->CreateOpContext(ready_nodes_->GetCurrNode()));
  for (auto node : new_nodes) {
    ready_nodes_->AddNode(node);
  }
  ready_nodes_->RemoveCurrNode();
  return !ready_nodes_->Empty();
}
}  
}  