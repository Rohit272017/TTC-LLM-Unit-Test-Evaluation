#include "tensorflow/compiler/tf2tensorrt/segment/segment.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <utility>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"
#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
namespace segment {
namespace {
using absl::StrAppend;
using absl::StrAppendFormat;
using absl::StrCat;
using absl::StrJoin;
class SimpleNode;
class SimpleGraph;
class SimpleEdge {
 public:
  SimpleEdge(int id, SimpleNode* src, int src_port, SimpleNode* dst,
             int dst_port, bool is_control = false)
      : id_(id),
        src_(src),
        src_port_(src_port),
        dst_(dst),
        dst_port_(dst_port),
        control_(is_control) {}
  ~SimpleEdge() {}
  SimpleNode* src() const { return src_; }
  SimpleNode* dst() const { return dst_; }
  int src_output() const { return src_port_; }
  int dst_input() const { return dst_port_; }
  int id() const { return id_; }
  bool IsControlEdge() const { return control_; }
 private:
  int id_;
  SimpleNode* src_;
  int src_port_;
  SimpleNode* dst_;
  int dst_port_;
  bool control_;
};
class SimpleNode {
 public:
  SimpleNode(const Node* node, const int id);
  const std::vector<SimpleEdge*>& in_edges() const { return in_edges_; }
  const std::vector<SimpleEdge*>& out_edges() const { return out_edges_; }
  std::vector<SimpleNode*> in_nodes() const {
    std::vector<SimpleNode*> res;
    res.reserve(in_edges_.size());
    for (const auto e : in_edges_) {
      if (e) res.push_back(e->src());
    }
    return res;
  }
  std::vector<SimpleNode*> out_nodes() const {
    std::vector<SimpleNode*> res;
    res.reserve(out_edges_.size());
    for (const auto e : out_edges_) {
      if (e) res.push_back(e->dst());
    }
    return res;
  }
  const string& name() const { return node_->name(); }
  const Node* tf_node() const { return node_; }
  int id() const { return id_; }
 private:
  const Node* node_;
  std::vector<SimpleEdge*> in_edges_;
  std::vector<SimpleEdge*> out_edges_;
  int id_;
  friend class SimpleGraph;
};
class SimpleGraph {
 public:
  explicit SimpleGraph(const Graph* g);
  ~SimpleGraph();
  void AddControlEdge(SimpleNode* src, SimpleNode* dst);
  void AddEdge(SimpleNode* src, int out_port, SimpleNode* dst, int in_port);
  void RemoveEdge(const SimpleEdge*);
  SimpleNode* FindNodeId(int node_id) {
    if (node_id < 0 || node_id > static_cast<int>(nodes_.size())) {
      return nullptr;
    }
    return nodes_[node_id];
  }
  int num_node_ids() const { return nodes_.size(); }
  const SimpleNode* source_node() const { return nodes_[Graph::kSourceId]; }
  const SimpleNode* sink_node() const { return nodes_[Graph::kSinkId]; }
 private:
  const Graph* g_;
  std::vector<SimpleNode*> nodes_;
  std::vector<SimpleEdge*> edges_;
  std::set<int> free_edge_ids_;
  std::set<int> free_node_ids_;
};
SimpleNode::SimpleNode(const Node* node, const int id) : node_(node), id_(id) {
  if (node_) {
    in_edges_.reserve(node_->in_edges().size());
    out_edges_.reserve(node_->out_edges().size());
  }
}
SimpleGraph::SimpleGraph(const Graph* g) : g_(g) {
  int n_nodes = g_->num_node_ids();
  nodes_.resize(n_nodes, nullptr);
  nodes_[g->kSourceId] = new SimpleNode(g->source_node(), g->kSourceId);
  nodes_[g->kSinkId] = new SimpleNode(g->sink_node(), g->kSinkId);
  int n_edges = g->num_edge_ids();
  edges_.resize(n_edges, nullptr);
  for (int i = 2; i < n_nodes; i++) {
    const auto n = g->FindNodeId(i);
    if (n) {
      nodes_[i] = new SimpleNode(n, i);
    } else {
      free_node_ids_.insert(i);
    }
  }
  for (int i = 0; i < n_edges; i++) {
    const auto e = g->FindEdgeId(i);
    if (e) {
      const auto tfsrc = e->src();
      const auto tfdst = e->dst();
      bool is_control = e->IsControlEdge();
      auto src = nodes_[tfsrc->id()];
      auto dst = nodes_[tfdst->id()];
      auto edge = new SimpleEdge(i, src, e->src_output(), dst, e->dst_input(),
                                 is_control);
      edges_[i] = edge;
      src->out_edges_.push_back(edge);
      dst->in_edges_.push_back(edge);
    } else {
      free_edge_ids_.insert(i);
    }
  }
}
void SimpleGraph::AddEdge(SimpleNode* src, int out_port, SimpleNode* dst,
                          int in_port) {
  int i = edges_.size();
  if (!free_edge_ids_.empty()) {
    auto it = free_edge_ids_.begin();
    i = *it;
    free_edge_ids_.erase(it);
  } else {
    edges_.push_back(nullptr);
  }
  bool is_control = (out_port == Graph::kControlSlot);
  is_control |= (in_port == Graph::kControlSlot);
  auto edge = new SimpleEdge(i, src, out_port, dst, in_port, is_control);
  edges_[i] = edge;
  src->out_edges_.push_back(edge);
  dst->in_edges_.push_back(edge);
}
void SimpleGraph::AddControlEdge(SimpleNode* src, SimpleNode* dst) {
  AddEdge(src, Graph::kControlSlot, dst, Graph::kControlSlot);
}
void SimpleGraph::RemoveEdge(const SimpleEdge* edge) {
  auto src = edge->src();
  auto dst = edge->dst();
  for (auto it = src->out_edges_.begin(); it != src->out_edges_.end(); ++it) {
    if (*it == edge) {
      src->out_edges_.erase(it);
      break;
    }
  }
  for (auto it = dst->in_edges_.begin(); it != dst->in_edges_.end(); ++it) {
    if (*it == edge) {
      dst->in_edges_.erase(it);
      break;
    }
  }
}
SimpleGraph::~SimpleGraph() {
  for (auto x : nodes_) delete x;
  for (auto x : edges_) delete x;
}
struct SimpleEdgePtrCompare {
  bool operator()(const SimpleEdge* lhs, const SimpleEdge* rhs) const {
    return lhs->id() < rhs->id();
  }
};
void StableDFS(const SimpleGraph& g, bool reverse,
               const std::vector<const SimpleNode*>& start,
               const std::function<bool(const SimpleNode*)>& enter,
               const std::function<bool(const SimpleNode*)>& leave) {
  struct Work {
    const SimpleNode* node;
    bool leave;  
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }
  auto get_nodes = [reverse](const SimpleNode* n) {
    return reverse ? n->in_nodes() : n->out_nodes();
  };
  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();
    auto n = w.node;
    if (w.leave) {
      if (leave && !leave(n)) return;
      continue;
    }
    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter && !enter(n)) return;
    if (leave) stack.push_back(Work{n, true});
    auto nodes = get_nodes(n);
    std::vector<const SimpleNode*> nodes_sorted(nodes.begin(), nodes.end());
    std::sort(nodes_sorted.begin(), nodes_sorted.end(),
              [](const SimpleNode* lhs, const SimpleNode* rhs) {
                return lhs->name() < rhs->name();
              });
    for (const SimpleNode* node : nodes_sorted) {
      if (!visited[node->id()]) {
        stack.push_back(Work{node, false});
      }
    }
  }
}
bool CanContractEdge(const SimpleEdge* edge,
                     const std::unique_ptr<SimpleGraph>& graph) {
  const auto src = edge->src();
  const auto dst = edge->dst();
  std::vector<const SimpleNode*> dfs_start_nodes;
  for (const SimpleNode* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }
  bool has_cycle = false;
  StableDFS(*graph, true, dfs_start_nodes, nullptr,
            [&has_cycle, src](const SimpleNode* n) {
              if (n == src) {
                has_cycle = true;
                return false;
              }
              return true;
            });
  return !has_cycle;
}
string TensorPropertiesToString(const OpInfo::TensorProperties& prop) {
  string s = StrCat(DataTypeString(prop.dtype()), ": ");
  StrAppend(&s, "[");
  if (prop.shape().unknown_rank()) {
    StrAppend(&s, "?");
  } else {
    StrAppend(&s, StrJoin(prop.shape().dim(), ",",
                          [](string* out, const TensorShapeProto_Dim& d) {
                            StrAppendFormat(out, "%d", d.size());
                          }));
  }
  StrAppend(&s, "]");
  return s;
}
string TensorPropertiesToString(
    const std::vector<OpInfo::TensorProperties>& properties) {
  return StrJoin(properties, "; ",
                 [](string* out, const OpInfo::TensorProperties& prop) {
                   StrAppend(out, TensorPropertiesToString(prop));
                 });
}
std::optional<const TensorShapeProto*> FindLeadingShape(
    absl::Span<const OpInfo::TensorProperties> properties) {
  DCHECK(!properties.empty());
  const TensorShapeProto* result;
  int max_batch_dim_value;
  auto choose_shape_with_higher_rank = [&](const TensorShapeProto* s) {
    result = s;
    max_batch_dim_value = s->dim_size() < 1 ? 1 : s->dim(0).size();
  };
  DCHECK(!properties[0].shape().unknown_rank());
  choose_shape_with_higher_rank(&properties[0].shape());
  for (const OpInfo::TensorProperties& p : properties.subspan(1)) {
    DCHECK(!p.shape().unknown_rank());
    if (p.shape().dim_size() < result->dim_size()) continue;
    if (p.shape().dim_size() > result->dim_size()) {
      choose_shape_with_higher_rank(&p.shape());
      continue;
    }
    if (result->dim_size() < 1) continue;
    if (p.shape().dim(0).size() < 0 || result->dim(0).size() < 0) {
      if (p.shape().dim(0).size() < 0 && result->dim(0).size() >= 0) {
        result = &p.shape();
      } else {
        max_batch_dim_value =
            std::max<int>(max_batch_dim_value, p.shape().dim(0).size());
      }
      continue;
    }
    if (p.shape().dim(0).size() > result->dim(0).size()) {
      result = &p.shape();
      max_batch_dim_value = result->dim(0).size();
    }
  }
  if (result->dim_size() > 0 && result->dim(0).size() < 0) {
    if (max_batch_dim_value <= 1) {
      return result;
    } else {
      return std::nullopt;
    }
  }
  return result;
}
absl::Span<const OpInfo::TensorProperties> GetInputsToDeterminateBatchSize(
    const Node* node, const std::vector<OpInfo::TensorProperties>& all_inputs) {
  static std::set<string> broadcast_supporting_ops = {
      "Add",
      "AddV2",
      "Mul",
      "Sub",
      "Div",
      "FloorDiv",
      "RealDiv",
      "Minimum",
      "Maximum",
      "Pow",
      "BiasAdd",
      "SquaredDifference",
      "BatchMatMul",
      "BatchMatMulV2",
  };
  const string& op = node->def().op();
  if (op == "Conv2DBackpropInput" || op == "Conv3DBackpropInputV2") {
    DCHECK_EQ(all_inputs.size(), 3);
    return absl::MakeSpan(all_inputs).subspan(2, 1);
  }
  if (broadcast_supporting_ops.count(op)) {
    return absl::MakeSpan(all_inputs);
  }
  return absl::MakeSpan(all_inputs).subspan(0, 1);
}
bool OperationCanBeTranslatedToImplicitBatch(
    const grappler::GraphProperties* graph_properties, const Node* node) {
  VLOG(3) << "process node " << node->name();
  if (node->num_inputs() == 0) return true;
  if (!graph_properties || !graph_properties->HasInputProperties(node->name()))
    return false;
  VLOG(3) << "input shapes "
          << TensorPropertiesToString(
                 graph_properties->GetInputProperties(node->name()));
  const std::vector<OpInfo::TensorProperties>& all_input_properties =
      graph_properties->GetInputProperties(node->name());
  absl::Span<const OpInfo::TensorProperties> input_properties =
      GetInputsToDeterminateBatchSize(node, all_input_properties);
  if (absl::c_any_of(input_properties, [](const OpInfo::TensorProperties& p) {
        return p.shape().unknown_rank();
      })) {
    return false;
  }
  std::optional<const TensorShapeProto*> leading_shape =
      FindLeadingShape(input_properties);
  return leading_shape.has_value() && leading_shape.value()->dim_size() >= 2;
}
bool HasDynamicNonBatchDimension(const OpInfo::TensorProperties& prop) {
  const TensorShapeProto& shape = prop.shape();
  if (shape.unknown_rank()) return true;
  if (shape.dim_size() == 0) return false;
  for (int i = 1; i < shape.dim_size(); ++i) {
    if (shape.dim(i).size() <= -1) {
      return true;
    }
  }
  return false;
}
bool OperationHasDynamicNonBatchDimension(
    const grappler::GraphProperties* graph_properties, const Node* node) {
  VLOG(3) << "process node " << node->name();
  if (node->num_inputs() == 0 || node->num_outputs() == 0) return false;
  if (!graph_properties->HasOutputProperties(node->name())) return true;
  VLOG(3) << "output shapes "
          << TensorPropertiesToString(
                 graph_properties->GetOutputProperties(node->name()));
  return HasDynamicNonBatchDimension(
      graph_properties->GetOutputProperties(node->name()).at(0));
}
void ContractEdge(SimpleEdge* edge, SimpleGraph* graph,
                  std::vector<const SimpleEdge*>* remove_edges) {
  auto src = edge->src();
  auto dst = edge->dst();
  std::vector<const SimpleEdge*> in_edges(dst->in_edges().begin(),
                                          dst->in_edges().end());
  for (const SimpleEdge* in_edge : in_edges) {
    if (in_edge->IsControlEdge()) {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        graph->AddControlEdge(e->src(), src);
      }
    } else {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        if (e->src() == graph->source_node()) {
          graph->AddEdge(e->src(), e->src_output(), src, Graph::kControlSlot);
        } else {
          graph->AddEdge(e->src(), e->src_output(), src, 0 );
        }
      }
    }
  }
  std::vector<const SimpleEdge*> out_edges(dst->out_edges().begin(),
                                           dst->out_edges().end());
  for (const SimpleEdge* out_edge : out_edges) {
    if (out_edge->IsControlEdge()) {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
      graph->AddControlEdge(src, e->dst());
    } else {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
      if (e->dst() == graph->sink_node()) {
        VLOG(1) << " edge to sink node " << src->name() << " -> "
                << e->dst()->name();
        graph->AddEdge(src, Graph::kControlSlot, e->dst(), e->dst_input());
      } else {
        graph->AddEdge(src, 0 , e->dst(), e->dst_input());
      }
    }
  }
  for (const auto& in_edge : dst->in_edges()) {
    remove_edges->push_back(in_edge);
  }
  for (const auto& out_edge : dst->out_edges()) {
    remove_edges->push_back(out_edge);
  }
}
ClusterBatchSize GetClusterBatchSizeForNode(
    const grappler::GraphProperties* graph_properties, const Node* node,
    bool use_implicit_batch) {
  ClusterBatchSize cluster_batch_size;
  if (!use_implicit_batch || !node || node->num_inputs() == 0) {
    return cluster_batch_size;
  }
  const NodeDef& node_def = node->def();
  if (node_def.attr().count(kTftrtOpMaxBatchSizeAttr)) {
    cluster_batch_size.SetMaxBatchSize(
        node_def.attr().at(kTftrtOpMaxBatchSizeAttr).i());
  }
  if (!graph_properties ||
      !graph_properties->HasInputProperties(node->name())) {
    VLOG(3) << "doesn't have input property";
    return cluster_batch_size;
  }
  const std::vector<OpInfo::TensorProperties>& input_properties =
      graph_properties->GetInputProperties(node->name());
  std::optional<const TensorShapeProto*> optional_leading_shape =
      FindLeadingShape(GetInputsToDeterminateBatchSize(node, input_properties));
  DCHECK(optional_leading_shape.has_value());
  const TensorShapeProto* leading_shape = optional_leading_shape.value();
  DCHECK(!leading_shape->unknown_rank() && leading_shape->dim_size() >= 2);
  VLOG(3) << "set batch size as " << leading_shape->dim(0).size();
  return cluster_batch_size.SetBatchSize(leading_shape->dim(0).size());
}
void AddSegmentForNode(const grappler::GraphProperties* graph_properties,
                       std::vector<UnionFind<SimpleNode*>>* segments,
                       SimpleNode* node,
                       const DeviceNameUtils::ParsedName& device_name,
                       bool use_implicit_batch) {
  tensorflow::profiler::TraceMe activity(
      "AddSegmentForNode", tensorflow::profiler::TraceMeLevel::kInfo);
  ClusterProperty property(
      GetClusterBatchSizeForNode(graph_properties,
                                 node == nullptr ? nullptr : node->tf_node(),
                                 use_implicit_batch),
      device_name);
  segments->emplace_back(node, std::move(property));
}
}  
Status ExportNonConversionReportToCSV(
    string filename,
    std::map<string, std::map<string, int>>& nonconverted_ops_map,
    string sep = "|") {
  tensorflow::profiler::TraceMe activity(
      "ExportNonConversionReportToCSV",
      tensorflow::profiler::TraceMeLevel::kInfo);
  std::unique_ptr<WritableFile> csv_file;
  auto open_status = Env::Default()->NewWritableFile(filename, &csv_file);
  if (!open_status.ok()) {
    return errors::Internal("Failed to open output file: `", filename, "`");
  }
  LOG(WARNING) << "TF-TRT Non-Conversion Report saved at: `" << filename << "`";
  std::ostringstream sstream;
  sstream << "OP Name" << sep << "Reason" << sep << "Count" << std::endl;
  for (auto& op_details : nonconverted_ops_map) {
    auto op_name = op_details.first;
    auto op_data = op_details.second;
    for (auto& reject_data : op_data) {
      auto reason = reject_data.first;
      auto count = reject_data.second;
      sstream << op_name << sep << reason << sep << count << std::endl;
    }
  }
  auto append_status = csv_file->Append(sstream.str());
  if (!append_status.ok()) {
    return errors::Internal("Error writing to output file `", filename, "`.");
  }
  auto close_status = csv_file->Close();
  if (!close_status.ok()) {
    return errors::Internal("Error closing the file `", filename,
                            "`. The file might be corrupted.");
  }
  return OkStatus();
}
string GenerateNonConversionReport(
    std::map<string, std::map<string, int>>& nonconverted_ops_map) {
  tensorflow::profiler::TraceMe activity(
      "GenerateNonConversionReport", tensorflow::profiler::TraceMeLevel::kInfo);
  string detailed_report_var;
  TF_CHECK_OK(ReadStringFromEnvVar("TF_TRT_SHOW_DETAILED_REPORT",
                                   "", &detailed_report_var));
  bool show_detailed_conversion_report = false;
  if (detailed_report_var != "") {
    if (detailed_report_var.find_first_not_of("-0123456789") != string::npos) {
      const Status status = ExportNonConversionReportToCSV(
          detailed_report_var, nonconverted_ops_map);
      if (!status.ok()) {
        LOG(ERROR) << "Problem encountered while generating the TF-TRT "
                   << "Non-Conversion Report in CSV Format:\n"
                   << status.message();
      }
      show_detailed_conversion_report = true;
    } else if (std::stoi(detailed_report_var) >= 1) {
      show_detailed_conversion_report = true;
    }
  }
  string unsupported_op_report =
      StrCat("\n\n", string(80, '#'), "\n",
             "TensorRT unsupported/non-converted OP Report:");
  int total_nonconverted_ops{0};
  using ReasonCounterVector = std::vector<std::pair<string, int>>;
  using NotConvertedOPTuple = std::tuple<string, int, ReasonCounterVector>;
  std::vector<NotConvertedOPTuple> nonconverted_ops_vec;
  for (auto& nonconverted_op_data : nonconverted_ops_map) {
    int total_nonconverted_op{0};
    ReasonCounterVector reason_occurances_vect;
    auto op_name = nonconverted_op_data.first;
    auto op_data = nonconverted_op_data.second;
    for (auto& notconversion_reason_data : op_data) {
      auto reason_count = notconversion_reason_data.second;
      total_nonconverted_op += reason_count;
      reason_occurances_vect.push_back(notconversion_reason_data);
    }
    std::sort(reason_occurances_vect.begin(), reason_occurances_vect.end(),
              [](const std::pair<string, int>& a,
                 const std::pair<string, int>& b) -> bool {
                return a.second > b.second;
              });
    nonconverted_ops_vec.push_back(std::make_tuple(
        op_name, total_nonconverted_op, reason_occurances_vect));
  }
  std::sort(nonconverted_ops_vec.begin(), nonconverted_ops_vec.end(),
            [](const NotConvertedOPTuple& a, const NotConvertedOPTuple& b) {
              return std::get<1>(a) > std::get<1>(b);
            });
  for (auto& notconverted_op_detail : nonconverted_ops_vec) {
    auto& op_name = std::get<0>(notconverted_op_detail);
    auto& op_total_nonconverted = std::get<1>(notconverted_op_detail);
    total_nonconverted_ops += op_total_nonconverted;
    unsupported_op_report = StrCat(unsupported_op_report, "\n\t- ", op_name,
                                   " -> ", op_total_nonconverted, "x");
    if (show_detailed_conversion_report) {
      auto& nonconverted_ops_details = std::get<2>(notconverted_op_detail);
      for (auto& nonconversion_details : nonconverted_ops_details) {
        auto& reason = nonconversion_details.first;
        auto& reason_count = nonconversion_details.second;
        if (reason_count == 0) {
          continue;
        }
        unsupported_op_report = StrCat(unsupported_op_report, "\n\t\t- ",
                                       "[Count: ", reason_count, "x] ", reason);
      }
      unsupported_op_report = StrCat(unsupported_op_report, "\n");
    }
  }
  unsupported_op_report =
      StrCat(unsupported_op_report, "\n", string(80, '-'),
             "\n\t- Total nonconverted OPs: ", total_nonconverted_ops,
             "\n\t- Total nonconverted OP Types: ", nonconverted_ops_map.size(),
             "\nFor more information see https:
             "/frameworks/tf-trt-user-guide/index.html#supported-ops.", "\n",
             string(80, '#'), "\n");
  return unsupported_op_report;
}
Status SegmentGraph(const Graph* tf_graph,
                    const grappler::GraphProperties* graph_properties,
                    const std::function<Status(const Node*)>& candidate_fn,
                    const std::function<bool(const Edge*)>& input_candidate_fn,
                    const std::function<bool(const Edge*)>& output_candidate_fn,
                    const SegmentOptions& options, SegmentVector* segments) {
  tensorflow::profiler::TraceMe activity(
      "SegmentGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  if (!options.use_implicit_batch && !options.allow_dynamic_non_batch_dim) {
    return errors::Internal(
        "Explicit batch mode should allow dynamic non-batch dimensions");
  }
  if (options.use_implicit_batch && !options.maximum_batch_size.has_value()) {
    return errors::Internal("Implicit batch mode requires maximum_batch_size");
  }
  if (!options.allow_dynamic_non_batch_dim && !graph_properties) {
    return errors::Internal(
        "Need graph propertities to disallow dynamic non-batch dimensions");
  }
  auto graph = std::unique_ptr<SimpleGraph>(new SimpleGraph(tf_graph));
  const absl::flat_hash_set<string> tftrt_op_denylist = [] {
    string tftrt_op_denylist_str;
    TF_CHECK_OK(ReadStringFromEnvVar("TF_TRT_OP_DENYLIST", "",
                                     &tftrt_op_denylist_str));
    absl::flat_hash_set<string> tftrt_op_denylist{};
    for (const auto& x : str_util::Split(tftrt_op_denylist_str, ",")) {
      tftrt_op_denylist.insert(x);
    }
    tftrt_op_denylist.rehash(0);
    return tftrt_op_denylist;
  }();
  std::map<string, std::map<string, int>> nonconverted_ops_map = {};
  std::vector<UnionFind<SimpleNode*>> node_segments;
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    SimpleNode* node = graph->FindNodeId(i);
    if (!node) {
      VLOG(3) << "Node " << i << " doesn't exist in the graph";
      continue;
    }
    const string node_op_type{node->tf_node()->type_string()};
    auto exclude_node = [&](absl::string_view reason) {
      VLOG(1) << "Not a TF-TRT candidate, " << "(Op type: " << node_op_type
              << "), " << "(Op name: " << node->name() << "), "
              << "(Reason: " << reason << ")";
      nonconverted_ops_map[node_op_type][string(reason)]++;
      node = nullptr;
    };
    std::optional<DeviceNameUtils::ParsedName> device_name =
        GetDeviceParsedName(node->tf_node());
    if (!device_name.has_value() ||
        (device_name->has_type && device_name->type != "GPU")) {
      exclude_node("node can't be placed on GPU");
    } else if (options.exclude_node_list.count(node->name()) != 0) {
      exclude_node(
          "excluded by segmenter option. Most likely an input or "
          "output node.");
    } else if (options.use_implicit_batch &&
               !OperationCanBeTranslatedToImplicitBatch(graph_properties,
                                                        node->tf_node())) {
      exclude_node(
          "implicit batch mode requires input shape with at least two "
          "dimensions");
    } else if (!options.allow_dynamic_non_batch_dim &&
               OperationHasDynamicNonBatchDimension(graph_properties,
                                                    node->tf_node())) {
      exclude_node("dynamic non-batch dimensions not allowed");
    } else {
      const Status status = candidate_fn(node->tf_node());
      if (!status.ok()) {
        exclude_node(status.message());
      } else if (tftrt_op_denylist.contains(node->tf_node()->type_string())) {
        LOG_WARNING_WITH_PREFIX
            << "Denylisted as TF-TRT candidate, "
            << "(Op type: " << node->tf_node()->type_string() << "), "
            << "(Op name: " << node->name() << ")";
        exclude_node("Denylisted with the env var TF_TRT_OP_DENYLIST");
      } else {
        VLOG(2) << "Accepted as a TF-TRT candidate, "
                << "(Op type: " << node->tf_node()->type_string() << "), "
                << "(Op name: " << node->name();
      }
    }
    AddSegmentForNode(graph_properties, &node_segments, node, *device_name,
                      options.use_implicit_batch);
  }
  LOG(WARNING) << GenerateNonConversionReport(nonconverted_ops_map);
  std::vector<const SimpleNode*> order;
  order.reserve(graph->num_node_ids());
  StableDFS(*graph, false, {graph->source_node()},
            nullptr, [&order](const SimpleNode* n) {
              order.push_back(n);
              return true;
            });
  for (const SimpleNode* node : order) {
    VLOG(3) << "Trying node " << node->name() << " id=" << node->id();
    if (node_segments[node->id()].Value() == nullptr) {
      VLOG(3) << "... not a TRT candidate";
      continue;
    }
    ClusterBatchSize expected_batch_size =
        node_segments[node->id()].Property().BatchSize();
    DeviceNameUtils::ParsedName expected_device_name =
        node_segments[node->id()].Property().DeviceName();
    VLOG(3) << "batch size " << expected_batch_size;
    while (true) {
      std::set<const SimpleEdge*, SimpleEdgePtrCompare> contract_edges;
      for (const SimpleEdge* out_edge : node->out_edges()) {
        VLOG(3) << "... out node " << out_edge->dst()->name() << " ( "
                << out_edge->dst()->id() << " <- " << node->id() << " )";
        if (out_edge->IsControlEdge()) {
          VLOG(3) << "... ... Control Edge, Skipping";
          continue;
        }
        UnionFind<SimpleNode*>* out_cluster =
            &node_segments[out_edge->dst()->id()];
        if (out_cluster->Value() == nullptr) {
          VLOG(3) << "... ... not a TRT candidate";
          continue;
        }
        ClusterBatchSize out_batch_size = out_cluster->Property().BatchSize();
        ClusterBatchSize merged_batch_size = expected_batch_size;
        if (!merged_batch_size.MergeIfCompatible(out_batch_size)) {
          VLOG(3) << "... ... incompatible batch sizes "
                  << expected_batch_size.ToString() << " "
                  << out_batch_size.ToString();
          continue;
        }
        const DeviceNameUtils::ParsedName& out_device_name =
            out_cluster->Property().DeviceName();
        std::optional<DeviceNameUtils::ParsedName> merged_device_name =
            MergeIfCompatible(expected_device_name, out_device_name);
        if (!merged_device_name.has_value()) {
          VLOG(3) << "... ... incompatible device names "
                  << expected_device_name << " " << out_device_name;
          continue;
        }
        if (CanContractEdge(out_edge, graph)) {
          VLOG(3) << "... ... can contract. new batch size "
                  << merged_batch_size.ToString();
          contract_edges.insert(out_edge);
          expected_batch_size = merged_batch_size;
          expected_device_name = *merged_device_name;
        } else {
          VLOG(3) << "... ... cannot contract, would form cycle";
        }
      }
      if (contract_edges.empty()) {
        break;
      }
      while (!contract_edges.empty()) {
        const SimpleEdge* contract_edge = *contract_edges.begin();
        const SimpleNode* src = contract_edge->src();
        const SimpleNode* dst = contract_edge->dst();
        VLOG(3) << "Merge " << src->name() << " <- " << dst->name() << " ("
                << src->id() << " <- " << dst->id();
        TF_RETURN_IF_ERROR(
            node_segments[src->id()].Merge(&node_segments[dst->id()]));
        SimpleEdge* e = const_cast<SimpleEdge*>(contract_edge);
        std::vector<const SimpleEdge*> remove_edges;
        ContractEdge(e, graph.get(), &remove_edges);
        for (const SimpleEdge* r : remove_edges) {
          contract_edges.erase(r);
          graph->RemoveEdge(r);
        }
      }
      if (expected_batch_size !=
          node_segments[node->id()].Property().BatchSize()) {
        return errors::Internal(
            "expected batch size is not the same as the actual batch size");
      }
      if (expected_device_name !=
          node_segments[node->id()].Property().DeviceName()) {
        return errors::Internal(
            "expected device name is not the same as the actual device name");
      }
    }
  }
  std::map<string, Segment> sg_map;
  for (auto& u : node_segments) {
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      sg_map[u.ParentValue()->name()].nodes.insert(u.Value()->tf_node());
    }
    if ((u.Value() != nullptr) && (u.ParentValue() == u.Value())) {
      sg_map[u.Value()->name()].property = u.Property();
    }
  }
  for (auto& itr : sg_map) {
    std::set<const Node*, NodePtrCompare>& segment_nodes = itr.second.nodes;
    VLOG(1) << "Segment original size: " << segment_nodes.size();
    while (true) {
      std::deque<const Node*> in_nodes_que, out_nodes_que;
      for (auto node : segment_nodes) {
        bool added = false;
        for (const Edge* edge : node->in_edges()) {
          if (!edge->IsControlEdge() && !edge->src()->IsSource() &&
              !segment_nodes.count(edge->src())) {  
            if (!input_candidate_fn(edge)) {
              in_nodes_que.push_back(node);
              added = true;
              break;
            }
          }
        }
        if (added) continue;  
        for (const Edge* edge : node->out_edges()) {
          if (!edge->dst()->IsSink() && !edge->IsControlEdge() &&
              !segment_nodes.count(edge->dst())) {  
            if (!output_candidate_fn(edge)) {
              out_nodes_que.push_back(node);
              break;
            }
          }
        }
      }
      if (in_nodes_que.empty() && out_nodes_que.empty()) {
        break;
      }
      auto remove_nodes = [&segment_nodes](bool is_input_nodes,
                                           std::deque<const Node*>* que) {
        std::set<const Node*, NodePtrCompare> visited;
        std::set<const Node*, NodePtrCompare> logged(que->begin(), que->end());
        while (!que->empty()) {
          auto node = que->front();
          que->pop_front();
          if (!visited.insert(node).second) continue;
          segment_nodes.erase(node);
          for (auto in : (is_input_nodes || node->type_string() == "Const")
                             ? node->in_nodes()
                             : node->out_nodes()) {
            if (segment_nodes.count(in)) {
              que->push_back(in);
              if (VLOG_IS_ON(2)) {
                if (!logged.count(in)) {
                  VLOG(2) << "----> Need to remove node " << in->name()
                          << " because one of its "
                          << (is_input_nodes ? "output" : "input")
                          << " nodes in the graph was removed: "
                          << node->name();
                  logged.insert(in);
                }
              }
            }
          }
        }
      };
      remove_nodes(true, &in_nodes_que);
      remove_nodes(false, &out_nodes_que);
    }
    VLOG(1) << "Segment new size: " << segment_nodes.size();
  }
  std::vector<int> effective_nodes_counts;
  for (const auto& itr : sg_map) {
    const string& segment_root = itr.first;
    std::set<const Node*, NodePtrCompare> segment_nodes(
        itr.second.nodes.begin(), itr.second.nodes.end());
    if (VLOG_IS_ON(1) && !segment_nodes.empty()) {
      string s;
      for (auto node : segment_nodes) {
        StrAppend(&s, "\n[Op type: ", node->type_string(), "] ", node->name());
      }
      VLOG(1) << "Nodes in segment " << segments->size()
              << " with parent=" << segment_root << ":" << s;
    }
    const int num_effective_nodes = std::count_if(
        segment_nodes.begin(), segment_nodes.end(), [](const Node* node) {
          static auto noops =
              new std::set<string>{"Identity", "Snapshot", "StopGradient"};
          return noops->count(node->type_string()) == 0;
        });
    if (num_effective_nodes == 0 ||
        num_effective_nodes < options.minimum_segment_size) {
      VLOG(1) << "Segment " << segments->size() << " has only "
              << num_effective_nodes << " effective nodes, dropping";
      continue;
    }
    segments->emplace_back(itr.second.property, segment_nodes);
    effective_nodes_counts.push_back(num_effective_nodes);
  }
  int64_t max_trt_engine_ops;
  TF_CHECK_OK(ReadInt64FromEnvVar("TF_TRT_MAX_ALLOWED_ENGINES",
                                  20, &max_trt_engine_ops));
  if (max_trt_engine_ops <= 0) {
    LOG(WARNING) << "The environment variable TF_TRT_MAX_ALLOWED_ENGINES is "
                 << "<= 0. TF-TRT did not limit the number of TensorRT engines "
                 << "created.";
  } else {
    if (segments->size() > max_trt_engine_ops) {
      LOG(WARNING) << "A total of " << segments->size() << " segments with at "
                   << "least minimum_segment_size="
                   << options.minimum_segment_size << " nodes have been found. "
                   << "TF-TRT will only convert the " << max_trt_engine_ops
                   << " largest segments. You can change this behavior by "
                   << "modifying the environment variable "
                   << "TF_TRT_MAX_ALLOWED_ENGINES=" << max_trt_engine_ops;
      std::vector<int> indices(segments->size());
      std::iota(indices.begin(), indices.end(), 0);
      std::stable_sort(indices.begin(), indices.end(),
                       [&effective_nodes_counts](int i1, int i2) {
                         return effective_nodes_counts[i1] >
                                effective_nodes_counts[i2];
                       });
      std::vector<bool> mask = std::vector<bool>(segments->size(), false);
      for (int i = 0; i < max_trt_engine_ops; i++) {
        mask[indices[i]] = true;
      }
      int j = 0;
      VLOG(1) << "The following segments have been accepted by TF-TRT:";
      for (int i = 0; i < segments->size(); i++) {
        if (mask[i]) {
          VLOG(1) << "[*] Segment " << i
                  << " [node count: " << effective_nodes_counts[i]
                  << "] accepted. Re-assigned " << "segment id=" << j;
          segments->at(j) = segments->at(i);
          j++;
        }
      }
      VLOG(1) << "The following segments have been rejected by TF-TRT:";
      for (int i = 0; i < segments->size(); i++) {
        if (!mask[i]) {
          VLOG(1) << "[*] Segment " << i
                  << " [node count: " << effective_nodes_counts[i]
                  << "] rejected.";
        }
      }
      segments->resize(max_trt_engine_ops);
    } else {
      LOG(WARNING) << "The environment variable TF_TRT_MAX_ALLOWED_ENGINES="
                   << max_trt_engine_ops << " has no effect since there are "
                   << "only " << segments->size() << " TRT Engines with  at "
                   << "least minimum_segment_size="
                   << options.minimum_segment_size << " nodes.";
    }
  }
  return OkStatus();
}
}  
}  
}  
#endif  