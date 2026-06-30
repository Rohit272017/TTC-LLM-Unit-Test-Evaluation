#include "tensorflow/core/grappler/optimizers/data/make_deterministic.h"
#include <algorithm>
#include <utility>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/split_utils.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
namespace {
constexpr char kInterleaveOp[] = "InterleaveDataset";
constexpr char kParallelInterleaveOp[] = "ParallelInterleaveDataset";
constexpr char kLegacyParallelInterleaveOp[] =
    "LegacyParallelInterleaveDatasetV2";
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kParallelMapOpV2[] = "ParallelMapDatasetV2";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kParallelBatchOp[] = "ParallelBatchDataset";
constexpr char kPrefetchOp[] = "PrefetchDataset";
constexpr std::array<const char*, 9> kDeterministicStatefulOps = {
    "TextLineDataset", "FixedLengthRecordDataset", "TFRecordDataset",
    "TensorSliceDataset", "RangeDataset", "SSTableDataset", "RecordIODataset",
    "Print", "Assert"};
constexpr std::array<const char*, 13> kDeterministicStatefulOpsWhenAsync = {
    "RandomUniform",
    "RandomUniformInt",
    "RandomStandardNormal",
    "ParameterizedTruncatedNormal",
    "TruncatedNormal",
    "RandomShuffle",
    "Multinomial",
    "RandomGamma",
    "RandomGammaGrad",
    "RandomPoisson",
    "RandomCrop",
    "SampleDistortedBoundingBox",
    "SampleDistortedBoundingBoxV2"};
bool IsDeterministicWhenRunInParallel(const std::string& stateful_op) {
  for (auto op_in_array : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  return false;
}
bool IsDeterministicWhenRunAsynchronously(const std::string& stateful_op) {
  for (auto op_in_array : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  for (auto op_in_array : kDeterministicStatefulOpsWhenAsync) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  return false;
}
bool IsParallelInterleave(const std::string& op) {
  return data::MatchesAnyVersion(kParallelInterleaveOp, op) ||
         op == kLegacyParallelInterleaveOp;
}
bool IsParallelMap(const std::string& op) {
  return data::MatchesAnyVersion(kParallelMapOp, op);
}
bool IsParallelBatch(const std::string& op) {
  return data::MatchesAnyVersion(kParallelBatchOp, op);
}
bool IsMapAndBatch(const std::string& op) {
  return data::MatchesAnyVersion(kMapAndBatchOp, op);
}
bool IsPrefetch(const std::string& op) {
  return data::MatchesAnyVersion(kPrefetchOp, op);
}
bool IntroducesFunctionParallelism(const std::string& op) {
  return IsParallelInterleave(op) || IsParallelMap(op) || IsMapAndBatch(op);
}
bool IntroducesAsynchrony(const std::string& op) {
  return IntroducesFunctionParallelism(op) || IsPrefetch(op) ||
         IsParallelBatch(op);
}
absl::flat_hash_map<absl::string_view, const NodeDef*> NameToNode(
    const FunctionDef& function) {
  absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node;
  for (const NodeDef& node : function.node_def()) {
    name_to_node.insert({node.name(), &node});
  }
  return name_to_node;
}
NodeDef* GetMutableNode(const string& node_name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  return graph->graph()->mutable_node(index);
}
Status ConvertMapOrInterleave(const string& node_name,
                              MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  auto Targuments = node->attr().find("Targuments");
  if (Targuments == node->attr().end()) {
    return errors::Internal("Failed to find Targuments attribute for node ",
                            node_name);
  }
  int num_inputs_after_rewrite;
  if (IsParallelInterleave(node->op())) {
    node->set_op(kInterleaveOp);
    num_inputs_after_rewrite = 3 + Targuments->second.list().type_size();
  } else {
    DCHECK(IsParallelMap(node->op()));
    node->set_op(kMapOp);
    num_inputs_after_rewrite = 1 + Targuments->second.list().type_size();
  }
  int inputs_processed = 0;
  for (int i = 0; i < node->input_size(); i++) {
    std::string input = node->input(i);
    if (IsControlInput(input)) {
      continue;
    }
    if (inputs_processed >= num_inputs_after_rewrite) {
      node->set_input(i, absl::StrCat("^", input));
    }
    inputs_processed++;
  }
  if (inputs_processed < num_inputs_after_rewrite) {
    return errors::Internal("Found only ", inputs_processed, " inputs to node ",
                            node_name, ", but expected to find at least ",
                            num_inputs_after_rewrite);
  }
  node->mutable_attr()->erase("deterministic");
  node->mutable_attr()->erase("sloppy");
  return absl::OkStatus();
}
absl::flat_hash_set<absl::string_view> GetAllTransitiveDependencies(
    const FunctionDef& function_def,
    const absl::flat_hash_set<absl::string_view>& nodes) {
  std::vector<absl::string_view> nodes_to_process;
  std::copy(nodes.begin(), nodes.end(), std::back_inserter(nodes_to_process));
  absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node =
      NameToNode(function_def);
  absl::flat_hash_set<absl::string_view> dependencies;
  while (!nodes_to_process.empty()) {
    absl::string_view node_name = nodes_to_process.back();
    nodes_to_process.pop_back();
    if (dependencies.contains(node_name)) {
      continue;
    }
    dependencies.insert(node_name);
    auto iter = name_to_node.find(node_name);
    if (iter == name_to_node.end()) {
      continue;
    }
    for (absl::string_view inp : iter->second->input()) {
      absl::string_view inp_node = inp.substr(0, inp.find(':'));
      if (inp_node.at(0) == '^') {
        inp_node = inp_node.substr(1);
      }
      if (name_to_node.contains(inp_node)) {
        nodes_to_process.push_back(inp_node);
      }
    }
  }
  return dependencies;
}
Status SplitMap(
    const FunctionLibraryDefinition& library, const string& map_node_name,
    MutableGraphView* graph,
    const absl::flat_hash_set<absl::string_view>& nondeterministic_nodes) {
  NodeDef* map_node = GetMutableNode(map_node_name, graph);
  NameAttrList func = map_node->attr().at("f").func();
  const FunctionDef* function_def = library.Find(func.name());
  if (!function_def) {
    return errors::Internal("Could not look up function ", func.name(),
                            " in FunctionLibraryDefinition");
  }
  absl::flat_hash_set<absl::string_view> nodes_to_move =
      GetAllTransitiveDependencies(*function_def, nondeterministic_nodes);
  VLOG(2) << "Will move nodes to nonparallel function: "
          << absl::StrJoin(nodes_to_move, ", ");
  int64_t num_captured_arguments =
      map_node->attr().find("Targuments")->second.list().type_size();
  TF_ASSIGN_OR_RETURN(
      split_utils::SplitResults split_results,
      split_utils::SplitFunction(*function_def, nodes_to_move,
                                 num_captured_arguments, library));
  if (split_results.first_function_output_types.empty()) {
    return errors::Unimplemented(
        "The case where the first function has no outputs is unimplemented.");
  }
  bool is_map_and_batch = map_node->op() == kMapAndBatchOp;
  NodeDef* first_map_node_ptr;
  {
    NodeDef first_map_node;
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("make_deterministic_sequential_map/", map_node->name()),
        graph->graph(), &first_map_node);
    first_map_node.set_op(kMapOp);
    int num_control_deps = NumControlInputs(*map_node);
    int num_extra_inputs = is_map_and_batch ? 3 : 1;
    int control_deps_index = map_node->input_size() - num_control_deps;
    int extra_inputs_index = control_deps_index - num_extra_inputs;
    for (int i = 0; i < extra_inputs_index; i++) {
      DCHECK(!IsControlInput(map_node->input(i)));
      first_map_node.add_input(map_node->input(i));
    }
    for (int i = extra_inputs_index; i < control_deps_index; i++) {
      DCHECK(!IsControlInput(map_node->input(i)));
      first_map_node.add_input(absl::StrCat("^", map_node->input(i)));
    }
    for (int i = control_deps_index; i < map_node->input_size(); i++) {
      DCHECK(IsControlInput(map_node->input(i)));
      first_map_node.add_input(map_node->input(i));
    }
    NameAttrList* name_attr_list =
        (*first_map_node.mutable_attr())["f"].mutable_func();
    name_attr_list->set_name(split_results.first_function.signature().name());
    graph_utils::CopyAttribute("Targuments", *map_node, &first_map_node);
    for (auto key : {"use_inter_op_parallelism", "preserve_cardinality"}) {
      if (gtl::FindOrNull(map_node->attr(), key)) {
        graph_utils::CopyAttribute(key, *map_node, &first_map_node);
      }
    }
    AddNodeAttr("output_types", split_results.first_function_output_types,
                &first_map_node);
    TensorShapeProto unknown_shape;
    unknown_shape.set_unknown_rank(true);
    std::vector<TensorShapeProto> output_shapes(
        split_results.first_function_output_types.size(), unknown_shape);
    AddNodeAttr("output_shapes", output_shapes, &first_map_node);
    first_map_node_ptr = graph->AddNode(std::move(first_map_node));
  }
  NodeDef* second_map_node_ptr;
  {
    NodeDef second_map_node;
    string node_name =
        map_node->op() == kMapAndBatchOp ? "map_and_batch" : "parallel_map";
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("make_deterministic_parallel_", node_name, "/",
                        map_node->name()),
        graph->graph(), &second_map_node);
    second_map_node.set_op(map_node->op());
    second_map_node.add_input(first_map_node_ptr->name());
    for (int i = 1; i < map_node->input_size(); i++) {
      second_map_node.add_input(map_node->input(i));
    }
    NameAttrList* name_attr_list =
        (*second_map_node.mutable_attr())["f"].mutable_func();
    name_attr_list->set_name(split_results.second_function.signature().name());
    graph_utils::CopyAttribute("Targuments", *map_node, &second_map_node);
    graph_utils::CopyAttribute("output_types", *map_node, &second_map_node);
    graph_utils::CopyAttribute("output_shapes", *map_node, &second_map_node);
    if (!is_map_and_batch) {
      AddNodeAttr("deterministic", "true", &second_map_node);
    }
    for (auto key : {"use_inter_op_parallelism", "preserve_cardinality"}) {
      if (gtl::FindOrNull(map_node->attr(), key)) {
        graph_utils::CopyAttribute(key, *map_node, &second_map_node);
      }
    }
    second_map_node_ptr = graph->AddNode(std::move(second_map_node));
  }
  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(map_node->name(), second_map_node_ptr->name()));
  *graph->graph()->mutable_library()->mutable_function()->Add() =
      split_results.first_function;
  *graph->graph()->mutable_library()->mutable_function()->Add() =
      split_results.second_function;
  return absl::OkStatus();
}
Status ConvertBatch(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  node->set_op(kBatchV2Op);
  std::string num_parallel_calls_input = node->input(2);
  node->set_input(2, node->input(3));
  node->set_input(3, absl::StrCat("^", num_parallel_calls_input));
  node->mutable_attr()->erase("deterministic");
  return absl::OkStatus();
}
Status ConvertMapAndBatch(const string& node_name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  const NodeDef& orig_node = graph->graph()->node(index);
  auto Targuments = orig_node.attr().find("Targuments");
  if (Targuments == orig_node.attr().end()) {
    return errors::Internal("Failed to find Targuments attribute for node ",
                            node_name);
  }
  NodeDef new_map_node;
  new_map_node.set_op(kMapOp);
  graph_utils::SetUniqueGraphNodeName(kMapOp, graph->graph(), &new_map_node);
  int num_map_inputs = 1 + Targuments->second.list().type_size();
  for (int i = 0; i < num_map_inputs; i++) {
    new_map_node.add_input(orig_node.input(i));
  }
  for (int i = num_map_inputs; i < orig_node.input_size(); i++) {
    if (IsControlInput(orig_node.input(i))) {
      new_map_node.add_input(orig_node.input(i));
    } else {
      new_map_node.add_input(absl::StrCat("^", orig_node.input(i)));
    }
  }
  for (auto key : {"f", "Targuments", "output_types"}) {
    graph_utils::CopyAttribute(key, orig_node, &new_map_node);
  }
  for (auto key : {"preserve_cardinality"}) {
    if (gtl::FindOrNull(new_map_node.attr(), key)) {
      graph_utils::CopyAttribute(key, orig_node, &new_map_node);
    }
  }
  auto orig_output_shapes = orig_node.attr().find("output_shapes");
  if (orig_output_shapes == orig_node.attr().end()) {
    return errors::Internal("Failed to find output_shapes attribute for node ",
                            node_name);
  }
  AttrValue& map_output_shapes =
      (*new_map_node.mutable_attr())["output_shapes"];
  for (const TensorShapeProto& orig_shape :
       orig_output_shapes->second.list().shape()) {
    TensorShapeProto* new_shape = map_output_shapes.mutable_list()->add_shape();
    if (orig_shape.unknown_rank()) {
      new_shape->set_unknown_rank(true);
    } else if (orig_shape.dim_size() == 0) {
      return errors::Internal(
          "Output shape of MapAndBatch node cannot be scalar");
    } else {
      for (int i = 1; i < orig_shape.dim_size(); i++) {
        *new_shape->add_dim() = orig_shape.dim(i);
      }
    }
  }
  NodeDef new_batch_node;
  new_batch_node.set_op(kBatchV2Op);
  graph_utils::SetUniqueGraphNodeName(kBatchOp, graph->graph(),
                                      &new_batch_node);
  new_batch_node.add_input(new_map_node.name());
  new_batch_node.add_input(orig_node.input(num_map_inputs));  
  new_batch_node.add_input(
      orig_node.input(num_map_inputs + 2));  
  graph_utils::CopyShapesAndTypesAttrs(orig_node, &new_batch_node);
  graph->AddNode(std::move(new_map_node));
  NodeDef* graph_batch_node = graph->AddNode(std::move(new_batch_node));
  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(orig_node.name(), graph_batch_node->name()));
  return absl::OkStatus();
}
Status ConvertPrefetch(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  constexpr int buffer_size_index = 1;
  node->add_input(absl::StrCat("^", node->input(buffer_size_index)));
  NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(0, graph);
  node->set_input(buffer_size_index, tmp->name());
  return absl::OkStatus();
}
enum class NondeterminismType { PARALLELISM, ASYNCHRONY };
bool IsDeterministicStatefulOp(NondeterminismType type,
                               const std::string& stateful_op) {
  return type == NondeterminismType::PARALLELISM
             ? IsDeterministicWhenRunInParallel(stateful_op)
             : IsDeterministicWhenRunAsynchronously(stateful_op);
}
bool FunctionNodeMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const NodeDef& node_def,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed);
bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const std::string& function_name,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed,
    absl::flat_hash_set<absl::string_view>* nondeterministic_nodes) {
  if (functions_processed->contains(function_name)) {
    return false;
  }
  functions_processed->insert(function_name);
  const FunctionDef* function_def = library.Find(function_name);
  if (!function_def) {
    VLOG(2) << "Could not look up function " << function_name
            << " in FunctionLibraryDefinition, so rewriting op to be safe";
    return true;
  }
  bool found = false;
  for (const NodeDef& node_def : function_def->node_def()) {
    bool nondeterministic = FunctionNodeMayIntroduceNondeterminism(
        library, node_def, nondeterminism_type, functions_processed);
    if (nondeterministic) {
      if (nondeterministic_nodes) {
        nondeterministic_nodes->insert(node_def.name());
        found = true;
      } else {
        return true;
      }
    }
  }
  return found;
}
bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const std::string& function_name,
    NondeterminismType nondeterminism_type) {
  absl::flat_hash_set<string> functions_processed;
  return FunctionMayIntroduceNondeterminism(library, function_name,
                                            nondeterminism_type,
                                            &functions_processed, nullptr);
}
bool FunctionNodeMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const NodeDef& node_def,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = library.LookUp(node_def.op(), &op_reg_data);
  if (!s.ok()) {
    VLOG(2) << "Could not look up op " << node_def.op()
            << " in FunctionLibraryDefinition, so rewriting op to be safe";
    return true;
  }
  bool is_function_op = op_reg_data->is_function_op;
  bool is_stateful = false;
  if (!is_function_op) {
    const OpDef* op_def;
    s = OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
    if (!s.ok()) {
      VLOG(2) << "Could not look up op " << node_def.op()
              << " in OpRegistry, so rewriting op to be safe";
      return true;
    }
    is_stateful = op_def->is_stateful();
  }
  if (is_stateful && !IsStatefulPartitionedCall((node_def)) &&
      !IsIf(node_def) && !IsWhile(node_def) &&
      !IsDeterministicStatefulOp(nondeterminism_type, node_def.op())) {
    VLOG(2) << "Will rewrite due to op: " << node_def.op();
    return true;
  }
  std::vector<std::string> attr_func_names;
  for (const auto& attr : node_def.attr()) {
    if (attr.second.has_func()) {
      attr_func_names.push_back(attr.second.func().name());
    }
    for (const auto& name_attr_list : attr.second.list().func()) {
      attr_func_names.push_back(name_attr_list.name());
    }
  }
  if (is_function_op) {
    attr_func_names.push_back(node_def.op());
  }
  for (const std::string& inner_function_name : attr_func_names) {
    if (FunctionMayIntroduceNondeterminism(library, inner_function_name,
                                           nondeterminism_type,
                                           functions_processed, nullptr)) {
      return true;
    }
  }
  return false;
}
bool NodeMayIntroduceNondeterminismWhenAsync(
    const FunctionLibraryDefinition& library, const NodeDef& node) {
  const OpDef* op_def;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (s.code() == error::NOT_FOUND) {
    return false;
  } else if (!s.ok()) {
    return true;
  }
  if (data::DatasetOpKernel::IsDatasetOp(*op_def)) {
    std::vector<std::string> attr_func_names;
    for (const auto& attr : node.attr()) {
      if (attr.second.has_func()) {
        attr_func_names.push_back(attr.second.func().name());
      }
      for (const auto& name_attr_list : attr.second.list().func()) {
        attr_func_names.push_back(name_attr_list.name());
      }
    }
    for (const std::string& inner_function_name : attr_func_names) {
      if (FunctionMayIntroduceNondeterminism(library, inner_function_name,
                                             NondeterminismType::ASYNCHRONY)) {
        return true;
      }
    }
  }
  return false;
}
bool GraphMayHaveAsyncNondeterminism(const FunctionLibraryDefinition& library,
                                     const GraphDef& graph) {
  for (const NodeDef& node : graph.node()) {
    if (NodeMayIntroduceNondeterminismWhenAsync(library, node)) {
      return true;
    }
  }
  for (const string& function_name : library.ListFunctionNames()) {
    const FunctionDef* function_def = library.Find(function_name);
    CHECK(function_def);  
    for (const NodeDef& node : function_def->node_def()) {
      if (NodeMayIntroduceNondeterminismWhenAsync(library, node)) {
        return true;
      }
    }
  }
  return false;
}
}  
Status MakeDeterministic::OptimizeAndCollectStats(Cluster* cluster,
                                                  const GrapplerItem& item,
                                                  GraphDef* output,
                                                  OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  absl::flat_hash_set<string> nodes_to_delete;
  bool remove_async_nodes =
      GraphMayHaveAsyncNondeterminism(function_library, item.graph);
  for (const NodeDef& node : item.graph.node()) {
    if (graph_utils::HasSloppyAttr(node.op())) {
      NodeDef* mutable_node = GetMutableNode(node.name(), &graph);
      (*mutable_node->mutable_attr())["sloppy"].set_b(false);
      stats->num_changes++;
    }
    if (graph_utils::HasDeterministicAttr(node.op())) {
      NodeDef* mutable_node = GetMutableNode(node.name(), &graph);
      (*mutable_node->mutable_attr())["deterministic"].set_s("true");
      stats->num_changes++;
    }
    bool rewrite_due_to_async =
        IntroducesAsynchrony(node.op()) && remove_async_nodes;
    absl::flat_hash_set<std::string> functions_processed;
    absl::flat_hash_set<absl::string_view> nondeterministic_nodes;
    bool rewrite_due_to_parallelism =
        IntroducesFunctionParallelism(node.op()) &&
        FunctionMayIntroduceNondeterminism(
            function_library, node.attr().at("f").func().name(),
            NondeterminismType::PARALLELISM, &functions_processed,
            &nondeterministic_nodes);
    if (!rewrite_due_to_async && !rewrite_due_to_parallelism) {
      continue;
    }
    VLOG(1) << "Rewriting node " << node.name() << " (" << node.op()
            << ") because it introduces nondeterminism through "
            << (rewrite_due_to_async ? "asynchrony" : "parallelism");
    bool maybe_can_split =
        !rewrite_due_to_async &&
        (node.op() == kParallelMapOpV2 || IsMapAndBatch(node.op()));
    if (maybe_can_split) {
      Status s = SplitMap(function_library, node.name(), &graph,
                          nondeterministic_nodes);
      if (s.ok()) {
        VLOG(1) << "Split node " << node.name() << " (" << node.op()
                << ") into two map nodes: a nonparallel version and a "
                   "parallel version.";
        nodes_to_delete.insert(node.name());
        continue;
      } else if (s.code() == error::UNIMPLEMENTED) {
        VLOG(1) << "Could not move stateful ops to their own function, so will "
                   "convert node "
                << node.name()
                << " to a nonparallel version instead. Reason: " << s;
      } else {
        return s;
      }
    }
    if (IsPrefetch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertPrefetch(node.name(), &graph));
    } else if (IsMapAndBatch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertMapAndBatch(node.name(), &graph));
      nodes_to_delete.insert(node.name());
    } else if (IsParallelBatch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertBatch(node.name(), &graph));
    } else {
      DCHECK(IsParallelInterleave(node.op()) || IsParallelMap(node.op()));
      TF_RETURN_IF_ERROR(ConvertMapOrInterleave(node.name(), &graph));
    }
    stats->num_changes++;
  }
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(MakeDeterministic, "make_deterministic");
}  
}  