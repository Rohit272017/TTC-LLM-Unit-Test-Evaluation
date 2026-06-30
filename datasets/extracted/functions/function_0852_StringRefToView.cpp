#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/IR/Location.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/OwningOpRef.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/Types.h"  
#include "mlir/IR/ValueRange.h"  
#include "mlir/IR/Verifier.h"  
#include "mlir/IR/Visitors.h"  
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/node_order.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/crash_analysis.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}
namespace tensorflow {
constexpr size_t kNumThreadToConvertSignatures = 10;
constexpr absl::string_view kOutputShapesAttrName = "_output_shapes";
using ::mlir::NamedAttrList;
using ::mlir::TensorType;
using ::mlir::tf_saved_model::AssetOp;
using ::mlir::tf_saved_model::GlobalTensorOp;
using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::kTfSavedModelInitializerInitType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr;
using ::mlir::tf_saved_model::SessionInitializerOp;
using ::tsl::StatusOr;
namespace {
bool IsOutputShapesAttribute(const AttrValue& attr_value,
                             llvm::StringRef attr_name) {
  return attr_name.compare(kOutputShapesAttrName) == 0 &&
         attr_value.value_case() == AttrValue::kList;
}
bool IsResourceOutputShapesAttribute(const AttrValue& attr_value,
                                     llvm::StringRef attr_name) {
  if (attr_name == "_handle_dtypes" || attr_name == "_handle_shapes")
    return attr_value.value_case() == AttrValue::kList;
  return false;
}
void LoadImporterDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialectsImpl(registry, false);
  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
}
absl::StatusOr<std::string> GetDenseTensorNameFromTensorInfo(
    const TensorInfo& tensor_info) {
  TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
      << "Only dense tensor is supported, but got encoding case "
      << tensor_info.encoding_case();
  return tensor_info.name();
}
class NameUniquifier : public OpOrArgNameMapper {
 public:
  explicit NameUniquifier(const FunctionLibraryDefinition& flib)
      : flib_(flib) {}
 private:
  bool IsUnique(llvm::StringRef name) override {
    return !flib_.Contains(std::string(name));
  }
  std::string GetName(OpOrVal op_or_val) override {
    DCHECK(false) << "Unimplemented";
    return "";
  }
  const FunctionLibraryDefinition& flib_;
};
class ImporterBase {
 protected:
  explicit ImporterBase(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier,
      llvm::StringRef function_name_for_debug_info = "")
      : builder_(module.getContext()),
        module_(module),
        context_(module.getContext()),
        tf_name_to_mlir_name_(tf_name_to_mlir_name),
        graph_flib_(flib),
        specs_(specs),
        debug_info_(debug_info),
        function_name_for_debug_info_(function_name_for_debug_info),
        function_name_uniquifier_(function_name_uniquifier),
        error_handler_(module.getContext()) {
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "Importing with: " << specs.str();
      for (auto& it : *tf_name_to_mlir_name) {
        LOG(INFO) << "\t" << it.first << " -> " << it.second;
      }
    }
    stack_traces_ = LoadTracesFromDebugInfo(debug_info_);
  }
  absl::StatusOr<mlir::FunctionType> InferLibFunctionType(
      const FunctionBody& fbody);
  void GetArgsAndRetsFromFunctionBody(
      const FunctionBody& fbody,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes,
      absl::InlinedVector<Node*, 4>* control_ret_nodes);
  Status PrepareConvert(const Graph& graph,
                        std::unique_ptr<GraphDef> graph_def = nullptr);
  Status Convert(llvm::StringRef func_name, mlir::FunctionType func_type,
                 const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
                 const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
                 const absl::InlinedVector<Node*, 4>& control_ret_nodes,
                 llvm::ArrayRef<mlir::NamedAttribute> attrs);
  Status ConvertLibFunction(llvm::StringRef func_name);
  llvm::ArrayRef<Node*> GetOrderedNodes() const { return ordered_nodes_; }
  absl::StatusOr<mlir::Type> InferInputType(const Node& node, int idx,
                                            mlir::Builder builder);
  absl::StatusOr<mlir::Type> InferOutputType(const Node& node, int idx,
                                             mlir::Builder builder);
  Status ConvertDeferredFunctions();
 private:
  using ElementSubtypes = llvm::SmallVector<TensorType, 1>;
  struct DeferredConversionMetaData {
    DeferredConversionMetaData(
        const std::string& function_name,
        const std::vector<mlir::NamedAttribute>& attributes)
        : function_name(function_name), attributes(attributes) {}
    std::string function_name;
    std::vector<mlir::NamedAttribute> attributes;
  };
  Status AddNodesToShapeRefiner(
      std::unordered_map<string, Node*>* node_name_map);
  Status PruneUnreachableNodes(
      std::unordered_map<string, Node*>* node_name_map);
  Status ConvertFeedsToPlaceholders(
      std::unordered_map<string, Node*>* node_name_map);
  absl::StatusOr<TensorType> ConvertDataTypeAndShape(
      DataType dtype, const shape_inference::ShapeHandle& handle,
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);
  absl::StatusOr<TensorType> ConvertElementTypeAndShape(
      mlir::Type element_type, const shape_inference::ShapeHandle& handle,
      shape_inference::InferenceContext* context, mlir::Builder builder);
  absl::StatusOr<ElementSubtypes> ConvertSubtypes(
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);
  absl::StatusOr<mlir::ElementsAttr> ConvertTensorProto(
      const TensorProto& value) {
    return ::tensorflow::ConvertTensorProto(value, &builder_);
  }
  absl::StatusOr<mlir::FlatSymbolRefAttr> ConvertFunctionCallName(
      const std::string& func_name);
  absl::StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value);
  Status ConvertFunctionCallAttribute(const std::string& base_name,
                                      const AttrValue& value,
                                      NamedAttrList* attributes);
  mlir::Operation* CreateOperation(
      const Node& node, llvm::StringRef node_type_name,
      const mlir::OperationState& result,
      const llvm::SmallVectorImpl<mlir::Value>& control_operands);
  Status ConvertNode(const Node& node);
  using BackEdge = BackEdgeHelper::BackEdge;
  Status RemoveBackedges();
  Status AddBackedges();
  Status AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                     int dst_input);
  Status ConvertFunctionArgAndRets(
      mlir::func::FuncOp func, mlir::tf_executor::GraphOp graph_op,
      llvm::ArrayRef<mlir::Type> arg_types,
      const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
      const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
      const absl::InlinedVector<Node*, 4>& control_ret_nodes);
  mlir::Location GetLocation(const Node& node);
  Status EmitErrorWithLocationStr(const Node& node, const Status& error_status);
  absl::StatusOr<std::pair<Node*, bool>> CreatePlaceholderNodeForFeed(
      const TensorShapeProto& shape, DataType dtype, Node* node, int index,
      const std::unordered_map<string, Node*>& node_name_map);
  Status GetInputOutputNodes(
      const std::unordered_map<string, Node*>& node_name_map,
      std::unordered_set<const Node*>* nodes);
  BackEdgeHelper back_edge_helper_;
  absl::flat_hash_map<const Node*, int> back_edge_node_output_;
  absl::flat_hash_map<const Node*, BackEdge> back_edge_dst_inputs_;
  absl::flat_hash_map<mlir::Operation*, mlir::Operation*>
      next_iteration_sink_source_;
  std::unique_ptr<Graph> graph_;
  std::vector<Node*> ordered_nodes_;
  using NodeValueMap = absl::flat_hash_map<int, mlir::Operation*>;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::MLIRContext* context_;
  std::unordered_map<std::string, std::string>* tf_name_to_mlir_name_;
  const FunctionLibraryDefinition& graph_flib_;
  const GraphImportConfig& specs_;
  const GraphDebugInfo& debug_info_;
  StackTracesMap stack_traces_;
  llvm::StringRef function_name_for_debug_info_;
  NodeValueMap node_values_;
  std::unique_ptr<ShapeRefiner> shape_refiner_ = nullptr;
  NameUniquifier* function_name_uniquifier_;
  mlir::StatusScopedDiagnosticHandler error_handler_;
  llvm::DenseSet<mlir::StringAttr> unmodelled_op_names_;
 protected:
  absl::flat_hash_map<TensorId, absl::string_view> remapped_feeds_;
  std::queue<DeferredConversionMetaData> deferred_functions_;
};
bool HasNonPrimaryOutputInUse(const GraphDef& graph_def,
                              const std::string& node) {
  for (const auto& node_def : graph_def.node()) {
    for (const auto& input : node_def.input()) {
      if (absl::StartsWith(input, node + ":") && input != node + ":0") {
        return true;
      }
    }
  }
  return false;
}
Status UpdateLegacyFedInputNode(const GraphDef& graph_def,
                                const GraphImportConfig::InputArrays& inputs,
                                NodeDef* node) {
  const std::string& node_name = node->name();
  auto it = inputs.find(node_name);
  if (it == inputs.end()) return absl::OkStatus();
  if (HasNonPrimaryOutputInUse(graph_def, node_name)) {
    return errors::InvalidArgument(
        "LegacyFedInput node ", node->name(),
        " has non primary output in use and can not be replaced with "
        "Placeholder node");
  }
  DataType dtype = it->second.imported_dtype;
  if (dtype == DT_INVALID) {
    dtype = node->attr().at("output_types").list().type(0);
  }
  *node->mutable_op() = "Placeholder";
  node->clear_attr();
  node->clear_input();
  AddNodeAttr("dtype", dtype, node);
  AddNodeAttr("shape", it->second.shape, node);
  return absl::OkStatus();
}
Status PreprocessGraphDef(const GraphImportConfig* specs, GraphDef* graph_def) {
  for (auto& node_def : *graph_def->mutable_node()) {
    if (specs && specs->convert_legacy_fed_inputs &&
        node_def.op() == "LegacyFedInput") {
      TF_RETURN_IF_ERROR(
          UpdateLegacyFedInputNode(*graph_def, specs->inputs, &node_def));
    }
    const tensorflow::OpRegistrationData* op_reg_data =
        tensorflow::OpRegistry::Global()->LookUp(node_def.op());
    if (!op_reg_data) {
      continue;
    }
    ::tensorflow::AddDefaultsToNodeDef(op_reg_data->op_def, &node_def);
  }
  return absl::OkStatus();
}
using FeedsByNode = absl::flat_hash_map<
    absl::string_view,
    absl::flat_hash_map<int, const std::pair<std::string, ArrayInfo>*>>;
absl::StatusOr<FeedsByNode> GetFeedsByNode(
    const GraphImportConfig::InputArrays& inputs) {
  FeedsByNode feeds_by_node;
  feeds_by_node.reserve(inputs.size());
  for (const auto& input : inputs) {
    TensorId tensor = ParseTensorName(input.first);
    if (tensor.index() < 0)
      return errors::FailedPrecondition(
          "Feed output tensor must be a data output '", tensor.ToString(), "'");
    auto& node = feeds_by_node[tensor.node()];
    if (!node.insert({tensor.index(), &input}).second)
      return errors::FailedPrecondition(
          "Multiple feeds for the same output tensor '", tensor.ToString(),
          "'");
  }
  return feeds_by_node;
}
std::string GetUniqueNodeName(
    absl::string_view node_name, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  std::string new_node_name_base = absl::StrCat(node_name, "_", index);
  int count = 0;
  std::string new_node_name = new_node_name_base;
  while (node_name_map.find(new_node_name) != node_name_map.end()) {
    new_node_name = absl::StrCat(new_node_name_base, "_", count++);
  }
  return new_node_name;
}
Status ImporterBase::ConvertDeferredFunctions() {
  while (!deferred_functions_.empty()) {
    auto conversion_metadata = deferred_functions_.front();
    deferred_functions_.pop();
    const FunctionDef* func_def =
        graph_flib_.Find(conversion_metadata.function_name);
    GraphImportConfig specs;
    specs.enable_shape_inference = specs_.enable_shape_inference;
    specs.unconditionally_use_set_output_shapes =
        specs_.unconditionally_use_set_output_shapes;
    for (const auto& name_and_value : func_def->attr()) {
      if (name_and_value.first == "_input_shapes") {
        auto& list = name_and_value.second.list();
        auto& signature = func_def->signature();
        if (list.shape_size() > 0 &&
            list.shape_size() != signature.input_arg_size()) {
          return errors::FailedPrecondition(
              "Number of input arguments must be equal to the length of "
              "_input_shapes attribute in function '",
              StringRefToView(conversion_metadata.function_name), "'.");
        }
        for (int i = 0, e = signature.input_arg_size(); i < e; i++) {
          auto& input_arg = signature.input_arg(i);
          auto& array_info = specs.inputs[input_arg.name()];
          array_info.imported_dtype = input_arg.type();
          if (list.shape_size() > 0)
            array_info.shape = list.shape(i);
          else
            array_info.shape.set_unknown_rank(true);
        }
      }
    }
    ImporterBase importer(graph_flib_, debug_info_, specs, module_,
                          tf_name_to_mlir_name_, function_name_uniquifier_,
                          conversion_metadata.function_name);
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*func_def, AttrSlice(), &graph_flib_, &fbody));
    TF_RETURN_IF_ERROR(importer.PrepareConvert(*fbody->graph));
    TF_ASSIGN_OR_RETURN(auto func_type, importer.InferLibFunctionType(*fbody));
    absl::InlinedVector<OutputTensor, 4> arg_nodes;
    absl::InlinedVector<OutputTensor, 4> ret_nodes;
    absl::InlinedVector<Node*, 4> control_ret_nodes;
    importer.GetArgsAndRetsFromFunctionBody(*fbody, &arg_nodes, &ret_nodes,
                                            &control_ret_nodes);
    const std::string& mlir_func_name =
        (*tf_name_to_mlir_name_)[conversion_metadata.function_name];
    TF_RETURN_IF_ERROR(importer.Convert(mlir_func_name, func_type, arg_nodes,
                                        ret_nodes, control_ret_nodes,
                                        conversion_metadata.attributes));
    while (!importer.deferred_functions_.empty()) {
      deferred_functions_.push(importer.deferred_functions_.front());
      importer.deferred_functions_.pop();
    }
  }
  return absl::OkStatus();
}
Status ImporterBase::RemoveBackedges() {
  TF_RETURN_IF_ERROR(back_edge_helper_.Remove(graph_.get()));
  VLOG(1) << "Found " << (back_edge_helper_.RemovedEdges().size())
          << " backedges.";
  for (const auto& edge : back_edge_helper_.RemovedEdges()) {
    if (back_edge_node_output_.find(edge.src) != back_edge_node_output_.end() &&
        back_edge_node_output_[edge.src] != edge.src_output) {
      return errors::FailedPrecondition(
          "More than one of the src node outputs are backedges!");
    }
    back_edge_node_output_[edge.src] = edge.src_output;
    DCHECK(!back_edge_dst_inputs_.contains(edge.dst));
    back_edge_dst_inputs_[edge.dst] = edge;
  }
  ordered_nodes_.clear();
  TopologicalOrdering(
      *graph_, [&](Node* n) { ordered_nodes_.push_back(n); }, GroupByDevice());
  return absl::OkStatus();
}
Status CopyStackTraces(const Graph& from, Graph* to) {
  std::unordered_map<string, Node*> node_map = from.BuildNodeNameIndex();
  for (Node* node : to->nodes()) {
    if (const Node* old_node = node_map[node->name()]) {
      if (const std::shared_ptr<AbstractStackTrace>& stack =
              old_node->GetStackTrace()) {
        DVLOG(2) << "Stack for " << node->name() << " "
                 << old_node->GetStackTrace()->ToString(
                        AbstractStackTrace::TracePrintingOptions());
        node->SetStackTrace(stack);
      } else {
        DVLOG(1) << "No stack for " << node->name() << " (" << node
                 << ") in Graph " << &from;
      }
    } else {
      DVLOG(1) << "No stack for " << node->name() << " (" << node
               << ") in Graph " << &from;
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<std::pair<Node*, bool>>
ImporterBase::CreatePlaceholderNodeForFeed(
    const TensorShapeProto& shape, DataType dtype, Node* node, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  DCHECK_LT(index, node->num_outputs());
  const bool update_inplace = node->num_outputs() == 1 && index == 0;
  std::string new_node_name =
      update_inplace ? node->name()
                     : GetUniqueNodeName(node->name(), index, node_name_map);
  Node* placeholder_node;
  NodeBuilder builder(new_node_name, "Placeholder");
  builder.Attr("shape", shape);
  builder.Attr("dtype", dtype);
  TF_RETURN_IF_ERROR(builder.Finalize(graph_.get(), &placeholder_node));
  std::vector<const Edge*> data_edges;
  std::vector<const Edge*> control_edges;
  for (const tensorflow::Edge* edge : node->out_edges()) {
    if (edge->src_output() == index) {
      data_edges.push_back(edge);
    } else if (update_inplace && edge->IsControlEdge()) {
      control_edges.push_back(edge);
    }
  }
  for (const auto* edge : data_edges) {
    TF_RETURN_IF_ERROR(graph_->UpdateEdge(placeholder_node, 0, edge->dst(),
                                          edge->dst_input()));
  }
  for (const auto* edge : control_edges) {
    graph_->AddControlEdge(placeholder_node, edge->dst());
    graph_->RemoveControlEdge(edge);
  }
  if (update_inplace) {
    graph_->RemoveNode(node);
  }
  return std::pair<Node*, bool>(placeholder_node, update_inplace);
}
Status ImporterBase::GetInputOutputNodes(
    const std::unordered_map<string, Node*>& node_name_map,
    std::unordered_set<const Node*>* nodes) {
  auto add_node = [&](absl::string_view name) {
    auto it = node_name_map.find(std::string(name));
    if (it == node_name_map.end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", name));
    }
    nodes->insert(it->second);
    return absl::OkStatus();
  };
  for (const auto& input : specs_.inputs) {
    TensorId tensor = ParseTensorName(input.first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }
  for (const auto& output : specs_.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }
  for (const auto& control_output : specs_.control_outputs)
    TF_RETURN_IF_ERROR(add_node(control_output));
  return absl::OkStatus();
}
Status ImporterBase::AddNodesToShapeRefiner(
    std::unordered_map<string, Node*>* node_name_map) {
  shape_refiner_ =
      std::make_unique<ShapeRefiner>(graph_->versions(), graph_->op_registry());
  shape_refiner_->set_require_shape_inference_fns(false);
  shape_refiner_->set_function_library_for_shape_inference(&graph_flib_);
  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));
  for (Node* node : ordered_nodes_) {
    bool node_added_to_shape_refiner = false;
    auto it = feeds_by_node.find(node->name());
    if (it != feeds_by_node.end()) {
      auto op_name = node->op_def().name();
      if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
          op_name != FunctionLibraryDefinition::kArgOp) {
        for (const auto& output_tensor : it->second) {
          const int index = output_tensor.first;
          const ArrayInfo& array_info = output_tensor.second->second;
          DataType dtype = array_info.imported_dtype;
          if (dtype == DT_INVALID) {
            dtype = node->output_type(index);
          }
          TF_ASSIGN_OR_RETURN(
              auto placeholder_node_and_removed,
              CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                           *node_name_map));
          Node* placeholder_node = placeholder_node_and_removed.first;
          if (placeholder_node_and_removed.second) {
            node = placeholder_node;
            node_added_to_shape_refiner = true;
          }
          remapped_feeds_[{it->first, index}] = placeholder_node->name();
          (*node_name_map)[placeholder_node->name()] = placeholder_node;
          Status status = shape_refiner_->AddNode(placeholder_node);
          if (!status.ok()) {
            return EmitErrorWithLocationStr(*placeholder_node, status);
          }
        }
      } else {
        auto index_it = it->second.find(0);
        if (index_it == it->second.end()) {
          return errors::FailedPrecondition(
              "Missing feed output tensor at index 0 for node '", node->name(),
              "'");
        }
        node->AddAttr("shape", index_it->second->second.shape);
        DataType dtype = index_it->second->second.imported_dtype;
        if (dtype == DT_INVALID) {
          dtype = node->output_type(0);
        }
        node->AddAttr("dtype", dtype);
      }
    }
    if (!node_added_to_shape_refiner) {
      Status status = shape_refiner_->AddNode(node);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
    }
    auto set_shape_from_list_attr = [&](const AttrValue* attr) {
      auto& list = attr->list();
      if (list.shape_size() != node->num_outputs()) {
        LOG(WARNING) << "Node '" << node->name() << "' has "
                     << node->num_outputs() << " outputs but the "
                     << kOutputShapesAttrName
                     << " attribute specifies shapes for " << list.shape_size()
                     << " outputs";
        return absl::OkStatus();
      }
      for (const auto& shape : llvm::enumerate(list.shape())) {
        auto* node_context = shape_refiner_->GetContext(node);
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(shape.value(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(shape.index(), handle);
      }
      return absl::OkStatus();
    };
    if (StringPiece(node->type_string()) == FunctionLibraryDefinition::kArgOp) {
      auto* node_context = shape_refiner_->GetContext(node);
      DCHECK(node_context != nullptr);
      if (const AttrValue* attr = node->attrs().Find("shape")) {
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(attr->shape(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(0, handle);
      } else if (const AttrValue* attr =
                     node->attrs().Find(kOutputShapesAttrName)) {
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
      } else {
        node_context->set_output(0, node_context->UnknownShape());
      }
    }
    if (specs_.unconditionally_use_set_output_shapes ||
        node->op_def().name() == "ReadVariableOp") {
      if (const AttrValue* attr = node->attrs().Find(kOutputShapesAttrName))
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
    }
  }
  FixupSourceAndSinkEdges(graph_.get());
  if (specs_.prune_unused_nodes) {
    std::unordered_set<const Node*> prune_start;
    TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));
    if (!prune_start.empty()) {
      if (PruneForReverseReachability(graph_.get(), prune_start)) {
        VLOG(1) << "Pruned unused nodes in graphdef";
      } else {
        VLOG(1) << "No unused nodes in graphdef to prune";
      }
    } else {
      VLOG(1) << "No output nodes specified, skipping pruning";
    }
  } else {
    VLOG(1) << "Pruning unused nodes in graphdef is disabled";
  }
  ordered_nodes_.clear();
  TopologicalOrdering(
      *graph_, [&](Node* n) { ordered_nodes_.push_back(n); }, GroupByDevice());
  VLOG(1) << "Inferring graph shapes to fixpoint";
  auto same_inferred_shape = [](shape_inference::InferenceContext* c,
                                shape_inference::ShapeHandle s0,
                                shape_inference::ShapeHandle s1) -> bool {
    if (s0.SameHandle(s1) || (!c->RankKnown(s0) && !c->RankKnown(s1))) {
      return true;
    }
    if (c->Rank(s0) != c->Rank(s1)) {
      return false;
    }
    for (int i = 0; i < c->Rank(s0); ++i) {
      if (!c->Dim(s0, i).SameHandle(c->Dim(s1, i))) {
        int64_t val0 = c->Value(c->Dim(s0, i));
        int64_t val1 = c->Value(c->Dim(s1, i));
        if (val0 >= 0 && val1 >= 0 && val0 != val1) return false;
      }
    }
    return true;
  };
  bool changed = true;
  int i = 0;
  const int kMaxIterationCount = 2;
  while (changed && i != kMaxIterationCount) {
    changed = false;
    for (const Node* node : ordered_nodes_) {
      auto* shape_context = shape_refiner_->GetContext(node);
      DCHECK(shape_context != nullptr);
      absl::InlinedVector<shape_inference::ShapeHandle, 4> existing;
      existing.reserve(shape_context->num_outputs());
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        existing.push_back(shape_context->output(o));
      }
      bool inferred = false;
      shape_inference::ShapeHandle handle;
      Status status =
          shape_refiner_->UpdateNode(node, false, &inferred);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        if (!same_inferred_shape(shape_context, shape_context->output(o),
                                 existing[o])) {
          changed = true;
          break;
        }
      }
    }
    ++i;
  }
  if (i >= kMaxIterationCount) {
    LOG(WARNING) << "Graph shapes did not converge to a fixpoint within "
                 << kMaxIterationCount
                 << " iterations. Graph shapes may be conservative.";
  }
  VLOG(1) << "Graph shapes were inferred with " << (i - 1)
          << " extra rounds of analysis to reach a fixpoint.";
  return absl::OkStatus();
}
absl::StatusOr<mlir::Type> ImporterBase::InferInputType(const Node& node,
                                                        int idx,
                                                        mlir::Builder builder) {
  if (specs_.enable_shape_inference) {
    auto* context = shape_refiner_->GetContext(&node);
    DataType dtype = node.input_type(idx);
    return ConvertDataTypeAndShape(dtype, context->input(idx),
                                   context->input_handle_shapes_and_types(idx),
                                   context, builder);
  }
  DataType dtype = node.properties()->input_types[idx];
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
  return mlir::UnrankedTensorType::get(element_type);
}
absl::StatusOr<mlir::Type> ImporterBase::InferOutputType(
    const Node& node, int idx, mlir::Builder builder) {
  DataType dtype = node.properties()->output_types[idx];
  auto shape_ic =
      [&](shape_inference::InferenceContext* c) -> absl::StatusOr<mlir::Type> {
    if (specs_.unconditionally_use_set_output_shapes) {
      if (const AttrValue* attr = node.attrs().Find(kOutputShapesAttrName)) {
        auto& list = attr->list();
        if (list.shape_size() > idx) {
          const TensorShapeProto& p = list.shape()[idx];
          shape_inference::ShapeHandle h;
          Status s = c->MakeShapeFromShapeProto(p, &h);
          if (!s.ok())
            return errors::InvalidArgument(
                "Node '", node.name(), " has an invalid ",
                kOutputShapesAttrName, " attribute (shape #", idx, " error:'",
                s.message(), "')");
          c->set_output(idx, h);
        }
      }
    }
    return ConvertDataTypeAndShape(dtype, c->output(idx),
                                   c->output_handle_shapes_and_types(idx), c,
                                   builder);
  };
  if (specs_.enable_shape_inference) {
    shape_inference::InferenceContext* shape_context =
        shape_refiner_->GetContext(&node);
    return shape_ic(shape_context);
  }
  if (node.type_string() == "TensorListReserve" ||
      node.type_string() == "EmptyTensorList") {
    mlir::Type etype;
    if (auto element_dtype = node.attrs().Find("element_dtype")) {
      TF_RETURN_IF_ERROR(
          ConvertDataType(element_dtype->type(), builder, &etype));
    }
    return GetTypeFromTFTensorShape(
        {}, mlir::TF::VariantType::get({mlir::UnrankedTensorType::get(etype)},
                                       etype.getContext()));
  }
  if (node.IsWhileNode()) {
    auto* output_shapes = node.attrs().Find("output_shapes");
    auto* element_types = node.attrs().Find("T");
    if (output_shapes && !output_shapes->list().shape().empty()) {
      const auto& output_shape = output_shapes->list().shape(idx);
      const auto& element_type = element_types->list().type(idx);
      return ConvertToMlirTensorType(output_shape, element_type, &builder);
    }
  }
  auto type_from_array_attr = [&node, &idx, &builder](
                                  absl::string_view output_shape_attr,
                                  absl::string_view element_type_attr) {
    auto* output_shapes = node.attrs().Find(output_shape_attr);
    auto* element_types = node.attrs().Find(element_type_attr);
    const auto& output_shape = output_shapes->list().shape(idx);
    const auto& element_type = element_types->list().type(idx);
    return ConvertToMlirTensorType(output_shape, element_type, &builder);
  };
  if (node.type_string() == "IteratorGetNext" ||
      node.type_string() == "IteratorGetNextSync" ||
      node.type_string() == "MultiDeviceIteratorGetNextFromShard")
    return type_from_array_attr("output_shapes", "output_types");
  if (node.type_string() == "InfeedDequeueTuple")
    return type_from_array_attr("shapes", "dtypes");
  if (node.type_string() == "InfeedDequeue") {
    assert(idx == 0);
    const auto& output_shape = node.attrs().Find("shape")->shape();
    const auto& element_type = node.attrs().Find("dtype")->type();
    return ConvertToMlirTensorType(output_shape, element_type, &builder);
  }
  auto default_type = [&]() -> absl::StatusOr<mlir::Type> {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
    if (specs_.unconditionally_use_set_output_shapes) {
      if (const AttrValue* attr = node.attrs().Find(kOutputShapesAttrName)) {
        auto& list = attr->list();
        if (list.shape_size() > idx) {
          llvm::SmallVector<int64_t, 4> shape;
          const TensorShapeProto& shape_proto = list.shape()[idx];
          if (shape_proto.unknown_rank())
            return mlir::UnrankedTensorType::get(element_type);
          TF_RETURN_IF_ERROR(ConvertToMlirShape(shape_proto, &shape));
          return GetTypeFromTFTensorShape(shape, element_type);
        }
      }
    }
    return mlir::UnrankedTensorType::get(element_type);
  };
  if (node.num_inputs() > 0) return default_type();
  if (node.IsArg()) {
    if (dtype == DT_RESOURCE) {
      const AttrValue* dtype_attr = node.attrs().Find("_handle_dtypes");
      const AttrValue* shape_attr = node.attrs().Find("_handle_shapes");
      if (dtype_attr && shape_attr) {
        if (dtype_attr->list().type().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_dtypes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        if (shape_attr->list().shape().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_shapes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        DataType dtype = dtype_attr->list().type(0);
        const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
        TF_ASSIGN_OR_RETURN(
            auto etype, ConvertToMlirTensorType(shape_proto, dtype, &builder));
        return mlir::UnrankedTensorType::get(mlir::TF::ResourceType::get(
            {mlir::cast<TensorType>(etype)}, builder.getContext()));
      } else {
        return mlir::UnrankedTensorType::get(
            mlir::TF::ResourceType::get(builder.getContext()));
      }
    } else if (auto shape = node.attrs().Find("_output_shapes")) {
      if (shape->has_list() && shape->list().shape_size() == 1) {
        return ConvertToMlirTensorType(shape->list().shape().at(0), dtype,
                                       &builder);
      }
    }
  }
  const tensorflow::OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(
      graph_->op_registry()->LookUp(node.type_string(), &op_reg_data));
  if (!op_reg_data) {
    DVLOG(1) << "Skipping inference for unregistered op " << node.type_string();
    return default_type();
  }
  if (op_reg_data->shape_inference_fn == nullptr) {
    DVLOG(1) << "Skipping inference for op without shape function "
             << node.type_string();
    return default_type();
  }
  shape_inference::InferenceContext c(graph_->versions().producer(),
                                      node.attrs(), op_reg_data->op_def,
                                      std::vector<PartialTensorShape>{}, {},
                                      {}, {});
  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));
  return shape_ic(&c);
}
absl::StatusOr<TensorType> ImporterBase::ConvertDataTypeAndShape(
    DataType dtype, const shape_inference::ShapeHandle& handle,
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  TF_ASSIGN_OR_RETURN(auto subtypes,
                      ConvertSubtypes(handle_subtypes, context, builder));
  mlir::Type element_type;
  if (dtype == DT_VARIANT)
    element_type = mlir::TF::VariantType::get(subtypes, context_);
  else if (dtype == DT_RESOURCE)
    element_type = mlir::TF::ResourceType::get(subtypes, context_);
  else
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(dtype, builder, &element_type));
  return ConvertElementTypeAndShape(element_type, handle, context, builder);
}
absl::StatusOr<TensorType> ImporterBase::ConvertElementTypeAndShape(
    mlir::Type element_type, const shape_inference::ShapeHandle& handle,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  if (!context->RankKnown(handle)) {
    return mlir::UnrankedTensorType::get(element_type);
  }
  const int64_t kUnknownDim = -1;
  absl::InlinedVector<int64_t, 4> dimensions;
  int32_t rank = context->Rank(handle);
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto dim_handle = context->Dim(handle, i);
    if (!context->ValueKnown(dim_handle))
      dimensions.push_back(kUnknownDim);
    else
      dimensions.push_back(context->Value(dim_handle));
  }
  return GetTypeFromTFTensorShape(
      llvm::ArrayRef(dimensions.begin(), dimensions.end()), element_type);
}
absl::StatusOr<ImporterBase::ElementSubtypes> ImporterBase::ConvertSubtypes(
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  ElementSubtypes subtypes;
  if (!handle_subtypes) return subtypes;
  subtypes.reserve(handle_subtypes->size());
  for (const auto& subtype : *handle_subtypes) {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(subtype.dtype, builder, &element_type));
    TF_ASSIGN_OR_RETURN(TensorType type,
                        ConvertElementTypeAndShape(element_type, subtype.shape,
                                                   context, builder));
    subtypes.push_back(type);
  }
  return subtypes;
}
Status ImporterBase::ConvertFunctionCallAttribute(const std::string& base_name,
                                                  const AttrValue& value,
                                                  NamedAttrList* attributes) {
  TF_ASSIGN_OR_RETURN(auto func_attr,
                      ConvertFunctionCallName(value.func().name()));
  if (!func_attr) return absl::OkStatus();
  attributes->push_back(builder_.getNamedAttr(base_name, func_attr));
  for (const auto& it : value.func().attr()) {
    auto name = absl::StrCat(base_name, ".", it.first);
    TF_ASSIGN_OR_RETURN(auto value, ConvertAttributeValue(it.second));
    attributes->push_back(builder_.getNamedAttr(name, value));
  }
  return absl::OkStatus();
}
absl::StatusOr<mlir::FlatSymbolRefAttr> ImporterBase::ConvertFunctionCallName(
    const std::string& func_name) {
  if (func_name.empty()) return mlir::FlatSymbolRefAttr();
  TF_RETURN_IF_ERROR(ConvertLibFunction(func_name));
  auto mlir_func_name = (*tf_name_to_mlir_name_)[func_name];
  return mlir::SymbolRefAttr::get(builder_.getContext(), mlir_func_name);
}
absl::StatusOr<mlir::Attribute> ImporterBase::ConvertAttributeValue(
    const AttrValue& value) {
  switch (value.value_case()) {
    case AttrValue::kFunc: {
      NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        TF_ASSIGN_OR_RETURN(
            auto attr, ImporterBase::ConvertAttributeValue(func_attr.second));
        attrs.push_back(builder_.getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder_.getDictionaryAttr(attrs);
      return mlir::TF::FuncAttr::get(context_, value.func().name(), func_attrs);
    }
    case AttrValue::kList: {
      if (!value.list().func().empty()) {
        absl::InlinedVector<mlir::Attribute, 8> attrs;
        for (const auto& item : value.list().func()) {
          TF_ASSIGN_OR_RETURN(auto attr, ConvertFunctionCallName(item.name()));
          if (item.attr_size() != 0)
            return errors::Unimplemented(
                "func attributes with non-zero attr.size()");
          if (attr) attrs.push_back(attr);
        }
        return builder_.getArrayAttr(
            llvm::ArrayRef(attrs.begin(), attrs.end()));
      }
      return ConvertNonFuncAttributeValue(value, &builder_);
    }
    default:
      return ConvertNonFuncAttributeValue(value, &builder_);
  }
}
void ImporterBase::GetArgsAndRetsFromFunctionBody(
    const FunctionBody& fbody, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  arg_nodes->reserve(fbody.arg_nodes.size());
  ret_nodes->reserve(fbody.ret_nodes.size());
  for (auto arg : fbody.arg_nodes) {
    arg_nodes->emplace_back(arg, 0);
  }
  for (auto ret : fbody.ret_nodes) {
    ret_nodes->emplace_back(ret, 0);
  }
  *control_ret_nodes = fbody.control_ret_nodes;
}
Status ImporterBase::ConvertLibFunction(llvm::StringRef func_name) {
  if (tf_name_to_mlir_name_->find(std::string(func_name)) !=
      tf_name_to_mlir_name_->end())
    return absl::OkStatus();
  std::string mlir_func_name(
      function_name_uniquifier_->GetUniqueName(func_name));
  (*tf_name_to_mlir_name_)[std::string(func_name)] = mlir_func_name;
  const auto& func_lib = graph_flib_;
  const auto* func_def = func_lib.Find(std::string(func_name));
  if (func_def == nullptr) {
    return errors::FailedPrecondition(
        absl::StrCat("Failed to find function '", StringRefToView(func_name),
                     "'. The imported TensorFlow GraphDef is ill-formed."));
  }
  std::vector<mlir::NamedAttribute> attributes;
  attributes.reserve(func_def->attr_size());
  for (const auto& name_and_value : func_def->attr()) {
    TF_ASSIGN_OR_RETURN(auto attr,
                        ConvertAttributeValue(name_and_value.second));
    std::string attr_name =
        mangling_util::MangleAttributeName(name_and_value.first);
    attributes.push_back(builder_.getNamedAttr(attr_name, attr));
  }
  if (func_def->signature().is_stateful()) {
    auto stateful_str = mlir::TF::TensorFlowDialect::GetStatefulAttrName();
    attributes.push_back(
        builder_.getNamedAttr(stateful_str, builder_.getUnitAttr()));
  }
  auto grad_func_name = func_lib.FindGradient(std::string(func_name));
  if (!grad_func_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(grad_func_name));
    auto mlir_grad_func_name = (*tf_name_to_mlir_name_)[grad_func_name];
    auto gradient_attr =
        mlir::SymbolRefAttr::get(builder_.getContext(), mlir_grad_func_name);
    auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
    attributes.push_back(builder_.getNamedAttr(grad_string, gradient_attr));
  }
  deferred_functions_.emplace(func_name.str(), attributes);
  return absl::OkStatus();
}
Status ImporterBase::PruneUnreachableNodes(
    std::unordered_map<string, Node*>* node_name_map) {
  std::unordered_set<const Node*> prune_start;
  TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));
  if (!prune_start.empty()) {
    if (PruneForReverseReachability(graph_.get(), prune_start)) {
      VLOG(1) << "Pruned unused nodes in graphdef";
    } else {
      VLOG(1) << "No unused nodes in graphdef to prune";
    }
  } else {
    VLOG(1) << "No output nodes specified, skipping pruning";
  }
  return absl::OkStatus();
}
Status ImporterBase::ConvertFeedsToPlaceholders(
    std::unordered_map<string, Node*>* node_name_map) {
  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));
  for (const auto& it : feeds_by_node) {
    TensorId tensor = ParseTensorName(it.first);
    auto jt = node_name_map->find(std::string(tensor.node()));
    if (jt == node_name_map->end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", tensor.node()));
    }
    Node* node = jt->second;
    auto op_name = node->op_def().name();
    if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
        op_name != FunctionLibraryDefinition::kArgOp) {
      for (const auto& output_tensor : it.second) {
        const int index = output_tensor.first;
        const ArrayInfo& array_info = output_tensor.second->second;
        DataType dtype = array_info.imported_dtype;
        if (dtype == DT_INVALID) {
          dtype = node->output_type(index);
        }
        TF_ASSIGN_OR_RETURN(
            auto placeholder_node_and_removed,
            CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                         *node_name_map));
        Node* placeholder_node = placeholder_node_and_removed.first;
        if (placeholder_node->in_edges().empty()) {
          graph_->AddControlEdge(graph_->source_node(), placeholder_node,
                                 true );
        }
        if (placeholder_node->out_edges().empty()) {
          graph_->AddControlEdge(placeholder_node, graph_->sink_node(),
                                 true );
        }
        remapped_feeds_[{it.first, index}] = placeholder_node->name();
        (*node_name_map)[placeholder_node->name()] = placeholder_node;
      }
    }
  }
  return absl::OkStatus();
}
Status ImporterBase::PrepareConvert(const Graph& graph,
                                    std::unique_ptr<GraphDef> graph_def) {
  if (graph_def == nullptr) {
    graph_def = std::make_unique<GraphDef>();
    graph.ToGraphDef(graph_def.get());
  }
  graph_ = std::make_unique<Graph>(graph.flib_def());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.add_default_attributes = true;
  TF_RETURN_IF_ERROR(::tensorflow::ConvertGraphDefToGraph(
      opts, std::move(*graph_def), graph_.get()));
  TF_RETURN_IF_ERROR(RemoveBackedges());
  TF_RETURN_IF_ERROR(CopyStackTraces(graph, graph_.get()));
  auto node_name_map = graph_->BuildNodeNameIndex();
  if (specs_.enable_shape_inference) {
    TF_RETURN_IF_ERROR(AddNodesToShapeRefiner(&node_name_map));
  } else {
    TF_RETURN_IF_ERROR(ConvertFeedsToPlaceholders(&node_name_map));
  }
  if (specs_.prune_unused_nodes) {
    TF_RETURN_IF_ERROR(PruneUnreachableNodes(&node_name_map));
  }
  if (!specs_.enable_shape_inference) {
    ordered_nodes_.clear();
    TopologicalOrdering(
        *graph_, [&](Node* n) { ordered_nodes_.push_back(n); },
        GroupByDevice());
  }
  return absl::OkStatus();
}
Status ImporterBase::Convert(
    llvm::StringRef func_name, mlir::FunctionType func_type,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes,
    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  auto function = mlir::func::FuncOp::create(mlir::UnknownLoc::get(context_),
                                             func_name, func_type, attrs);
  module_.push_back(function);
  function.addEntryBlock();
  builder_ = mlir::OpBuilder(function.getBody());
  auto graph = builder_.create<mlir::tf_executor::GraphOp>(
      function.getLoc(), func_type.getResults());
  builder_.createBlock(&graph.getBody());
  for (const Node* node : ordered_nodes_) {
    TF_RETURN_IF_ERROR(ConvertNode(*node));
  }
  TF_RETURN_IF_ERROR(AddBackedges());
  TF_RETURN_IF_ERROR(ConvertFunctionArgAndRets(function, graph,
                                               func_type.getInputs(), arg_nodes,
                                               ret_nodes, control_ret_nodes));
  if (!specs_.enable_shape_inference) {
    auto fetch = graph.GetFetch();
    bool all_equal = true;
    for (auto it :
         llvm::zip_first(graph.getResults(), fetch.getOperandTypes())) {
      auto rt = std::get<1>(it);
      if (rt == std::get<0>(it).getType()) continue;
      std::get<0>(it).setType(rt);
      all_equal = false;
    }
    if (!all_equal) {
      function.setType(mlir::FunctionType::get(function.getContext(),
                                               func_type.getInputs(),
                                               graph.getResultTypes()));
    }
  }
  return absl::OkStatus();
}
Status ImporterBase::ConvertFunctionArgAndRets(
    mlir::func::FuncOp func, mlir::tf_executor::GraphOp graph_op,
    llvm::ArrayRef<mlir::Type> arg_types,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes) {
  llvm::SmallVector<mlir::NamedAttrList, 4> arg_attrs;
  arg_attrs.resize(func.getNumArguments());
  llvm::SmallVector<mlir::NamedAttrList, 4> ret_attrs;
  ret_attrs.resize(func.getNumResults());
  auto set_attributes_on_func = [&](Node* node, int64_t index, bool is_arg) {
    for (const auto& node_attr : node->attrs()) {
      const auto& key = node_attr.first;
      if (key.empty() || key[0] != '_') continue;
      if (IsOutputShapesAttribute(node_attr.second, key) ||
          IsResourceOutputShapesAttribute(node_attr.second, key))
        continue;
      TF_ASSIGN_OR_RETURN(auto converted_attr,
                          ConvertAttributeValue(node_attr.second));
      std::string dialect_attribute = "tf." + key;
      if (is_arg) {
        arg_attrs[index].set(dialect_attribute, converted_attr);
      } else {
        func.setResultAttr(index, dialect_attribute, converted_attr);
        ret_attrs[index].set(dialect_attribute, converted_attr);
      }
    }
    return absl::OkStatus();
  };
  auto* bb = &func.front();
  llvm::SmallDenseMap<std::pair<Node*, int>, mlir::Value, 4>
      arg_nodes_to_values;
  for (int i = 0, e = arg_types.size(); i < e; ++i) {
    auto& arg_node = arg_nodes[i];
    mlir::Operation* island = node_values_.find(arg_node.node->id())->second;
    auto bb_arg = bb->getArgument(i);
    mlir::Value arg_def = bb_arg;
    if (island->getNumResults() != 2)
      return errors::InvalidArgument(
          "Only feed output tensors of single output nodes are supported");
    arg_nodes_to_values.try_emplace({arg_node.node, arg_node.index}, arg_def);
    island->getResult(0).replaceAllUsesWith(arg_def);
    auto control_uses = island->getResult(1).getUses();
    for (auto& control_use : llvm::make_early_inc_range(control_uses))
      control_use.getOwner()->eraseOperand(control_use.getOperandNumber());
    if (!arg_node.node->requested_device().empty())
      arg_attrs[i].set("tf.device", builder_.getStringAttr(
                                        arg_node.node->requested_device()));
    if (arg_node.node->IsArg()) {
      TF_RETURN_IF_ERROR(
          set_attributes_on_func(arg_node.node, i, true));
    }
    island->dropAllReferences();
    island->erase();
  }
  llvm::SmallVector<mlir::Value, 8> inst_to_return;
  for (const auto& ret_and_idx : llvm::enumerate(ret_nodes)) {
    const auto& ret = ret_and_idx.value();
    auto* inst = node_values_[ret.node->id()];
    if (ret.node->IsRetval()) {
      if (!ret.node->requested_device().empty())
        ret_attrs[ret_and_idx.index()].set(
            "tf.device", builder_.getStringAttr(ret.node->requested_device()));
      TF_RETURN_IF_ERROR(set_attributes_on_func(ret.node, ret_and_idx.index(),
                                                false));
      auto island_op = llvm::cast<mlir::tf_executor::IslandOp>(inst);
      mlir::Operation* inner_op = &island_op.GetBody().front();
      if (inner_op->getNumOperands() != 1)
        return errors::Unimplemented("Return node with multiple inputs.");
      inst_to_return.push_back(inner_op->getOperand(0));
      inst->dropAllReferences();
      inst->erase();
    } else {
      auto it = arg_nodes_to_values.find({ret.node, ret.index});
      if (it != arg_nodes_to_values.end())
        inst_to_return.push_back(it->second);
      else
        inst_to_return.push_back(inst->getResult(ret.index));
    }
  }
  for (Node* control_ret : control_ret_nodes) {
    auto* inst = node_values_[control_ret->id()];
    inst_to_return.push_back(*std::prev(inst->result_end()));
  }
  builder_.setInsertionPointToEnd(&graph_op.getBody().front());
  builder_.create<mlir::tf_executor::FetchOp>(graph_op.getLoc(),
                                              inst_to_return);
  builder_.setInsertionPointToEnd(bb);
  builder_.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(context_),
                                        graph_op.getResults());
  func.setAllArgAttrs(
      llvm::to_vector<4>(llvm::map_range(arg_attrs, [&](NamedAttrList& list) {
        return list.getDictionary(context_);
      })));
  func.setAllResultAttrs(
      llvm::to_vector<4>(llvm::map_range(ret_attrs, [&](NamedAttrList& list) {
        return list.getDictionary(context_);
      })));
  return absl::OkStatus();
}
mlir::Location ImporterBase::GetLocation(const Node& node) {
  DVLOG(1) << "Getting location for " << node.name() << " " << &node;
  auto create_location = [&](llvm::StringRef name,
                             llvm::StringRef function_name) -> mlir::Location {
    std::string debug_info_key = (name + "@" + function_name).str();
    std::string name_for_name_loc =
        function_name.empty() ? name.str() : debug_info_key;
    auto name_loc_id = mlir::StringAttr::get(context_, name_for_name_loc);
    std::shared_ptr<AbstractStackTrace> stack_trace = node.GetStackTrace();
    if (stack_trace != nullptr) {
    } else if (stack_traces_.contains(name_for_name_loc)) {
      stack_trace = stack_traces_.at(name_for_name_loc);
    } else if (stack_traces_.contains(debug_info_key)) {
      stack_trace = stack_traces_.at(debug_info_key);
    } else {
      DVLOG(1) << "No stack trace for " << node.name();
    }
    llvm::SmallVector<mlir::Location, 4> locations;
    if (stack_trace != nullptr) {
      DVLOG(1) << "Stack available for " << node.name();
      for (const StackFrame& frame : stack_trace->ToUncachedFrames()) {
        auto file_name = mlir::StringAttr::get(context_, frame.file_name);
        auto file_line_loc =
            mlir::FileLineColLoc::get(file_name, frame.line_number, 1);
        locations.push_back(file_line_loc);
      }
    }
    if (locations.empty()) return mlir::NameLoc::get(name_loc_id);
    mlir::Location node_name_loc =
        mlir::NameLoc::get(name_loc_id, locations.front());
    auto callsite_locs = llvm::ArrayRef(locations).drop_front();
    return callsite_locs.empty()
               ? node_name_loc
               : mlir::CallSiteLoc::get(node_name_loc, callsite_locs);
  };
  auto create_op_type_and_name_locations = [&]() {
    return mlir::FusedLoc::get(
        context_,
        {mlir::NameLoc::get(
             mlir::StringAttr::get(context_, node.type_string() + ":")),
         create_location(node.name(), function_name_for_debug_info_)});
  };
  if (node.type_string() == "NextIteration") {
    return create_op_type_and_name_locations();
  }
  const auto& node_def = node.def();
  auto original_nodes =
      node_def.experimental_debug_info().original_node_names();
  auto original_funcs =
      node_def.experimental_debug_info().original_func_names();
  if (original_nodes.empty()) {
    return create_op_type_and_name_locations();
  } else {
    llvm::SmallVector<mlir::Location, 4> node_locations;
    node_locations.reserve(original_nodes.size() + 2);
    node_locations.push_back(mlir::NameLoc::get(
        mlir::StringAttr::get(context_, node.type_string() + ":")));
    for (int i = 0, e = original_nodes.size(); i != e; ++i) {
      const auto& node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    node_locations.push_back(
        create_location(node.name(), function_name_for_debug_info_));
    return mlir::FusedLoc::get(context_, node_locations);
  }
}
Status ImporterBase::EmitErrorWithLocationStr(const Node& node,
                                              const Status& error_status) {
  const mlir::Location location = GetLocation(node);
  mlir::emitError(location);
  return error_handler_.Combine(error_status);
}
mlir::Operation* ImporterBase::CreateOperation(
    const Node& node, llvm::StringRef node_type_name,
    const mlir::OperationState& result,
    const llvm::SmallVectorImpl<mlir::Value>& control_operands) {
  mlir::SmallVector<mlir::Type, 4> types(result.types);
  types.push_back(mlir::tf_executor::ControlType::get(builder_.getContext()));
  mlir::SmallVector<mlir::Value, 4> operands(result.operands);
  operands.append(control_operands.begin(), control_operands.end());
  auto loc = result.location;
  if (node.IsSwitch()) {
    if (node.op_def().name() == "_SwitchN") {
      return builder_.create<mlir::tf_executor::SwitchNOp>(loc, types, operands,
                                                           result.attributes);
    }
    return builder_.create<mlir::tf_executor::SwitchOp>(loc, types, operands,
                                                        result.attributes);
  }
  if (node.IsMerge()) {
    return builder_.create<mlir::tf_executor::MergeOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsNextIteration()) {
    mlir::OpBuilder builder_at_begin(builder_.getBlock(),
                                     builder_.getBlock()->begin());
    auto source_op =
        builder_at_begin.create<mlir::tf_executor::NextIterationSourceOp>(
            loc, operands[0].getType(), result.attributes);
    return builder_.create<mlir::tf_executor::NextIterationSinkOp>(
        loc, source_op.getToken(), operands, result.attributes);
  }
  if (node.IsLoopCond()) {
    return builder_.create<mlir::tf_executor::LoopCondOp>(loc, types, operands,
                                                          result.attributes);
  }
  if (node.IsEnter()) {
    return builder_.create<mlir::tf_executor::EnterOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsExit()) {
    return builder_.create<mlir::tf_executor::ExitOp>(loc, types, operands,
                                                      result.attributes);
  }
  if (node.IsControlTrigger()) {
    return builder_.create<mlir::tf_executor::ControlTriggerOp>(
        loc, mlir::ValueRange(operands), result.attributes);
  }
  auto island = builder_.create<mlir::tf_executor::IslandOp>(
      result.location, types, control_operands,
      mlir::ArrayRef<mlir::NamedAttribute>{});
  island.getBody().push_back(new mlir::Block);
  mlir::OpBuilder island_builder =
      mlir::OpBuilder::atBlockEnd(&island.GetBody());
  mlir::Operation* inner_op = island_builder.create(result);
  const auto set_segment_sizes_attr =
      [&](const NameRangeMap& arg_ranges,
          const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
          llvm::StringRef attr_name) {
        std::vector<int32_t> values;
        values.reserve(args.size());
        for (const auto& arg : args) {
          auto range = arg_ranges.at(arg.name());
          values.push_back(range.second - range.first);
        }
        auto attr_value =
            mlir::DenseI32ArrayAttr::get(inner_op->getContext(), values);
        inner_op->setAttr(attr_name, attr_value);
      };
  if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>() ||
      inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    NameRangeMap input_ranges, output_ranges;
    TF_CHECK_OK(
        NameRangesForNode(node, node.op_def(), &input_ranges, &output_ranges));
    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
      set_segment_sizes_attr(input_ranges, node.op_def().input_arg(),
                             mlir::OpTrait::AttrSizedOperandSegments<
                                 void>::getOperandSegmentSizeAttr());
    }
    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
      set_segment_sizes_attr(output_ranges, node.op_def().output_arg(),
                             mlir::OpTrait::AttrSizedResultSegments<
                                 void>::getResultSegmentSizeAttr());
    }
  }
  if (VLOG_IS_ON(1)) {
    mlir::OperationName name = inner_op->getName();
    if (!name.isRegistered() &&
        (node_type_name != "_Arg" && node_type_name != "_Retval") &&
        !unmodelled_op_names_.count(name.getIdentifier())) {
      if (node.op_def().is_stateful()) {
        VLOG(1) << "[potentially conservative] Op type `" << node.type_string()
                << "` is stateful but effects not modelled";
      } else {
        bool resource = false;
        std::function<bool(mlir::Type)> record_resource;
        record_resource = [&](mlir::Type type) {
          type.walk([&](mlir::Type t) {
            if (resource) return mlir::WalkResult::interrupt();
            if (mlir::isa<mlir::TF::ResourceType>(type)) {
              resource = true;
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
          return resource;
        };
        for (mlir::Type t : inner_op->getResultTypes())
          if (record_resource(t)) break;
        for (mlir::Type t : inner_op->getOperandTypes())
          if (record_resource(t)) break;
        if (resource) {
          unmodelled_op_names_.insert(name.getIdentifier());
          VLOG(1) << "[potentially conservative] Op type `"
                  << node.type_string()
                  << "` has resource operands/results but effects not modelled";
        }
      }
    }
  }
  island_builder.create<mlir::tf_executor::YieldOp>(result.location,
                                                    inner_op->getResults());
  return island.getOperation();
}
Status ImporterBase::ConvertNode(const Node& node) {
  if (!node.IsOp()) {
    return absl::OkStatus();
  }
  std::string node_type_name = node.type_string();
  const auto* func_def = graph_flib_.Find(node_type_name);
  bool convert_to_legacy_call = false;
  if (func_def) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(node_type_name));
    node_type_name = (*tf_name_to_mlir_name_)[node_type_name];
    convert_to_legacy_call = true;
  }
  auto get_full_op_name = [&](const std::string& op_name) {
    const char* kTfPrefix = "tf.";
    return kTfPrefix + op_name;
  };
  std::string op_name = get_full_op_name(node_type_name);
  if (back_edge_node_output_.contains(&node)) {
    op_name = op_name + ".sink";
  }
  mlir::OperationState result(GetLocation(node), op_name);
  for (int i = 0; i < node.num_outputs(); ++i) {
    if (back_edge_node_output_.contains(&node) &&
        back_edge_node_output_[&node] == i) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto type, InferOutputType(node, i, builder_));
    result.types.push_back(type);
  }
  absl::InlinedVector<const Edge*, 8> in_edges(node.in_edges().size());
  absl::c_copy(node.in_edges(), in_edges.begin());
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    if (e1->IsControlEdge() && !e2->IsControlEdge()) return false;
    if (!e1->IsControlEdge() && e2->IsControlEdge()) return true;
    if (e1->IsControlEdge() && e2->IsControlEdge())
      return e1->src()->id() < e2->src()->id();
    return e1->dst_input() < e2->dst_input();
  });
  result.operands.reserve(in_edges.size());
  mlir::SmallVector<mlir::Value, 8> control_operands;
  for (const auto* input_edge : in_edges) {
    const Node& input_node = *input_edge->src();
    if (input_node.IsSource()) {
      if (in_edges.size() != 1) {
        return errors::FailedPrecondition(
            "The node has other inputs besides the _Source node");
      }
      continue;
    }
    if (input_node.IsArg() && input_edge->IsControlEdge()) {
      continue;
    }
    if (node_values_.find(input_node.id()) == node_values_.end())
      return errors::FailedPrecondition(
          "Graph not traversed in reverse post order; use seen before def!");
    mlir::Operation* inst = node_values_[input_node.id()];
    if (input_edge->IsControlEdge())
      control_operands.push_back(inst->getResult(inst->getNumResults() - 1));
    else
      result.operands.push_back(inst->getResult(input_edge->src_output()));
  }
  using FuncPairType = std::pair<const std::string*, const AttrValue*>;
  std::vector<FuncPairType> funcs;
  result.attributes.reserve(node.attrs().size() + 2);
  auto abstract_op = result.name.getRegisteredInfo();
  auto derived_op =
      abstract_op
          ? abstract_op->getInterface<mlir::DerivedAttributeOpInterface>()
          : nullptr;
  for (const auto& name_and_value : node.attrs()) {
    const auto& attr_name = name_and_value.first;
    if (derived_op && derived_op->isDerivedAttribute(attr_name)) continue;
    const AttrValue& attr_value = name_and_value.second;
    if (IsOutputShapesAttribute(attr_value, attr_name)) continue;
    if (attr_value.value_case() == AttrValue::kFunc) {
      funcs.emplace_back(&attr_name, &attr_value);
    } else {
      TF_ASSIGN_OR_RETURN(auto attr, ConvertAttributeValue(attr_value));
      result.attributes.push_back(builder_.getNamedAttr(attr_name, attr));
    }
  }
  auto comparator = [](const FuncPairType& a, const FuncPairType& b) {
    return *a.first < *b.first;
  };
  std::sort(funcs.begin(), funcs.end(), comparator);
  for (const auto& func : funcs) {
    TF_RETURN_IF_ERROR(ConvertFunctionCallAttribute(*func.first, *func.second,
                                                    &result.attributes));
  }
  const auto& node_def = node.def();
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_def.device(), &parsed_name)) {
    return errors::InvalidArgument(
        "Op ", op_name, " has invalid device name: ", node_def.device());
  }
  if (!node_def.device().empty()) {
    if (!parsed_name.has_type) {
      parsed_name.type = "CPU";
      parsed_name.has_type = true;
    }
    if (!parsed_name.has_id) {
      parsed_name.id = 0;
      parsed_name.has_id = true;
    }
  }
  result.attributes.push_back(builder_.getNamedAttr(
      "device", builder_.getStringAttr(
                    DeviceNameUtils::ParsedNameToString(parsed_name))));
  if (convert_to_legacy_call) {
    result.name = mlir::OperationName(get_full_op_name("LegacyCall"), context_);
    mlir::SymbolRefAttr val =
        mlir::SymbolRefAttr::get(builder_.getContext(), node_type_name);
    result.addAttribute("f", val);
    if (!result.attributes.get("_disable_call_shape_inference")) {
      result.addAttribute("_disable_call_shape_inference",
                          builder_.getBoolAttr(false));
    }
  }
  auto composite_control_flow_op = [&](const std::string& name) {
    result.name = mlir::OperationName(get_full_op_name(name), context_);
    bool stateless = absl::StartsWith(node_type_name, "Stateless");
    mlir::BoolAttr val = builder_.getBoolAttr(stateless);
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  };
  if (node.IsCaseNode()) composite_control_flow_op("Case");
  if (node.IsIfNode()) composite_control_flow_op("If");
  if (node.IsWhileNode()) {
    composite_control_flow_op("While");
    auto* output_shapes = node.attrs().Find("output_shapes");
    if (output_shapes && !output_shapes->list().shape().empty())
      result.attributes.push_back(
          builder_.getNamedAttr("shape_invariant", builder_.getUnitAttr()));
  }
  node_values_[node.id()] =
      CreateOperation(node, node_type_name, result, control_operands);
  return absl::OkStatus();
}
Status ImporterBase::AddBackedges() {
  for (auto it : back_edge_dst_inputs_) {
    BackEdge& edge = it.second;
    if (!edge.src->IsNextIteration() || !edge.dst->IsMerge()) {
      return errors::FailedPrecondition(
          "Invalid backedge; should be from NextIteration to Merge!");
    }
    auto* sink = node_values_[edge.src->id()];
    auto* dst = node_values_[edge.dst->id()];
    TF_RETURN_IF_ERROR(AddBackedge(sink, dst, edge.dst_input));
  }
  return absl::OkStatus();
}
Status ImporterBase::AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                                 int dst_input) {
  mlir::Operation* source = sink->getOperand(0).getDefiningOp();
  mlir::OperationState state(dst->getLoc(), dst->getName());
  auto num_operands = dst->getNumOperands();
  state.operands.reserve(num_operands + 1);
  for (int input = 0, e = num_operands + 1; input != e; ++input) {
    if (input < dst_input) {
      state.operands.push_back(dst->getOperand(input));
    } else if (input == dst_input) {
      state.operands.push_back(source->getResult(0));
    } else {
      state.operands.push_back(dst->getOperand(input - 1));
    }
  }
  state.attributes.assign(dst->getAttrs().begin(), dst->getAttrs().end());
  state.types.assign(dst->getResultTypes().begin(),
                     dst->getResultTypes().end());
  builder_.setInsertionPoint(dst);
  auto* new_dst = builder_.create(state);
  for (unsigned i = 0, e = dst->getNumResults(); i != e; ++i) {
    auto new_output = new_dst->getResult(i);
    dst->getResult(i).replaceAllUsesWith(new_output);
  }
  dst->dropAllReferences();
  dst->erase();
  return absl::OkStatus();
}
absl::StatusOr<mlir::FunctionType> ImporterBase::InferLibFunctionType(
    const FunctionBody& fbody) {
  mlir::Builder builder(context_);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  if (specs_.inputs.empty()) {
    arg_types.reserve(fbody.arg_types.size());
    for (auto arg : fbody.arg_nodes) {
      auto* node = graph_->FindNodeId(arg->id());
      TF_ASSIGN_OR_RETURN(auto type,
                          InferOutputType(*node, 0, builder));
      arg_types.push_back(type);
    }
  } else {
    arg_types.reserve(fbody.arg_types.size());
    for (const auto& it : llvm::enumerate(specs_.inputs)) {
      mlir::Type element_type;
      const auto& node_info = it.value().second;
      DataType dtype = node_info.imported_dtype;
      if (dtype == DT_INVALID) {
        auto arg = fbody.arg_nodes[it.index()];
        auto* node = graph_->FindNodeId(arg->id());
        dtype = node->output_type(0);
        if (dtype == DT_INVALID) {
          return errors::InvalidArgument("Input ", it.index(),
                                         "has invalid data type");
        }
      }
      TF_RETURN_IF_ERROR(
          ::tensorflow::ConvertDataType(dtype, builder, &element_type));
      if (node_info.shape.unknown_rank()) {
        arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
      } else {
        llvm::SmallVector<int64_t, 4> shape;
        TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
        arg_types.push_back(GetTypeFromTFTensorShape(shape, element_type));
      }
    }
  }
  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(fbody.ret_types.size());
  for (auto ret : fbody.ret_nodes) {
    auto* node = graph_->FindNodeId(ret->id());
    TF_ASSIGN_OR_RETURN(auto type, InferInputType(*node, 0, builder));
    ret_types.push_back(type);
  }
  return builder.getFunctionType(arg_types, ret_types);
}
class GraphDefImporter : public ImporterBase {
 public:
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      mlir::MLIRContext* context, const Graph& graph,
      const GraphDebugInfo& debug_info,
      const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
      std::unordered_map<std::string, std::string>& tf_name_to_mlir_name,
      bool disable_crash_analysis = false);
 private:
  explicit GraphDefImporter(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier)
      : ImporterBase(flib, debug_info, specs, module, tf_name_to_mlir_name,
                     function_name_uniquifier) {}
  absl::StatusOr<mlir::FunctionType> InferMainFunctionType(
      const GraphImportConfig& specs, mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);
  absl::StatusOr<mlir::FunctionType> GetArgsRetsAndTypesFromFunctionGraph(
      mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);
  Status GetControlRetsFromGraph(
      llvm::ArrayRef<std::string> control_outputs,
      absl::InlinedVector<Node*, 4>* control_ret_nodes);
};
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphDefImporter::Convert(
    mlir::MLIRContext* context, const Graph& graph,
    const GraphDebugInfo& debug_info, const FunctionLibraryDefinition& flib_def,
    const GraphImportConfig& specs,
    std::unordered_map<std::string, std::string>& tf_name_to_mlir_name,
    bool disable_crash_analysis) {
  LoadImporterDialects(*context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  NameUniquifier function_name_uniquifier(flib_def);
  auto graph_def = std::make_unique<GraphDef>();
  graph.ToGraphDef(graph_def.get(), false);
  auto scope_exit = [&]() {
    std::function<void()> cleanup = []() {};
    if (!disable_crash_analysis) {
      static std::atomic<uint32> counter(0);
      uint32 current_file_prefix = counter++;
      const auto* graph_crash_handle = crash_analysis::ReportProtoDataOnCrash(
          absl::StrCat(current_file_prefix, "_mlir_import_graph.pbtxt"),
          *graph_def);
      auto reachable_flib = flib_def.ReachableDefinitions(*graph_def);
      const auto* flib_crash_handle = crash_analysis::ReportProtoDataOnCrash(
          absl::StrCat(current_file_prefix, "_mlir_import_flib.pbtxt"),
          reachable_flib.ToProto());
      cleanup = [=]() {
        crash_analysis::RemoveReportData(graph_crash_handle);
        crash_analysis::RemoveReportData(flib_crash_handle);
      };
    }
    return llvm::make_scope_exit(std::move(cleanup));
  }();
  VLOG(2) << "Importing: "
          << ::tensorflow::DumpGraphToFile("tf_mlir_importer_base", graph,
                                           &flib_def);
  GraphDefImporter importer(flib_def, debug_info, specs, module.get(),
                            &tf_name_to_mlir_name, &function_name_uniquifier);
  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph, std::move(graph_def)));
  mlir::FunctionType func_type;
  absl::InlinedVector<OutputTensor, 4> arg_nodes;
  absl::InlinedVector<OutputTensor, 4> ret_nodes;
  absl::InlinedVector<Node*, 4> control_ret_nodes;
  llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
  if (specs.graph_as_function) {
    if (specs.prune_unused_nodes || !specs.inputs.empty() ||
        !specs.outputs.empty())
      return errors::InvalidArgument(
          "Pruning of graph is currently unsupported when the main graph is "
          "converted to a function.");
    TF_ASSIGN_OR_RETURN(func_type,
                        importer.GetArgsRetsAndTypesFromFunctionGraph(
                            context, &arg_nodes, &ret_nodes));
    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));
    mlir::Builder b(context);
    std::string s;
    llvm::raw_string_ostream ss(s);
    auto node_name = [&](const OutputTensor& tensor) {
      ss << tensor.node->name();
    };
    llvm::interleave(arg_nodes, ss, node_name, ",");
    auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
    s.clear();
    llvm::interleave(ret_nodes, ss, node_name, ",");
    auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
    s.clear();
    llvm::interleave(specs.control_outputs, ss, ",");
    auto control_outputs =
        b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));
    attrs.push_back(b.getNamedAttr(
        "tf.entry_function",
        b.getDictionaryAttr({inputs, outputs, control_outputs})));
    if (!specs.xla_compile_device_type.empty()) {
      attrs.push_back(
          b.getNamedAttr("_xla_compile_device_type",
                         b.getStringAttr(specs.xla_compile_device_type)));
    }
    attrs.push_back(b.getNamedAttr("allow_soft_placement",
                                   b.getBoolAttr(specs.enable_soft_placement)));
  } else {
    TF_ASSIGN_OR_RETURN(func_type, importer.InferMainFunctionType(
                                       specs, context, &arg_nodes, &ret_nodes));
    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));
    if (!specs.inputs.empty() || !specs.outputs.empty() ||
        !specs.control_outputs.empty()) {
      mlir::Builder b(context);
      std::string s;
      llvm::raw_string_ostream ss(s);
      llvm::interleave(
          specs.inputs, ss,
          [&](const std::pair<std::string, ArrayInfo>& v) { ss << v.first; },
          ",");
      auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.outputs, ss, ",");
      auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.control_outputs, ss, ",");
      auto control_outputs =
          b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));
      attrs.push_back(b.getNamedAttr(
          "tf.entry_function",
          b.getDictionaryAttr({inputs, outputs, control_outputs})));
    }
  }
  PopulateTfVersions(module.get(), graph.versions());
  const llvm::StringRef& graph_func_name =
      specs.graph_func_name.empty() ? kImportModelDefaultGraphFuncName
                                    : specs.graph_func_name;
  TF_RETURN_IF_ERROR(importer.ImporterBase::Convert(graph_func_name, func_type,
                                                    arg_nodes, ret_nodes,
                                                    control_ret_nodes, attrs));
  if (specs.convert_all_functions_to_mlir) {
    auto fn_names = graph.flib_def().ListFunctionNames();
    for (const auto& fn_name : fn_names) {
      TF_RETURN_IF_ERROR(importer.ConvertLibFunction(fn_name));
    }
  }
  TF_RETURN_IF_ERROR(importer.ImporterBase::ConvertDeferredFunctions());
  for (auto function : module.get().getOps<mlir::func::FuncOp>()) {
    auto visibility = function.getName() == graph_func_name
                          ? mlir::func::FuncOp::Visibility::Public
                          : mlir::func::FuncOp::Visibility::Private;
    function.setVisibility(visibility);
  }
  VLOG(2) << "Imported: "
          << tensorflow::DumpMlirOpToFile("tf_mlir_imported_base",
                                          module.get());
  return module;
}
absl::StatusOr<mlir::FunctionType> GraphDefImporter::InferMainFunctionType(
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  absl::flat_hash_map<absl::string_view, int> inputs;
  for (const auto& input_and_idx : llvm::enumerate(specs.inputs)) {
    TensorId tensor = ParseTensorName(input_and_idx.value().first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      inputs.insert({remapped_it->second, input_and_idx.index()});
    } else {
      inputs.insert({tensor.node(), input_and_idx.index()});
    }
  }
  absl::flat_hash_set<absl::string_view> output_node_names;
  std::vector<TensorId> outputs;
  output_node_names.reserve(specs.outputs.size());
  for (const auto& output : specs.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      output_node_names.insert(remapped_it->second);
      outputs.push_back({remapped_it->second, 0});
    } else {
      output_node_names.insert(tensor.node());
      outputs.push_back(tensor);
    }
  }
  if (!inputs.empty() || !outputs.empty()) {
    arg_nodes->resize(inputs.size());
    ret_nodes->resize(outputs.size());
    for (Node* n : GetOrderedNodes()) {
      auto input_it = inputs.find(n->name());
      if (input_it != inputs.end()) {
        (*arg_nodes)[input_it->second] = {n, 0};
      }
      if (output_node_names.contains(n->name())) {
        for (int i = 0, e = outputs.size(); i != e; ++i) {
          TensorId tensor = outputs[i];
          if (n->name() != tensor.node()) continue;
          (*ret_nodes)[i] = {n, tensor.index()};
        }
      }
    }
  }
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(specs.inputs.size());
  int i = 0;
  for (const auto& it : specs.inputs) {
    Node* arg_node = arg_nodes->at(i).node;
    if (arg_node == nullptr) {
      return errors::InvalidArgument("Input ", it.first,
                                     " was not found in graph");
    }
    mlir::Type element_type;
    const auto& node_info = it.second;
    DataType imported_dtype = node_info.imported_dtype;
    if (imported_dtype == DT_INVALID) {
      imported_dtype = arg_node->output_type(0);
      if (imported_dtype == DT_INVALID) {
        return errors::InvalidArgument("Input ", i, "has invalid data type");
      }
    }
    if (!node_info.subtypes.empty()) {
      std::vector<mlir::TensorType> subtypes;
      for (const auto& st : node_info.subtypes) {
        mlir::Type st_data_type;
        llvm::SmallVector<int64_t> shape;
        TF_RETURN_IF_ERROR(ConvertToMlirShape(st.shape, &shape));
        TF_RETURN_IF_ERROR(
            ConvertDataType(st.imported_dtype, builder, &st_data_type));
        subtypes.push_back(GetTypeFromTFTensorShape(shape, st_data_type));
      }
      if (imported_dtype == DT_RESOURCE) {
        element_type =
            mlir::TF::ResourceType::get(subtypes, builder.getContext());
      } else if (imported_dtype == DT_VARIANT) {
        element_type =
            mlir::TF::VariantType::get(subtypes, builder.getContext());
      } else {
        return errors::InvalidArgument(DataType_Name(imported_dtype),
                                       " takes no subtypes.");
      }
    } else {
      TF_RETURN_IF_ERROR(
          ConvertDataType(imported_dtype, builder, &element_type));
    }
    if (node_info.shape.unknown_rank()) {
      arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
    } else {
      llvm::SmallVector<int64_t, 4> shape;
      TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
      arg_types.push_back(GetTypeFromTFTensorShape(shape, element_type));
    }
    i++;
  }
  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(specs.outputs.size());
  for (int i = 0, e = specs.outputs.size(); i != e; ++i) {
    if (ret_nodes->at(i).node == nullptr) {
      return errors::InvalidArgument("Output ", specs.outputs[i],
                                     " was not found in graph");
    }
  }
  for (const auto& ret : *ret_nodes) {
    if (ret.node->num_outputs() <= ret.index) {
      return errors::InvalidArgument("Invalid output index ", ret.index,
                                     " specified for node: ", ret.node->name());
    }
    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*ret.node, ret.index, builder));
    ret_types.push_back(type);
  }
  return builder.getFunctionType(arg_types, ret_types);
}
absl::StatusOr<mlir::FunctionType>
GraphDefImporter::GetArgsRetsAndTypesFromFunctionGraph(
    mlir::MLIRContext* context, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  auto add_node = [](Node* node, absl::InlinedVector<OutputTensor, 4>* nodes) {
    auto* attr = node->attrs().Find("index");
    if (!attr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(),
                                     "' is missing attribute 'index'");
    auto index = attr->i();
    const int num_nodes = nodes->size();
    if (num_nodes < index + 1) nodes->resize(index + 1);
    if ((*nodes)[index].node != nullptr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(), "' has attribute 'index' ",
                                     index, " that conflicts with node '",
                                     (*nodes)[index].node->name(), "'");
    (*nodes)[index] = {node, 0};
    return absl::OkStatus();
  };
  for (auto* node : GetOrderedNodes())
    if (node->IsArg())
      TF_RETURN_IF_ERROR(add_node(node, arg_nodes));
    else if (node->IsRetval())
      TF_RETURN_IF_ERROR(add_node(node, ret_nodes));
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(arg_nodes->size());
  for (const auto& arg_node_and_idx : llvm::enumerate(*arg_nodes)) {
    auto& arg_node = arg_node_and_idx.value();
    if (arg_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Arg at index ",
                                     arg_node_and_idx.index());
    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*arg_node.node, 0, builder));
    arg_types.push_back(type);
  }
  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(ret_nodes->size());
  for (const auto& ret_node_and_idx : llvm::enumerate(*ret_nodes)) {
    auto& ret_node = ret_node_and_idx.value();
    if (ret_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Retval at index ",
                                     ret_node_and_idx.index());
    TF_ASSIGN_OR_RETURN(auto type,
                        InferInputType(*ret_node.node, 0, builder));
    ret_types.push_back(type);
  }
  return builder.getFunctionType(arg_types, ret_types);
}
Status GraphDefImporter::GetControlRetsFromGraph(
    llvm::ArrayRef<std::string> control_outputs,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  if (control_outputs.empty()) return absl::OkStatus();
  llvm::SmallDenseMap<llvm::StringRef, int32_t> controls_to_idx;
  for (const auto& control_and_idx : llvm::enumerate(control_outputs))
    controls_to_idx.insert({control_and_idx.value(), control_and_idx.index()});
  if (controls_to_idx.size() != control_outputs.size())
    return errors::InvalidArgument("Control outputs must be unique");
  control_ret_nodes->resize(controls_to_idx.size());
  for (auto* node : GetOrderedNodes()) {
    auto it = controls_to_idx.find(node->name());
    if (it != controls_to_idx.end()) (*control_ret_nodes)[it->second] = node;
  }
  for (auto node_and_name : llvm::zip(*control_ret_nodes, control_outputs))
    if (std::get<0>(node_and_name) == nullptr)
      return errors::InvalidArgument(
          "Control output '", std::get<1>(node_and_name), "' is missing");
  return absl::OkStatus();
}
class ObjectNames {
 public:
  explicit ObjectNames(const SavedObjectGraph& object_graph,
                       absl::Span<std::string> exported_names);
  llvm::ArrayRef<llvm::StringRef> GetExportedNames(int node_id) const;
  llvm::StringRef GetSymbolTableName(int node_id) const;
 private:
  std::string GetDefaultSymbolTableName(int node_id) const;
  bool IsExported(const std::string& name);
  void RecursivelyVisitObjectGraph(int node_id);
  llvm::StringRef SaveString(const std::string& s) const;
  const SavedObjectGraph& object_graph_;
  std::unordered_set<std::string> names_to_export_;
  llvm::SmallVector<std::string, 8> path_segments_;
  absl::flat_hash_set<int> on_stack_nodes_;
  absl::flat_hash_map<int, std::vector<std::string>> object_names_;
  absl::flat_hash_map<int, llvm::SmallVector<llvm::StringRef, 1>>
      exported_names_;
  absl::flat_hash_map<int, llvm::StringRef> pretty_symbol_table_name_;
  mutable std::unordered_set<std::string> saved_strings_;
};
ObjectNames::ObjectNames(const SavedObjectGraph& object_graph,
                         absl::Span<std::string> exported_names)
    : object_graph_(object_graph),
      names_to_export_(exported_names.begin(), exported_names.end()) {
  RecursivelyVisitObjectGraph(0);
  for (auto& kv : object_names_) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](absl::string_view a, absl::string_view b) {
                return std::make_tuple(isdigit(a.back()), a.size(), a) <
                       std::make_tuple(isdigit(b.back()), b.size(), b);
              });
    for (const std::string& name : kv.second) {
      if (IsExported(name)) {
        exported_names_[kv.first].push_back(SaveString(name));
      }
    }
  }
  for (auto& kv : object_names_) {
    int node_id = kv.first;
    std::string internal_name =
        absl::StrCat(GetDefaultSymbolTableName(node_id), "__");
    if (exported_names_.find(node_id) != exported_names_.end()) {
      internal_name += exported_names_[node_id][0].str();
    } else {
      internal_name += object_names_[node_id][0];
    }
    pretty_symbol_table_name_[node_id] = SaveString(internal_name);
  }
}
llvm::ArrayRef<llvm::StringRef> ObjectNames::GetExportedNames(
    int node_id) const {
  auto it = exported_names_.find(node_id);
  if (it != exported_names_.end()) {
    return it->second;
  }
  return {};
}
llvm::StringRef ObjectNames::GetSymbolTableName(int node_id) const {
  auto it = pretty_symbol_table_name_.find(node_id);
  if (it != pretty_symbol_table_name_.end()) {
    return it->second;
  }
  return SaveString(GetDefaultSymbolTableName(node_id));
}
std::string ObjectNames::GetDefaultSymbolTableName(int node_id) const {
  return absl::StrCat("__sm_node", node_id);
}
bool ObjectNames::IsExported(const std::string& name) {
  if (names_to_export_.empty()) {
    return true;
  }
  return names_to_export_.find(name) != names_to_export_.end();
}
void ObjectNames::RecursivelyVisitObjectGraph(int node_id) {
  const SavedObject& object = object_graph_.nodes(node_id);
  switch (object.kind_case()) {
    case SavedObject::kConstant:
    case SavedObject::kFunction:
    case SavedObject::kVariable: {
      object_names_[node_id].push_back(absl::StrJoin(path_segments_, "."));
      break;
    }
    default:
      break;
  }
  for (const auto& child_ref : object.children()) {
    bool on_stack = !on_stack_nodes_.insert(child_ref.node_id()).second;
    if (on_stack) {
      continue;
    }
    path_segments_.push_back(child_ref.local_name());
    RecursivelyVisitObjectGraph(child_ref.node_id());
    path_segments_.pop_back();
    on_stack_nodes_.erase(child_ref.node_id());
  }
}
llvm::StringRef ObjectNames::SaveString(const std::string& s) const {
  return llvm::StringRef(*saved_strings_.insert(s).first);
}
const TensorProto* ExtractConstTensorFromGraph(const GraphDef& graph_def,
                                               const std::string& op_name) {
  const NodeDef* match_node = nullptr;
  for (const auto& node : graph_def.node()) {
    if (node.name() == op_name) {
      match_node = &node;
    }
  }
  if (!match_node) {
    return nullptr;
  }
  auto value_it = match_node->attr().find("value");
  if (value_it == match_node->attr().end()) {
    return nullptr;
  }
  if (!value_it->second.has_tensor()) {
    return nullptr;
  }
  return &value_it->second.tensor();
}
const TrackableObjectGraph::TrackableObject::SerializedTensor*
FindSerializedTensorInTrackable(
    const TrackableObjectGraph::TrackableObject& trackable_object,
    StringPiece name) {
  for (const auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == name) {
      return &maybe_serialized_tensor;
    }
  }
  return nullptr;
}
Status DiagnoseMultipleConcreteFunctions(const SavedObjectGraph& object_graph,
                                         const ObjectNames& object_names) {
  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    if (object_names.GetExportedNames(node_id).empty()) {
      continue;
    }
    if (object.kind_case() == SavedObject::kFunction) {
      if (object.function().concrete_functions_size() != 1) {
        llvm::SmallVector<std::string, 4> names;
        for (llvm::StringRef s : object_names.GetExportedNames(node_id)) {
          names.push_back("'" + s.str() + "'");
        }
        return errors::InvalidArgument(
            "Exported function with exported name(s) ",
            absl::StrJoin(names, ", "),
            " with multiple concrete functions. Add "
            "@tf.function(input_signature=[...]) on this function, or use a "
            "narrower list of exported names that excludes this function.");
      }
    }
  }
  return absl::OkStatus();
}
class StructuredValueLinearizer {
 public:
  StructuredValueLinearizer(const StructuredValue& value,
                            mlir::MLIRContext* context);
  absl::StatusOr<llvm::ArrayRef<mlir::ArrayAttr>> GetLeafIndexPaths(
      llvm::StringRef error_context) const;
 private:
  void RecursivelyFindLeaves(const StructuredValue& value);
  mlir::Builder builder_;
  llvm::SmallVector<mlir::Attribute, 4> current_index_path_;
  llvm::SmallVector<mlir::ArrayAttr, 4> leaf_index_paths_;
  std::string error_message_;
};
StructuredValueLinearizer::StructuredValueLinearizer(
    const StructuredValue& value, mlir::MLIRContext* context)
    : builder_(context) {
  RecursivelyFindLeaves(value);
}
absl::StatusOr<llvm::ArrayRef<mlir::ArrayAttr>>
StructuredValueLinearizer::GetLeafIndexPaths(
    llvm::StringRef error_context) const {
  if (error_message_.empty()) {
    return llvm::ArrayRef(leaf_index_paths_);
  }
  return errors::InvalidArgument(
      error_context.str(), error_message_,
      "This likely means that you have @tf.function "
      "on an exported function instead of "
      "@tf.function(input_signature=[...]). Consider annotating an "
      "input_signature or narrowing your set of "
      "exported names to not include this function.");
}
void StructuredValueLinearizer::RecursivelyFindLeaves(
    const StructuredValue& value) {
  switch (value.kind_case()) {
    case StructuredValue::kDictValue: {
      const DictValue& dict = value.dict_value();
      using FieldTy = protobuf::MapPair<std::string, StructuredValue>;
      llvm::SmallVector<const FieldTy*, 4> fields;
      for (auto& field : dict.fields()) {
        fields.push_back(&field);
      }
      llvm::sort(fields, [](const FieldTy* a, const FieldTy* b) {
        return a->first < b->first;
      });
      for (auto& field : fields) {
        current_index_path_.push_back(builder_.getStringAttr(field->first));
        RecursivelyFindLeaves(field->second);
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTupleValue: {
      const TupleValue& tuple = value.tuple_value();
      for (int i = 0, e = tuple.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(tuple.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kListValue: {
      const ListValue& list = value.list_value();
      for (int i = 0, e = list.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(list.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTensorSpecValue: {
      leaf_index_paths_.push_back(builder_.getArrayAttr(current_index_path_));
      return;
    }
    case StructuredValue::kNoneValue: {
      return;
    }
    default: {
      llvm::raw_string_ostream os(error_message_);
      os << "Unhandled structured value kind " << value.kind_case()
         << " at index path: <value>";
      for (auto path_element : current_index_path_) {
        os << ".";
        if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(path_element)) {
          os << integer.getValue();
        } else {
          auto str = mlir::cast<mlir::StringAttr>(path_element);
          os << str.getValue();
        }
      }
      os << "\n";
    }
  }
}
void AdjustBoundInputArgTypes(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (!mlir::tf_saved_model::IsExported(func)) continue;
    mlir::OpBuilder builder(func.getBody());
    llvm::SmallVector<mlir::Type, 4> new_input_types;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto arg = func.getArgument(i);
      auto global_tensor = mlir::tf_saved_model::LookupBoundInputOfType<
          mlir::tf_saved_model::GlobalTensorOp>(func, i, symbol_table);
      if (global_tensor) {
        auto old_type = arg.getType();
        auto new_type =
            mlir::tf_saved_model::GetBoundInputArgTypeFor(global_tensor);
        arg.setType(new_type);
        if (global_tensor.getIsMutable()) {
          auto arg_with_original_type = builder.create<mlir::TF::CastOp>(
              global_tensor.getLoc(), old_type, arg,
              builder.getBoolAttr(false));
          arg.replaceAllUsesWith(arg_with_original_type);
          arg_with_original_type.setOperand(arg);
        } else {
          auto arg_with_original_type =
              builder.create<mlir::TF::ReadVariableOp>(global_tensor.getLoc(),
                                                       old_type, arg);
          arg.replaceAllUsesWith(arg_with_original_type);
          arg_with_original_type.setOperand(arg);
        }
      }
      new_input_types.push_back(arg.getType());
    }
    func.setType(mlir::FunctionType::get(module.getContext(), new_input_types,
                                         func.getFunctionType().getResults()));
  }
}
void MarkSavedModelFunctionVisibility(mlir::ModuleOp module) {
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto visibility = mlir::tf_saved_model::IsExported(func)
                          ? mlir::func::FuncOp::Visibility::Public
                          : mlir::func::FuncOp::Visibility::Private;
    func.setVisibility(visibility);
  }
}
void SortSavedModelModule(mlir::ModuleOp module) {
  struct NamedGlobalTensor {
    llvm::StringRef name;
    GlobalTensorOp global_tensor;
  };
  llvm::SmallVector<NamedGlobalTensor, 8> named_global_tensors;
  for (auto global_tensor : module.getOps<GlobalTensorOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(global_tensor);
    named_global_tensors.push_back(
        {exported_names.empty() ? "" : exported_names.front(), global_tensor});
  }
  llvm::stable_sort(named_global_tensors,
                    [](const NamedGlobalTensor& a, const NamedGlobalTensor& b) {
                      return std::make_tuple(a.name.empty(), a.name) <
                             std::make_tuple(b.name.empty(), b.name);
                    });
  struct NamedFunc {
    llvm::StringRef name;
    mlir::func::FuncOp func;
  };
  llvm::SmallVector<NamedFunc, 8> named_funcs;
  llvm::SmallVector<mlir::func::FuncOp, 8> private_funcs;
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
    if (!exported_names.empty())
      named_funcs.push_back({exported_names.front(), func});
    else
      private_funcs.push_back(func);
  }
  llvm::stable_sort(named_funcs, [](const NamedFunc& a, const NamedFunc& b) {
    return a.name < b.name;
  });
  llvm::stable_sort(private_funcs,
                    [](mlir::func::FuncOp a, mlir::func::FuncOp b) {
                      return a.getName() < b.getName();
                    });
  struct NamedAsset {
    llvm::StringRef name;
    AssetOp asset;
  };
  llvm::SmallVector<NamedAsset, 4> assets;
  for (auto asset : module.getOps<AssetOp>()) {
    assets.push_back({asset.getName(), asset});
  }
  llvm::stable_sort(assets, [](const NamedAsset& a, const NamedAsset& b) {
    return a.name < b.name;
  });
  for (auto func : llvm::reverse(private_funcs)) {
    func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_func : llvm::reverse(named_funcs)) {
    named_func.func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_global_tensor : llvm::reverse(named_global_tensors)) {
    named_global_tensor.global_tensor.getOperation()->moveBefore(
        &module.getBody()->front());
  }
  for (auto asset : assets) {
    asset.asset.getOperation()->moveBefore(&module.getBody()->front());
  }
  auto initializers = module.getOps<SessionInitializerOp>();
  if (!initializers.empty()) {
    (*initializers.begin())
        .getOperation()
        ->moveBefore(&module.getBody()->front());
  }
}
Status CreateSavedModelIR(
    const ObjectNames& object_names, mlir::ModuleOp module,
    const SavedObjectGraph& object_graph,
    const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name,
    SavedModelV2Bundle* saved_model, MLIRImportOptions import_options) {
  mlir::OpBuilder builder(module.getBodyRegion());
  mlir::SymbolTable symbol_table(module);
  absl::flat_hash_map<int, const TrackableObjectGraph::TrackableObject*>
      restored_objects;
  TF_RETURN_IF_ERROR(saved_model->VisitObjectsToRestore(
      [&](int saved_node_id,
          const TrackableObjectGraph::TrackableObject& trackable_object) {
        restored_objects.insert(
            std::make_pair(saved_node_id, &trackable_object));
        return absl::OkStatus();
      }));
  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    if (object.kind_case() == SavedObject::kFunction) {
      if (object_names.GetExportedNames(node_id).empty()) {
        continue;
      }
      std::string error_context =
          "While importing SavedModel function '" +
          object_names.GetExportedNames(node_id)[0].str() + "': ";
      const SavedFunction& function = object.function();
      auto orig_func = symbol_table.lookup<mlir::func::FuncOp>(
          tf_name_to_mlir_name.find(function.concrete_functions(0))->second);
      mlir::func::FuncOp func = orig_func;
      if (!mlir::SymbolTable::symbolKnownUseEmpty(orig_func.getSymNameAttr(),
                                                  &module.getBodyRegion())) {
        func = orig_func.cloneWithoutRegions();
        module.insert(module.getBody()->begin(), func);
        func.addEntryBlock();
        func.setName(builder.getStringAttr("__sm_exported_" +
                                           orig_func.getName().str()));
        llvm::SmallVector<mlir::Value, 4> args_as_values;
        for (auto block_argument : func.getArguments()) {
          args_as_values.push_back(block_argument);
        }
        mlir::OpBuilder body_builder(&func.getBody());
        auto call = body_builder.create<mlir::TF::StatefulPartitionedCallOp>(
            func.getLoc(), orig_func.getFunctionType().getResults(),
            args_as_values,
            mlir::SymbolRefAttr::get(builder.getContext(), orig_func.getName()),
            builder.getStringAttr(""),
            builder.getStringAttr(""),
            builder.getStringAttr(""));
        body_builder.create<mlir::func::ReturnOp>(func.getLoc(),
                                                  call.getResults());
      }
      func->setAttr(
          kTfSavedModelExportedNamesAttr,
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      const SavedConcreteFunction& concrete_function =
          object_graph.concrete_functions().at(function.concrete_functions(0));
      auto positional_arg_structure =
          concrete_function.canonicalized_input_signature()
              .tuple_value()
              .values(0);
      StructuredValueLinearizer input_linearizer(positional_arg_structure,
                                                 builder.getContext());
      int bound_input_base =
          func.getNumArguments() - concrete_function.bound_inputs_size();
      TF_ASSIGN_OR_RETURN(auto input_index_paths,
                          input_linearizer.GetLeafIndexPaths(
                              error_context + "in input signature: "));
      const int input_index_paths_size = input_index_paths.size();
      if (bound_input_base != input_index_paths_size) {
        return errors::InvalidArgument(
            error_context,
            "Argument mismatch between concrete function input signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", input_index_paths.size(),
            " vs ", bound_input_base, ")");
      }
      for (const auto& index_path : llvm::enumerate(input_index_paths)) {
        func.setArgAttr(index_path.index(), kTfSavedModelIndexPathAttr,
                        index_path.value());
      }
      for (const auto& bound_input :
           llvm::enumerate(concrete_function.bound_inputs())) {
        int arg_index = bound_input_base + bound_input.index();
        auto symbol_ref = mlir::SymbolRefAttr::get(
            builder.getContext(),
            object_names.GetSymbolTableName(bound_input.value()));
        func.setArgAttr(arg_index, "tf_saved_model.bound_input", symbol_ref);
      }
      StructuredValueLinearizer output_linearizer(
          concrete_function.output_signature(), builder.getContext());
      TF_ASSIGN_OR_RETURN(auto output_index_paths,
                          output_linearizer.GetLeafIndexPaths(
                              error_context + "in output signature: "));
      if (func.getNumResults() != output_index_paths.size()) {
        return errors::InvalidArgument(
            error_context,
            "Result mismatch between concrete function output signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", output_index_paths.size(),
            " vs ", func.getNumResults(), ")");
      }
      for (const auto& index_path : llvm::enumerate(output_index_paths)) {
        func.setResultAttr(index_path.index(), kTfSavedModelIndexPathAttr,
                           index_path.value());
      }
    } else if (object.kind_case() == SavedObject::kVariable) {
      const SavedVariable& variable = object.variable();
      auto variable_trackable_it = restored_objects.find(node_id);
      TF_ASSIGN_OR_RETURN(
          auto type, ConvertToMlirTensorType(variable.shape(), variable.dtype(),
                                             &builder));
      if (variable_trackable_it == restored_objects.end()) {
        if (!import_options.allow_uninitialized_variables) {
          return errors::FailedPrecondition(
              "Could not restore saved variable: ", variable.name());
        }
        auto op = builder.create<mlir::tf_saved_model::GlobalTensorOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
            mlir::ElementsAttr(),
            mlir::TypeAttr::get(type),
            builder.getUnitAttr());
        op->setAttr(
            kTfSavedModelExportedNamesAttr,
            builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      } else {
        const auto* serialized_tensor_attr = FindSerializedTensorInTrackable(
            *variable_trackable_it->second, "VARIABLE_VALUE");
        if (!serialized_tensor_attr) {
          return errors::FailedPrecondition(
              "Could not find serialized tensor for saved variable: ",
              variable.name());
        }
        const auto& checkpoint_key = serialized_tensor_attr->checkpoint_key();
        Tensor value;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            saved_model->variable_reader()->Lookup(checkpoint_key, &value),
            "Could not read checkpoint key from variables bundle: ",
            checkpoint_key);
        TF_ASSIGN_OR_RETURN(auto value_attr, ConvertTensor(value, &builder));
        auto op = builder.create<GlobalTensorOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
            value_attr,
            mlir::TypeAttr::get(type),
            builder.getUnitAttr());
        op->setAttr(
            kTfSavedModelExportedNamesAttr,
            builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      }
    } else if (object.kind_case() == SavedObject::kConstant) {
      const SavedConstant& constant = object.constant();
      const TensorProto* value = ExtractConstTensorFromGraph(
          saved_model->meta_graph_def().graph_def(), constant.operation());
      if (!value) {
        return errors::FailedPrecondition(
            "Unable to find const node referenced in object graph: ",
            constant.operation());
      }
      TF_ASSIGN_OR_RETURN(auto value_attr,
                          ConvertTensorProto(*value, &builder));
      auto op = builder.create<GlobalTensorOp>(
          builder.getUnknownLoc(),
          builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
          value_attr,
          mlir::TypeAttr::get(value_attr.getType()),
          nullptr);
      op->setAttr(
          kTfSavedModelExportedNamesAttr,
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    }
  }
  AdjustBoundInputArgTypes(module);
  module->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(module);
  MarkSavedModelFunctionVisibility(module);
  return absl::OkStatus();
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelObjectGraph(
    SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions import_options) {
  LoadImporterDialects(*context);
  GraphDebugInfo dummy_debug_info;
  const GraphDebugInfo& debug_info =
      saved_model->debug_info() ? *saved_model->debug_info() : dummy_debug_info;
  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.unconditionally_use_set_output_shapes =
      import_options.unconditionally_use_set_output_shapes;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  const auto& graphdef = saved_model->meta_graph_def().graph_def();
  PopulateTfVersions(module.get(), graphdef.versions());
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = import_options.add_default_attributes;
  Graph graph(OpRegistry::Global());
  GraphDef preprocessed_graphdef(graphdef);
  if (import_options.add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(nullptr, &preprocessed_graphdef));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  NameUniquifier function_name_uniquifier(graph.flib_def());
  for (const auto& fn_name : graph.flib_def().ListFunctionNames()) {
    std::string mlir_func_name(function_name_uniquifier.GetUniqueName(fn_name));
    (tf_name_to_mlir_name)[std::string(fn_name)] = mlir_func_name;
  }
  specs.convert_all_functions_to_mlir = true;
  TF_ASSIGN_OR_RETURN(
      module, ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                                 module->getContext()));
  if (!saved_model->meta_graph_def().has_object_graph_def()) {
    return errors::InvalidArgument(
        "SavedModel does not have an object graph. Please use TF2.");
  }
  auto& object_graph = saved_model->meta_graph_def().object_graph_def();
  ObjectNames object_names(object_graph, exported_names);
  for (auto func :
       llvm::make_early_inc_range(module->getOps<mlir::func::FuncOp>())) {
    if (func.getName().starts_with("__inference__traced_save_") ||
        func.getName().starts_with("__inference__traced_restore_") ||
        func.getName().starts_with("__inference_signature_wrapper_") ||
        func.getName().starts_with("main")) {
      func.erase();
    }
  }
  TF_RETURN_IF_ERROR(
      DiagnoseMultipleConcreteFunctions(object_graph, object_names));
  TF_RETURN_IF_ERROR(CreateSavedModelIR(object_names, module.get(),
                                        object_graph, tf_name_to_mlir_name,
                                        saved_model, import_options));
  assert(mlir::succeeded(mlir::verify(module.get())));
  return module;
}
class SimpleSavedModelMLIRImportInput : public SavedModelMLIRImportInput {
 public:
  static absl::StatusOr<SimpleSavedModelMLIRImportInput> Create(
      const MLIRImportOptions& import_options,
      const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info) {
    DCHECK(meta_graph_def);
    GraphDef graph_def(meta_graph_def->graph_def());
    auto graph = std::make_unique<Graph>(OpRegistry::Global());
    if (import_options.upgrade_legacy) {
      TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
          graph_def, graph->flib_def().default_registry()));
    }
    GraphConstructorOptions graph_ctor_options;
    graph_ctor_options.allow_internal_ops = true;
    graph_ctor_options.add_default_attributes = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        graph_ctor_options, std::move(graph_def), graph.get()));
    if (import_options.upgrade_legacy) {
      TF_RETURN_IF_ERROR(UpgradeLegacyGraph(
          graph.get(),
          const_cast<FunctionLibraryDefinition*>(&graph->flib_def()),
          false));
    }
    return SimpleSavedModelMLIRImportInput(meta_graph_def, debug_info,
                                           std::move(graph));
  }
  SimpleSavedModelMLIRImportInput(const MetaGraphDef* meta_graph_def,
                                  const GraphDebugInfo& debug_info,
                                  std::unique_ptr<Graph> graph)
      : SavedModelMLIRImportInput(meta_graph_def, debug_info),
        graph_(std::move(graph)) {}
  absl::StatusOr<const Graph*> GetSubGraph(absl::string_view name,
                                           GraphImportConfig& specs) override {
    DCHECK(CheckGraphNameValidity(name));
    DCHECK(CheckGraphContainsFeedsAndFetches(specs));
    return graph_.get();
  }
 private:
  bool CheckGraphContainsFeedsAndFetches(const GraphImportConfig& specs) const {
    absl::flat_hash_set<std::string> feed_fetch_nodes;
    for (const auto& iter : specs.inputs) {
      TensorId tensor_id = ParseTensorName(iter.first);
      feed_fetch_nodes.insert(std::string(tensor_id.node()));
    }
    for (const auto& output : llvm::concat<const std::string>(
             specs.outputs, specs.control_outputs)) {
      TensorId tensor_id = ParseTensorName(output);
      feed_fetch_nodes.insert(std::string(tensor_id.node()));
    }
    for (Node* node : graph_->op_nodes()) {
      feed_fetch_nodes.erase(node->name());
    }
    return feed_fetch_nodes.empty();
  }
  bool CheckGraphNameValidity(absl::string_view name) const {
    const auto& signature_defs = meta_graph_def().signature_def();
    if (signature_defs.contains(std::string(name))) return true;
    if (meta_graph_def().has_saver_def() &&
        meta_graph_def().saver_def().restore_op_name() == name)
      return true;
    std::string init_op_name;
    if (internal::GetInitOp("", meta_graph_def(), &init_op_name).ok()) {
      if (init_op_name == name) return true;
    }
    return false;
  }
  std::unique_ptr<Graph> graph_;
};
static absl::flat_hash_set<std::string> GetOriginalTfFuncNamesFromGraphDef(
    const GraphDef& graph_def) {
  absl::flat_hash_set<std::string> original_func_tf_names;
  for (const auto& function : graph_def.library().function()) {
    original_func_tf_names.insert(function.signature().name());
  }
  return original_func_tf_names;
}
class SavedModelSignatureDefImporterLite {
 public:
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      SavedModelMLIRImportInput& input,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, bool import_restore = true,
      bool unconditionally_use_set_output_shapes = false) {
    SavedModelSignatureDefImporterLite importer(
        input, exported_names, context, import_restore,
        unconditionally_use_set_output_shapes);
    return importer.ConvertSignatures();
  }
 private:
  SavedModelSignatureDefImporterLite(
      SavedModelMLIRImportInput& input,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, bool import_restore,
      bool unconditionally_use_set_output_shapes)
      : input_(input),
        original_func_tf_names_(GetOriginalTfFuncNamesFromGraphDef(
            input.meta_graph_def().graph_def())),
        exported_names_(exported_names),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(context))),
        symbol_table_(module_.get()),
        import_restore_(import_restore),
        unconditionally_use_set_output_shapes_(
            unconditionally_use_set_output_shapes) {}
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSignatures();
  Status ConvertSignature(const std::string& sig_def_key,
                          const SignatureDef& signature_def);
  struct AssetInfo {
    std::string tensor_name;
    mlir::tf_saved_model::AssetOp op;
  };
  absl::StatusOr<std::vector<AssetInfo>> ConvertAssets();
  Status ConvertInitializer(const std::string& target_node_name,
                            const std::vector<AssetInfo>& assets,
                            llvm::StringRef initializer_type);
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraph(
      const std::string& name,
      const std::vector<std::pair<std::string, TensorInfo>>& inputs,
      const std::vector<std::pair<std::string, TensorInfo>>& outputs,
      std::vector<std::string> control_outputs,
      std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);
  Status MoveConvertedFunctionsToModule(
      absl::string_view name, mlir::ModuleOp sub_module,
      const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);
  absl::StatusOr<GraphImportConfig::InputArrays> ParseInputArrays(
      llvm::ArrayRef<std::pair<std::string, TensorInfo>> inputs);
 private:
  SavedModelMLIRImportInput& input_;
  absl::flat_hash_set<std::string> original_func_tf_names_;
  std::optional<absl::Span<const std::string>> exported_names_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  absl::Mutex symbol_table_mu_;
  mlir::SymbolTable symbol_table_ ABSL_GUARDED_BY(symbol_table_mu_);
  bool import_restore_ = true;
  bool unconditionally_use_set_output_shapes_ = false;
};
absl::StatusOr<std::vector<SavedModelSignatureDefImporterLite::AssetInfo>>
SavedModelSignatureDefImporterLite::ConvertAssets() {
  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(
      internal::GetAssetFileDefs(input_.meta_graph_def(), &asset_file_defs));
  std::vector<AssetInfo> results;
  results.reserve(asset_file_defs.size());
  mlir::OpBuilder builder(module_->getBodyRegion());
  unsigned i = 0;  
  for (const auto& asset : asset_file_defs) {
    auto asset_op = builder.create<mlir::tf_saved_model::AssetOp>(
        module_->getLoc(),
        builder.getStringAttr(
            absl::StrCat("__tf_saved_model_asset", i++, "_", asset.filename())),
        builder.getStringAttr(
            io::JoinPath(kSavedModelAssetsDirectory, asset.filename())));
    results.push_back({asset.tensor_info().name(), asset_op});
  }
  return results;
}
Status SavedModelSignatureDefImporterLite::MoveConvertedFunctionsToModule(
    absl::string_view name, mlir::ModuleOp sub_module,
    const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name) {
  mlir::Builder builder(sub_module.getContext());
  mlir::SymbolTable sub_module_symbol_table(sub_module);
  absl::flat_hash_set<std::string> original_func_mlir_names;
  for (const auto& kv : tf_name_to_mlir_name) {
    if (original_func_tf_names_.contains(kv.first))
      original_func_mlir_names.insert(kv.second);
  }
  for (auto func : sub_module.getOps<mlir::func::FuncOp>()) {
    if (mlir::tf_saved_model::IsExported(func)) continue;
    if (original_func_mlir_names.count(func.getSymName().str())) continue;
    std::string new_sym_name = absl::StrCat(name, "/", func.getSymName().str());
    mlir::StringAttr new_sym_name_attr = builder.getStringAttr(new_sym_name);
    if (mlir::failed(sub_module_symbol_table.replaceAllSymbolUses(
            func, new_sym_name_attr, sub_module)))
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "SavedModelSignatureDefImporterLite: failed to assign a unique "
          "name to the private function used in a signature: ",
          func.getSymName().str()));
    mlir::SymbolTable::setSymbolName(func, new_sym_name);
  }
  for (auto func : sub_module.getOps<mlir::func::FuncOp>()) {
    absl::MutexLock l(&symbol_table_mu_);
    symbol_table_.insert(func.clone());
  }
  return absl::OkStatus();
}
Status SavedModelSignatureDefImporterLite::ConvertInitializer(
    const std::string& target_node_name, const std::vector<AssetInfo>& assets,
    llvm::StringRef initializer_type) {
  std::vector<std::pair<std::string, TensorInfo>> inputs;
  inputs.reserve(assets.size());
  for (const auto& asset : assets) {
    TensorInfo tensor_info;
    tensor_info.set_name(asset.tensor_name);
    tensor_info.set_dtype(DT_STRING);
    tensor_info.mutable_tensor_shape();
    inputs.push_back({asset.tensor_name, tensor_info});
  }
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  TF_ASSIGN_OR_RETURN(auto sub_module,
                      ConvertGraph(target_node_name, inputs, {},
                                   {target_node_name}, tf_name_to_mlir_name));
  mlir::SymbolTable sub_symbol_table(*sub_module);
  auto init_func_op =
      sub_symbol_table.lookup<mlir::func::FuncOp>(target_node_name);
  init_func_op->removeAttr("tf.entry_function");
  mlir::OpBuilder builder(module_->getBodyRegion());
  DCHECK_EQ(init_func_op.getNumArguments(), assets.size());
  for (const auto& iter : llvm::enumerate(assets)) {
    auto asset_op = iter.value().op;
    init_func_op.setArgAttr(
        iter.index(), "tf_saved_model.bound_input",
        mlir::SymbolRefAttr::get(builder.getContext(), asset_op.getName()));
  }
  init_func_op->setAttr(
      kTfSavedModelExportedNamesAttr,
      builder.getStrArrayAttr({absl::StrCat(
          "__tf_saved_model_session_initializer_", target_node_name)}));
  init_func_op->setAttr(kTfSavedModelInitializerTypeAttr,
                        builder.getStringAttr(initializer_type));
  return MoveConvertedFunctionsToModule(target_node_name, *sub_module,
                                        tf_name_to_mlir_name);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefImporterLite::ConvertGraph(
    const std::string& name,
    const std::vector<std::pair<std::string, TensorInfo>>& inputs,
    const std::vector<std::pair<std::string, TensorInfo>>& outputs,
    const std::vector<std::string> control_outputs,
    std::unordered_map<std::string, std::string>& tf_name_to_mlir_name) {
  VLOG(1) << "Importing Signature: " << name;
  GraphImportConfig specs;
  specs.graph_func_name = name;
  specs.prune_unused_nodes = true;
  TF_ASSIGN_OR_RETURN(specs.inputs, ParseInputArrays(inputs));
  for (auto& output : outputs) {
    TF_ASSIGN_OR_RETURN(std::string name,
                        GetDenseTensorNameFromTensorInfo(output.second));
    specs.outputs.push_back(std::move(name));
  }
  specs.control_outputs = control_outputs;
  specs.enable_shape_inference = false;
  specs.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes_;
  TF_ASSIGN_OR_RETURN(const auto* subgraph, input_.GetSubGraph(name, specs));
  return GraphDefImporter::Convert(module_->getContext(), *subgraph,
                                   input_.debug_info(), subgraph->flib_def(),
                                   specs, tf_name_to_mlir_name,
                                   true);
}
Status SavedModelSignatureDefImporterLite::ConvertSignature(
    const std::string& sig_def_key, const SignatureDef& signature_def) {
  std::vector<std::pair<std::string, TensorInfo>> inputs(
      signature_def.inputs().begin(), signature_def.inputs().end());
  llvm::sort(inputs, [](const auto& lhs, const auto& rhs) {
    return tensorflow::Fingerprint64(lhs.first) <
           tensorflow::Fingerprint64(rhs.first);
  });
  std::vector<std::pair<std::string, TensorInfo>> outputs(
      signature_def.outputs().begin(), signature_def.outputs().end());
  llvm::sort(outputs, [](const auto& lhs, const auto& rhs) {
    return tensorflow::Fingerprint64(lhs.first) <
           tensorflow::Fingerprint64(rhs.first);
  });
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  TF_ASSIGN_OR_RETURN(
      auto sub_module,
      ConvertGraph(sig_def_key, inputs, outputs, {}, tf_name_to_mlir_name));
  mlir::OpBuilder builder(sub_module->getBodyRegion());
  mlir::SymbolTable sub_symbol_table(*sub_module);
  auto func_op = sub_symbol_table.lookup<mlir::func::FuncOp>(sig_def_key);
  TF_RET_CHECK(func_op)
      << "Graphdef importer should have created a function named "
      << sig_def_key << ".";
  func_op->setAttr(kTfSavedModelExportedNamesAttr,
                   builder.getStrArrayAttr({sig_def_key}));
  for (const auto& input_and_idx : llvm::enumerate(inputs)) {
    func_op.setArgAttr(input_and_idx.index(), kTfSavedModelIndexPathAttr,
                       builder.getStrArrayAttr({input_and_idx.value().first}));
  }
  for (const auto& output_and_idx : llvm::enumerate(outputs)) {
    func_op.setResultAttr(
        output_and_idx.index(), kTfSavedModelIndexPathAttr,
        builder.getStrArrayAttr({output_and_idx.value().first}));
  }
  for (const auto& [tf_name, mlir_name] : tf_name_to_mlir_name) {
    auto func_op = sub_symbol_table.lookup<mlir::func::FuncOp>(mlir_name);
    TF_RET_CHECK(func_op)
        << "Graphdef importer should have created a function named "
        << mlir_name << ".";
    func_op->setAttr("tf._original_func_name", builder.getStringAttr(tf_name));
  }
  return MoveConvertedFunctionsToModule(sig_def_key, *sub_module,
                                        tf_name_to_mlir_name);
}
absl::StatusOr<GraphImportConfig::InputArrays>
SavedModelSignatureDefImporterLite::ParseInputArrays(
    llvm::ArrayRef<std::pair<std::string, TensorInfo>> inputs) {
  GraphImportConfig::InputArrays results;
  for (const auto& iter : inputs) {
    const auto& tensor_info = iter.second;
    TF_ASSIGN_OR_RETURN(std::string name,
                        GetDenseTensorNameFromTensorInfo(tensor_info));
    VLOG(1) << "Importing Signature Input: input_name = " << iter.first
            << ", tensor_info = " << tensor_info.DebugString();
    ArrayInfo array_info;
    array_info.imported_dtype = tensor_info.dtype();
    if (tensor_info.has_tensor_shape()) {
      array_info.shape = tensor_info.tensor_shape();
    } else {
      array_info.shape.set_unknown_rank(true);
    }
    results.insert(std::pair<std::string, ArrayInfo>(std::move(name),
                                                     std::move(array_info)));
  }
  return results;
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefImporterLite::ConvertSignatures() {
  LoadImporterDialects(*module_->getContext());
  const auto& signatures = input_.meta_graph_def().signature_def();
  PopulateTfVersions(module_.get(),
                     input_.meta_graph_def().graph_def().versions());
  llvm::DenseSet<llvm::StringRef> exported_name_set;
  bool import_all_signatures = !exported_names_.has_value();
  if (exported_names_.has_value()) {
    exported_name_set.insert(exported_names_->begin(), exported_names_->end());
  }
  absl::Mutex error_status_mu;  
  tensorflow::Status error_status;
  {
    thread::ThreadPool thread_pool(Env::Default(), "ConvertSignatures",
                                   kNumThreadToConvertSignatures);
    for (const auto& key_and_signature_def : signatures) {
      const std::string& sig_def_key = key_and_signature_def.first;
      const SignatureDef& signature_def = key_and_signature_def.second;
      if (sig_def_key == "__saved_model_init_op") {
        continue;
      }
      if (!import_all_signatures && exported_name_set.count(sig_def_key) == 0) {
        continue;
      }
      thread_pool.Schedule([&]() {
        auto status = ConvertSignature(sig_def_key, signature_def);
        if (!status.ok()) {
          absl::MutexLock l(&error_status_mu);
          error_status = std::move(status);
        }
      });
    }
  }
  TF_RETURN_IF_ERROR(error_status);
  TF_ASSIGN_OR_RETURN(auto assets, ConvertAssets());
  mlir::OpBuilder builder(module_->getBodyRegion());
  llvm::SmallVector<mlir::Attribute, 2> init_sym_refs;
  if (import_restore_ && input_.meta_graph_def().has_saver_def()) {
    std::vector<AssetInfo> variable_and_assets;
    auto variable_filename_op = builder.create<mlir::tf_saved_model::AssetOp>(
        module_->getLoc(),
        builder.getStringAttr("__tf_saved_model_variables"),
        builder.getStringAttr(io::JoinPath(kSavedModelVariablesDirectory,
                                           kSavedModelVariablesFilename)));
    variable_and_assets.push_back(
        {input_.meta_graph_def().saver_def().filename_tensor_name(),
         variable_filename_op});
    variable_and_assets.insert(variable_and_assets.end(), assets.begin(),
                               assets.end());
    const auto& restore_op_name =
        input_.meta_graph_def().saver_def().restore_op_name();
    TF_RETURN_IF_ERROR(ConvertInitializer(restore_op_name, variable_and_assets,
                                          kTfSavedModelInitializerRestoreType));
    init_sym_refs.push_back(
        mlir::SymbolRefAttr::get(builder.getContext(), restore_op_name));
  }
  std::string init_op_name;
  TF_RETURN_IF_ERROR(
      internal::GetInitOp("", input_.meta_graph_def(), &init_op_name));
  if (!init_op_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertInitializer(init_op_name, assets,
                                          kTfSavedModelInitializerInitType));
    init_sym_refs.push_back(
        mlir::SymbolRefAttr::get(builder.getContext(), init_op_name));
  }
  builder.create<mlir::tf_saved_model::SessionInitializerOp>(
      module_->getLoc(), builder.getArrayAttr(init_sym_refs));
  (*module_)->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(*module_);
  MarkSavedModelFunctionVisibility(*module_);
  return std::move(module_);
}
class SavedModelSignatureDefImporter {
 public:
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      const SavedModelBundle& bundle,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, tensorflow::MLIRImportOptions options) {
    GraphDebugInfo debug_info;
    if (bundle.debug_info != nullptr) debug_info = *bundle.debug_info;
    TF_ASSIGN_OR_RETURN(auto input,
                        SimpleSavedModelMLIRImportInput::Create(
                            options, &bundle.meta_graph_def, debug_info));
    TF_ASSIGN_OR_RETURN(auto module,
                        SavedModelSignatureDefImporterLite::Convert(
                            input, exported_names, context,
                            false));
    mlir::OpBuilder builder(module->getContext());
    (*module)->setAttr("tf_saved_model.under_construction",
                       builder.getUnitAttr());
    TF_RETURN_IF_ERROR(
        LiftVariables(bundle, *module, options.lift_variables,
                      options.include_variables_in_initializers));
    (*module)->removeAttr("tf_saved_model.under_construction");
    return module;
  }
 private:
  static Status LiftVariables(const SavedModelBundle& bundle,
                              mlir::ModuleOp module,
                              bool lift_varhandle_ops_to_args,
                              bool include_variables_in_initializers);
};
Status SavedModelSignatureDefImporter::LiftVariables(
    const SavedModelBundle& bundle, mlir::ModuleOp module,
    const bool lift_varhandle_ops_to_args,
    const bool include_variables_in_initializers) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  mlir::PassManager pm(module.getContext());
  SetCrashReproducer(pm);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  if (!include_variables_in_initializers) {
    pm.addPass(
        mlir::tf_saved_model::CreateRemoveVariablesInSessionInitializerPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::
          CreateConvertReadonlyReferenceVariablesToResourceVariablesPass());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("Failed to prepare to lift variables."));
  if (lift_varhandle_ops_to_args) {
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to prepare to mark initialized variables."));
    pm.clear();
    pm.addPass(mlir::TF::CreatePromoteVarHandlesToArgsPass());
    if (mlir::failed(pm.run(module)))
      return diag_handler.Combine(
          errors::Internal("Failed to promote var handles to args."));
    if (failed(
            mlir::tf_saved_model::LiftVariables(module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to lift variables."));
  } else {
    if (failed(mlir::tf_saved_model::InitializeVariablesInSessionInitializer(
            module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to initialize variables in session init."));
  }
  pm.clear();
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_saved_model::CreateDedupBoundInputBindingPass());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("Failed to dedup bound inputs."));
  return absl::OkStatus();
}
}  
SavedModelMLIRImportInput::~SavedModelMLIRImportInput() = default;
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const GraphImportConfig& specs, mlir::MLIRContext* context) {
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  Graph graph(OpRegistry::Global());
  GraphDef preprocessed_graphdef(graphdef);
  TF_RETURN_IF_ERROR(PreprocessGraphDef(&specs, &preprocessed_graphdef));
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
        preprocessed_graphdef, graph.flib_def().default_registry()));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  return ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                            context);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context) {
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(
        UpgradeLegacyGraph(const_cast<Graph*>(&graph),
                           const_cast<FunctionLibraryDefinition*>(&flib_def),
                           specs.restrict_functionalization_to_compiled_nodes));
  }
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  TF_ASSIGN_OR_RETURN(auto module, GraphDefImporter::Convert(
                                       context, graph, debug_info, flib_def,
                                       specs, tf_name_to_mlir_name));
  if (specs.set_original_tf_func_name) {
    mlir::Builder builder(module->getContext());
    mlir::SymbolTable symbol_table(*module);
    for (const auto& [tf_name, mlir_name] : tf_name_to_mlir_name) {
      auto func_op = symbol_table.lookup<mlir::func::FuncOp>(mlir_name);
      TF_RET_CHECK(func_op)
          << "Graphdef importer should have created a function named "
          << mlir_name << ".";
      func_op->setAttr("tf._original_func_name",
                       builder.getStringAttr(tf_name));
    }
  }
  return module;
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertFunctionToMlir(
    const FunctionBody* fbody, const FunctionLibraryDefinition& flib_def,
    mlir::MLIRContext* context) {
  tensorflow::GraphDebugInfo dummy_debug_info;
  tensorflow::GraphImportConfig specs;
  specs.graph_func_name = fbody->record->fdef().signature().name();
  specs.enable_shape_inference = false;
  specs.graph_as_function = true;
  for (const auto* control_ret_node : fbody->control_ret_nodes)
    specs.control_outputs.push_back(control_ret_node->name());
  return ConvertGraphToMlir(*fbody->graph, dummy_debug_info, flib_def, specs,
                            context);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, MLIRImportOptions options) {
  return ConvertSavedModelObjectGraph(saved_model, exported_names, context,
                                      options);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options) {
  std::optional<absl::Span<const std::string>> optional_exported_names;
  if (!exported_names.empty()) optional_exported_names = exported_names;
  return SavedModelSignatureDefImporter::Convert(
      saved_model, optional_exported_names, context, options);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    const MetaGraphDef& meta_graph_def, const GraphDebugInfo& debug_info,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options) {
  TF_ASSIGN_OR_RETURN(auto input, SimpleSavedModelMLIRImportInput::Create(
                                      options, &meta_graph_def, debug_info));
  return ConvertSavedModelV1ToMlirLite(
      input, exported_names, context,
      options.unconditionally_use_set_output_shapes);
}
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    SavedModelMLIRImportInput& input,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, bool unconditionally_use_set_output_shapes) {
  return SavedModelSignatureDefImporterLite::Convert(
      input, exported_names, context,
      true, unconditionally_use_set_output_shapes);
}
std::string MlirModuleToString(mlir::ModuleOp module,
                               mlir::OpPrintingFlags flags) {
  std::string txt_module;
  {
    llvm::raw_string_ostream os{txt_module};
    module.print(os, flags);
  }
  return txt_module;
}
std::string MlirModuleToString(mlir::ModuleOp module, bool show_debug_info) {
  mlir::OpPrintingFlags flags;
  if (show_debug_info) flags.enableDebugInfo();
  return MlirModuleToString(module, flags);
}
}  