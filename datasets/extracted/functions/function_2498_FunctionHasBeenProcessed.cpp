#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/functionalize_cond.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/functionalize_while.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/status_macros.h"
#include "xla/union_find.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
using FuncMap = std::map<string, std::optional<string>>;
using FuncMapIter = std::map<string, std::optional<string>>::const_iterator;
bool FunctionHasBeenProcessed(FuncMapIter func_iter, const FuncMap* func_map) {
  return func_iter != func_map->end();
}
bool FunctionHasBeenModified(FuncMapIter func_iter) {
  return func_iter->second.has_value();
}
string GetNewFunctionName(
    const string& func_name, Node* n,
    AssociatedFunctionInfo::AssociatedFunctionType func_type,
    FunctionLibraryDefinition* fld) {
  return (
      func_type ==
              AssociatedFunctionInfo::AssociatedFunctionType::kSymbolicGradient
          ? fld->UniqueFunctionName(absl::StrCat(n->name(), "_f15n_"))
          : fld->UniqueFunctionName(absl::StrCat(func_name, "_f15n_")));
}
const string& GetMappedFunctionName(FuncMapIter func_iter) {
  DCHECK(func_iter->second.has_value());
  return func_iter->second.value();
}
void UpdateFunctionMap(FuncMap* func_map, const string& canonicalized_name,
                       const string& new_func_name, bool function_modified) {
  (*func_map)[canonicalized_name] =
      function_modified ? absl::make_optional(new_func_name) : std::nullopt;
}
Status AddFunctionDefToGraphLibrary(
    const string& func_name, const AssociatedFunctionInfo& associated_function,
    Graph* graph, FunctionLibraryDefinition* fld) {
  const OpRegistrationData* op_reg_data;
  if (graph->flib_def().LookUp(func_name, &op_reg_data).ok())
    return absl::OkStatus();
  const FunctionDef* new_fdef = fld->Find(func_name);
  DCHECK(new_fdef != nullptr);
  FunctionDefLibrary fdef_lib;
  *(fdef_lib.add_function()) = *new_fdef;
  return graph->AddFunctionLibrary(fdef_lib);
}
Status FunctionalizeControlFlowForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    FuncMap* func_map, bool* function_modified,
    const NodeFilter& node_filter = {});
Status FunctionalizeControlFlowForNodeAssociatedFunctions(
    FuncMap* func_map, Graph* graph, FunctionLibraryDefinition* fld,
    FunctionLibraryRuntime* flr, bool* any_function_modified,
    const NodeFilter& node_filter) {
  std::vector<std::pair<Node*, std::vector<AssociatedFunctionInfo>>>
      nodes_to_associated_functions;
  for (auto* n : graph->nodes()) {
    auto associated_functions = GetAssociatedFunctions(*n, fld);
    if (!associated_functions.empty()) {
      nodes_to_associated_functions.push_back({n, associated_functions});
    }
  }
  for (const auto& pair : nodes_to_associated_functions) {
    Node* n = pair.first;
    auto associated_functions = pair.second;
    for (auto& associated_function : associated_functions) {
      DCHECK(associated_function.type() !=
                 AssociatedFunctionInfo::kFunctionCallNode ||
             associated_functions.size() == 1);
      string func_name = associated_function.func_name();
      string canonicalized_name =
          Canonicalize(func_name, AttrSlice(&associated_function.attrs()));
      auto func_iter = func_map->find(canonicalized_name);
      string new_func_name;
      if (FunctionHasBeenProcessed(func_iter, func_map)) {
        if (FunctionHasBeenModified(func_iter)) {
          *any_function_modified = true;
          new_func_name = GetMappedFunctionName(func_iter);
          TF_RETURN_IF_ERROR(RewriteAssociatedFunction(
              graph, n, fld, associated_function, new_func_name));
        }
        continue;
      }
      bool function_modified = false;
      new_func_name =
          GetNewFunctionName(func_name, n, associated_function.type(), fld);
      TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
          func_name, new_func_name, associated_function.attrs(), fld, flr,
          func_map, &function_modified, node_filter));
      UpdateFunctionMap(func_map, canonicalized_name, new_func_name,
                        function_modified);
      if (function_modified) {
        *any_function_modified = true;
        TF_RETURN_IF_ERROR(AddFunctionDefToGraphLibrary(
            new_func_name, associated_function, graph, fld));
        TF_RETURN_IF_ERROR(RewriteAssociatedFunction(
            graph, n, fld, associated_function, new_func_name));
      }
    }
  }
  return absl::OkStatus();
}
Status FunctionalizeControlFlowForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    FuncMap* func_map, bool* function_modified, const NodeFilter& node_filter) {
  *function_modified = false;
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flr->Instantiate(func_name, AttrSlice(&attrs), &handle));
  Status ret_status = absl::OkStatus();
  auto cleanup_handle = gtl::MakeCleanup([&]() {
    auto s = flr->ReleaseHandle(handle);
    if (!s.ok()) {
      ret_status.Update(s);
    }
  });
  const FunctionBody* body = flr->GetFunctionBody(handle);
  Graph* g = body->graph;
  bool has_switch_or_merge = false;
  for (Node* n : body->graph->nodes()) {
    if (node_filter && !node_filter(n)) continue;
    if (n->type_string() == "Switch" || n->type_string() == "Merge") {
      has_switch_or_merge = true;
      break;
    }
  }
  TF_RETURN_IF_ERROR(FunctionalizeControlFlowForNodeAssociatedFunctions(
      func_map, g, fld, flr, function_modified, node_filter));
  if (has_switch_or_merge) {
    *function_modified = true;
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile(
          absl::StrCat("functionalize_control_flow_before_fdef_", func_name),
          *g, fld);
    }
    TF_RETURN_IF_ERROR(FunctionalizeControlFlow(g, fld, node_filter));
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile(
          absl::StrCat("functionalize_control_flow_after_fdef_", func_name), *g,
          fld);
    }
  }
  if (*function_modified) {
    FunctionDef functionalized_fdef;
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*g, new_func_name, &functionalized_fdef));
    if (func_name == new_func_name) {
      VLOG(2) << "Replacing function " << func_name;
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(new_func_name, functionalized_fdef));
    } else {
      VLOG(2) << "Adding function " << new_func_name;
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(functionalized_fdef));
    }
  }
  return ret_status;
}
Status FunctionalizeControlFlow(Graph* graph,
                                FunctionLibraryDefinition* library,
                                const NodeFilter& node_filter,
                                bool include_functions) {
  VLOG(2) << "FunctionalizeControlFlow (initial): "
          << DumpGraphToFile("functionalize_initial", *graph, library);
  if (include_functions) {
    auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
        nullptr, tensorflow::Env::Default(),
        nullptr, TF_GRAPH_DEF_VERSION, library,
        tensorflow::OptimizerOptions());
    FunctionLibraryRuntime* flr =
        pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
    FuncMap func_map;
    bool modified = false;
    TF_RETURN_IF_ERROR(FunctionalizeControlFlowForNodeAssociatedFunctions(
        &func_map, graph, library, flr, &modified, node_filter));
  }
  TF_RETURN_IF_ERROR(FunctionalizeWhileLoop(graph, library, node_filter));
  TF_RETURN_IF_ERROR(FunctionalizeCond(graph, library, node_filter));
  VLOG(2) << "FunctionalizeControlFlow (final): "
          << DumpGraphToFile("functionalize_final", *graph, library);
  return absl::OkStatus();
}
Status FunctionalizeControlFlowForGraphDef(GraphDef* graph_def,
                                           FunctionLibraryDefinition* library,
                                           const NodeFilter& node_filter,
                                           bool include_functions) {
  FunctionDefLibrary function_lib = graph_def->library();
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph({}, *graph_def, &graph));
  TF_RETURN_IF_ERROR(FunctionalizeControlFlow(&graph, library, node_filter,
                                              include_functions));
  graph.ToGraphDef(graph_def);
  std::swap(*graph_def->mutable_library(), function_lib);
  return absl::OkStatus();
}
Status FunctionalizeControlFlowForXlaPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("functionalize_control_flow_before", *graph,
                    options.flib_def);
  }
  const auto* config = &options.session_options->config;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          nullptr, options.session_options->env, config,
          TF_GRAPH_DEF_VERSION, options.flib_def,
          config->graph_options().optimizer_options()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  static std::map<string, string>* kNodeTypeToFunctionAttrMapping =
      new std::map<string, string>{
          {"_TPUReplicate", "computation"},
          {"XlaLaunch", "function"},
      };
  FuncMap func_map;
  bool fld_modified = false;
  for (Node* n : graph->nodes()) {
    auto it = kNodeTypeToFunctionAttrMapping->find(n->type_string());
    if (it == kNodeTypeToFunctionAttrMapping->end()) {
      continue;
    }
    const string func_attr = it->second;
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), func_attr, &func));
    VLOG(2) << "Graph has node " << n->type_string()
            << ". Corresponding function: " << func.name();
    string new_func_name = options.flib_def->UniqueFunctionName(
        absl::StrCat(func.name(), "_f15n_"));
    bool modified;
    TF_RETURN_IF_ERROR(FunctionalizeControlFlowForFunction(
        func.name(), new_func_name, func.attr(), options.flib_def, flr,
        &func_map, &modified));
    if (modified) {
      n->ClearAttr(func_attr);
      func.set_name(new_func_name);
      n->AddAttr(func_attr, func);
      fld_modified = true;
    }
  }
  if (false) {
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile("functionalize_control_flow_before_prune", *graph,
                      options.flib_def);
    }
    TF_RETURN_IF_ERROR(
        PruneUnreachableFunctionsFromGraph(*graph, options.flib_def));
  }
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("functionalize_control_flow_after", *graph,
                    options.flib_def);
  }
  return absl::OkStatus();
}
}  