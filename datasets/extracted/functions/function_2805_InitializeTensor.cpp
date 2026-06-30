#include "tensorflow/core/grappler/grappler_item_builder.h"
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variable.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session_options.h"
namespace tensorflow {
namespace grappler {
namespace {
void InitializeTensor(DataType type, Tensor* tensor) {
  const int period = 7;
  if (type == DT_FLOAT) {
    auto flat = tensor->flat<float>();
    for (int i = 0; i < flat.size(); i++) {
      flat(i) = static_cast<float>(i % period) / 10.0f;
    }
  } else if (type == DT_INT64) {
    auto flat = tensor->flat<int64_t>();
    for (int i = 0; i < flat.size(); i++) {
      flat(i) = i % period;
    }
  } else if (type != DT_STRING && type != DT_RESOURCE && type != DT_VARIANT) {
    memset(const_cast<char*>(tensor->tensor_data().data()), 0,
           tensor->tensor_data().size());
  }
}
Status PruneGraph(GrapplerItem* item) {
  ModelPruner pruner;
  GraphDef pruned_graph;
  Cluster* cluster = nullptr;  
  TF_RETURN_IF_ERROR(pruner.Optimize(cluster, *item, &pruned_graph));
  item->graph = std::move(pruned_graph);
  return absl::OkStatus();
}
Status ReplaceUnknownShapeDim(const ItemConfig& cfg,
                              const TensorShapeProto& shape_pb_in,
                              TensorShapeProto* shape_pb_out,
                              TensorShape* shape_out) {
  std::vector<int32> dims;
  for (const auto& dim_proto : shape_pb_in.dim()) {
    if (cfg.placeholder_unknown_output_shape_dim >= 0 &&
        dim_proto.size() == -1) {
      dims.push_back(cfg.placeholder_unknown_output_shape_dim);
      shape_pb_out->add_dim()->set_size(
          cfg.placeholder_unknown_output_shape_dim);
    } else {
      dims.push_back(std::max<int32>(1, dim_proto.size()));
      shape_pb_out->add_dim()->set_size(dim_proto.size());
    }
  }
  return TensorShapeUtils::MakeShape(dims.data(), dims.size(), shape_out);
}
Status UpdatePlaceholderShape(
    const ItemConfig& cfg,
    const std::unordered_set<string>& signature_feed_nodes,
    GrapplerItem* new_item, NodeDef* node) {
  if (node->attr().count("dtype") == 0) {
    return absl::InternalError(absl::StrCat("Unknown type for placeholder ",
                                            node->name(),
                                            ", skipping this input"));
  }
  DataType type = node->attr().at("dtype").type();
  if (node->attr().count("shape") == 0) {
    return absl::InternalError(absl::StrCat("Unknown shape for placeholder ",
                                            node->name(),
                                            ", skipping this input"));
  }
  TensorShape shape;
  TensorShapeProto shape_proto;
  Status make_shape_status = ReplaceUnknownShapeDim(
      cfg, node->attr().at("shape").shape(), &shape_proto, &shape);
  if (!make_shape_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Invalid shape for placeholder ", node->name(), ": ",
                     make_shape_status.ToString(), ", skipping this input"));
  }
  if ((cfg.placeholder_unknown_output_shape_dim >= 0) && (shape.dims() == 0) &&
      (node->attr().count("_output_shapes") == 1)) {
    const auto& output_shapes =
        node->attr().at("_output_shapes").list().shape(0);
    if (output_shapes.dim_size() != 0) {
      shape.Clear();
      shape_proto.clear_dim();
      for (const auto& dim : output_shapes.dim()) {
        auto size = dim.size();
        if (size == -1) size = cfg.placeholder_unknown_output_shape_dim;
        TF_RETURN_IF_ERROR(shape.AddDimWithStatus(size));
        shape_proto.add_dim()->set_size(size);
      }
    }
  }
  Tensor fake_input(type, shape);
  InitializeTensor(type, &fake_input);
  if (cfg.feed_nodes.empty()) {
    if (signature_feed_nodes.count(node->name()) == 0) {
      new_item->feed.emplace_back(node->name(), fake_input);
    }
  } else if (cfg.feed_nodes.count(node->name()) > 0) {
    auto it = find_if(new_item->feed.begin(), new_item->feed.end(),
                      [&node](std::pair<string, Tensor>& f) {
                        return f.first == node->name();
                      });
    DCHECK(it != new_item->feed.end());
    it->second = fake_input;
  }
  if (!shape_proto.dim().empty())
    *(node->mutable_attr()->at("shape").mutable_shape()) = shape_proto;
  return absl::OkStatus();
}
}  
Status RuntimeGraphOptimizer(const GraphDef& graph_def_arg,
                             GraphDef* output_graph_def,
                             const ItemConfig& cfg) {
  if (!cfg.apply_optimizations && !cfg.inline_functions &&
      !cfg.erase_noinline_attributes) {
    if (output_graph_def != &graph_def_arg) {
      *output_graph_def = graph_def_arg;
    }
    return absl::OkStatus();
  }
  SessionOptions options;
  GraphDef graph_def(graph_def_arg);
  if (cfg.erase_noinline_attributes) {
    for (auto& func : *graph_def.mutable_library()->mutable_function()) {
      func.mutable_attr()->erase("_noinline");
    }
  }
  std::vector<std::unique_ptr<Device>> devices;
  DeviceFactory* cpu_factory = DeviceFactory::GetFactory("CPU");
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  Device* cpu_device = devices[0].get();
  auto dvc_mgr = std::make_unique<StaticDeviceMgr>(std::move(devices));
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph_def.library());
  Env* env = Env::Default();
  OptimizerOptions* optimizer_opts =
      options.config.mutable_graph_options()->mutable_optimizer_options();
  if (cfg.apply_optimizations) {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions::L1);
  } else {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions::L0);
  }
  optimizer_opts->set_do_function_inlining(cfg.inline_functions);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(dvc_mgr.get(), env, &options.config,
                                        graph_def.versions().producer(),
                                        &function_library, *optimizer_opts));
  FunctionLibraryRuntime* flr = pflr->GetFLR(cpu_device->name());
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  std::unique_ptr<Graph> graphptr(new Graph(function_library));
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      graph_ctor_opts, std::move(graph_def), graphptr.get()));
  ::tensorflow::GraphOptimizer optimizer(*optimizer_opts);
  optimizer.Optimize(flr, env, cpu_device, &graphptr,
                     tensorflow::GraphOptimizer::Options());
  graphptr->ToGraphDef(output_graph_def);
  return AddDefaultAttrsToGraphDef(output_graph_def, *graphptr->op_registry(),
                                   0, true);
}
std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDef(
    const string& id, const MetaGraphDef& meta_graph, const ItemConfig& cfg) {
  if (id.empty()) {
    LOG(ERROR) << "id must be non-empty.";
    return nullptr;
  }
  std::unique_ptr<GrapplerItem> new_item(new GrapplerItem());
  new_item->id = id;
  new_item->graph = meta_graph.graph_def();
  for (const auto& feed_node : cfg.feed_nodes) {
    const string feed_name = NodeName(feed_node);
    new_item->feed.emplace_back(feed_name, Tensor());
  }
  for (const auto& fetch_node : cfg.fetch_nodes) {
    new_item->fetch.emplace_back(NodeName(fetch_node));
  }
  if (new_item->fetch.empty() &&
      meta_graph.collection_def().count("train_op") > 0) {
    const CollectionDef& nodes = meta_graph.collection_def().at("train_op");
    if (nodes.has_node_list()) {
      for (const auto& node : nodes.node_list().value()) {
        new_item->fetch.push_back(NodeName(node));
      }
    }
  }
  std::unordered_set<string> signature_feed_nodes;
  std::unordered_set<string> signature_fetch_nodes;
  for (const auto& name_and_signature : meta_graph.signature_def()) {
    for (const auto& name_and_input : name_and_signature.second.inputs()) {
      const TensorInfo& input = name_and_input.second;
      if (input.has_coo_sparse()) {
        int64_t dim = std::max(1, cfg.placeholder_unknown_output_shape_dim);
        TensorShape shape_1d({dim});
        TensorShape shape_2d({dim, dim});
        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().values_tensor_name()))) {
          Tensor value_tensor(input.dtype(), shape_1d);
          InitializeTensor(input.dtype(), &value_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().values_tensor_name()), value_tensor);
        }
        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().indices_tensor_name()))) {
          Tensor indices_tensor(DT_INT64, shape_2d);
          InitializeTensor(input.dtype(), &indices_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().indices_tensor_name()),
              indices_tensor);
        }
        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().dense_shape_tensor_name()))) {
          Tensor dense_shape_tensor(DT_INT64, shape_1d);
          InitializeTensor(input.dtype(), &dense_shape_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().dense_shape_tensor_name()),
              dense_shape_tensor);
        }
      } else {
        if (gtl::InsertIfNotPresent(&signature_feed_nodes,
                                    NodeName(input.name()))) {
          TensorShape shape;
          TensorShapeProto shape_proto;
          Status s = ReplaceUnknownShapeDim(cfg, input.tensor_shape(),
                                            &shape_proto, &shape);
          if (!s.ok()) {
            LOG(ERROR) << "Invalid shape for signature input " << input.name()
                       << ": " << s << ", skipping this input";
            return nullptr;
          }
          Tensor fake_input(input.dtype(), shape);
          InitializeTensor(input.dtype(), &fake_input);
          new_item->feed.emplace_back(NodeName(input.name()), fake_input);
        }
      }
    }
    for (const auto& name_and_output : name_and_signature.second.outputs()) {
      const TensorInfo& output = name_and_output.second;
      if (output.has_coo_sparse()) {
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().values_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().values_tensor_name()));
        }
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().indices_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().indices_tensor_name()));
        }
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().dense_shape_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().dense_shape_tensor_name()));
        }
      } else {
        if (gtl::InsertIfNotPresent(&signature_fetch_nodes,
                                    NodeName(output.name()))) {
          new_item->fetch.push_back(NodeName(output.name()));
        }
      }
    }
  }
  for (const auto& feed : new_item->feed) {
    if (feed.first.empty()) {
      LOG(ERROR) << "Invalid feed node name skipping this input";
      return nullptr;
    } else {
      VLOG(1) << "Will use feed node " << feed.first;
    }
  }
  for (const auto& fetch : new_item->fetch) {
    if (fetch.empty()) {
      LOG(ERROR) << "Invalid fetch node name skipping this input";
      return nullptr;
    } else {
      VLOG(1) << "Will use fetch node " << fetch;
    }
  }
  if (new_item->fetch.empty()) {
    LOG(ERROR) << "Failed to detect the fetch node(s), skipping this input";
    return nullptr;
  }
  for (const string& var_collection :
       {"variables", "local_variables", "model_variables",
        "trainable_variables"}) {
    if (meta_graph.collection_def().count(var_collection) == 0) {
      continue;
    }
    const CollectionDef& vars = meta_graph.collection_def().at(var_collection);
    for (const auto& raw_var : vars.bytes_list().value()) {
      VariableDef var;
      var.ParseFromString(raw_var);
      if (!var.initializer_name().empty()) {
        new_item->init_ops.push_back(NodeName(var.initializer_name()));
      }
    }
  }
  if (meta_graph.collection_def().count("table_initializer") > 0) {
    const CollectionDef& inits =
        meta_graph.collection_def().at("table_initializer");
    if (inits.has_node_list()) {
      for (const auto& node : inits.node_list().value()) {
        new_item->init_ops.push_back(NodeName(node));
        new_item->expected_init_time += 30 * 60;
      }
    }
  }
  std::unordered_map<string, string> asset_node_to_value;
  if (!cfg.assets_directory_override.empty()) {
    if (meta_graph.collection_def().count("saved_model_assets") > 0) {
      const CollectionDef& collection =
          meta_graph.collection_def().at("saved_model_assets");
      const auto& any_assets = collection.any_list().value();
      if (!any_assets.empty()) {
        if (std::is_base_of<protobuf::Message, AssetFileDef>()) {
          for (const auto& any_asset : any_assets) {
            AssetFileDef asset_file_def;
            if (!ParseAny(any_asset, &asset_file_def, "tensorflow.AssetFileDef")
                     .ok()) {
              LOG(ERROR) << "Failed to parse AssetFile.";
              continue;
            }
            string asset_filepath = io::JoinPath(cfg.assets_directory_override,
                                                 asset_file_def.filename());
            if (!FilesExist({asset_filepath}, nullptr)) {
              LOG(ERROR) << "Can't access one or more of the asset files "
                         << asset_filepath << ", skipping this input";
              return nullptr;
            }
            asset_node_to_value[NodeName(asset_file_def.tensor_info().name())] =
                asset_filepath;
          }
        } else {
          LOG(ERROR) << "Can't parse AssetFileDef when using lite protos.";
          return nullptr;
        }
      }
    }
  } else if (meta_graph.collection_def().count("asset_filepaths") > 0) {
    const CollectionDef& file_paths =
        meta_graph.collection_def().at("asset_filepaths");
    std::vector<string> paths;
    for (const auto& raw_path : file_paths.bytes_list().value()) {
      paths.push_back(raw_path);
    }
    if (!FilesExist(paths, nullptr)) {
      LOG(ERROR) << "Can't access one or more of the asset files, skipping "
                    "this input";
      return nullptr;
    }
  }
  if (meta_graph.collection_def().count("queue_runners") > 0) {
    const CollectionDef& vars = meta_graph.collection_def().at("queue_runners");
    for (const auto& raw : vars.bytes_list().value()) {
      QueueRunnerDef queue_runner;
      if (!queue_runner.ParseFromString(raw)) {
        LOG(ERROR) << "Could not parse queue_runners, skipping this input";
        return nullptr;
      }
      if (queue_runner.cancel_op_name().empty()) {
        LOG(ERROR) << "Queue without a cancel op, skipping this input";
        return nullptr;
      }
      new_item->queue_runners.push_back(queue_runner);
    }
  }
  for (const auto& col : meta_graph.collection_def()) {
    const CollectionDef& collection = col.second;
    for (const string& node : collection.node_list().value()) {
      new_item->keep_ops.push_back(NodeName(node));
    }
  }
  for (auto& node : *new_item->graph.mutable_node()) {
    if (IsPlaceholder(node) && node.op() != "PlaceholderWithDefault") {
      Status s = UpdatePlaceholderShape(cfg, signature_feed_nodes,
                                        new_item.get(), &node);
      if (!s.ok()) return nullptr;
    } else if (IsConstant(node)) {
      auto it = asset_node_to_value.find(node.name());
      if (it != asset_node_to_value.end()) {
        auto iter = node.mutable_attr()->find("value");
        if (iter == node.attr().end()) {
          LOG(ERROR) << "Value attribute expected in const op for asset files";
          return nullptr;
        }
        if (!iter->second.has_tensor() ||
            iter->second.tensor().string_val_size() != 1) {
          LOG(INFO) << "Unexpected AttrValue proto: "
                    << iter->second.DebugString();
          return nullptr;
        }
        LOG(INFO) << "Using asset file " << it->second << " for node "
                  << node.name();
        *(iter->second.mutable_tensor()->mutable_string_val(0)) = it->second;
      }
    }
    node.mutable_attr()->erase("_output_shapes");
    if (cfg.ignore_user_placement) {
      node.clear_device();
    }
    if (cfg.ignore_colocation) {
      auto attr = node.mutable_attr();
      auto it = attr->find("_class");
      if (it != attr->end()) {
        attr->erase(it);
      }
    }
  }
  if (meta_graph.collection_def().count("savers") > 0) {
    const CollectionDef& savers = meta_graph.collection_def().at("savers");
    for (const auto& raw : savers.bytes_list().value()) {
      SaverDef saver;
      if (!saver.ParseFromString(raw)) {
        continue;
      }
      if (saver.filename_tensor_name().empty()) {
        continue;
      }
      new_item->save_op = saver.save_tensor_name();
      new_item->restore_op = saver.restore_op_name();
      new_item->save_restore_loc_tensor = saver.filename_tensor_name();
      break;
    }
  } else {
    const SaverDef& saver = meta_graph.saver_def();
    new_item->save_op = saver.save_tensor_name();
    new_item->restore_op = saver.restore_op_name();
    new_item->save_restore_loc_tensor = saver.filename_tensor_name();
  }
  Status attr_status = AddDefaultAttrsToGraphDef(
      &new_item->graph,
      FunctionLibraryDefinition(OpRegistry::Global(),
                                new_item->graph.library()),
      0, true);
  if (!attr_status.ok()) {
    LOG(ERROR) << "Failed to instantiate default attribute values: "
               << attr_status.message();
    return nullptr;
  }
  VLOG(1) << "Number of nodes in graph before RuntimeGraphOptimizer: "
          << new_item->graph.node_size();
  Status optimize_status =
      RuntimeGraphOptimizer(new_item->graph, &new_item->graph, cfg);
  if (!optimize_status.ok()) {
    LOG(ERROR) << "Graph preprocessing failed: " << optimize_status;
    return nullptr;
  }
  VLOG(1) << "Number of nodes in graph after RuntimeGraphOptimizer: "
          << new_item->graph.node_size();
  if (cfg.prune_graph) {
    VLOG(1) << "Pruning graph...";
    auto status = PruneGraph(new_item.get());
    if (!status.ok()) {
      LOG(ERROR) << "Pruning failed: " << status.message();
      return nullptr;
    }
    VLOG(1) << "Number of nodes in graph after pruning: "
            << new_item->graph.node_size();
  }
  std::unordered_set<string> nodes;
  for (const auto& node : new_item->graph.node()) {
    nodes.insert(node.name());
  }
  for (const auto& feed : new_item->feed) {
    if (nodes.find(feed.first) == nodes.end()) {
      LOG(ERROR) << "Feed node " << feed.first << " doesn't exist in graph";
      return nullptr;
    }
  }
  for (const auto& fetch : new_item->fetch) {
    if (nodes.find(fetch) == nodes.end()) {
      LOG(ERROR) << "Fetch node " << fetch << " doesn't exist in graph";
      return nullptr;
    }
  }
  for (const auto& init : new_item->init_ops) {
    if (nodes.find(init) == nodes.end()) {
      LOG(ERROR) << "Init node " << init << " doesn't exist in graph";
      return nullptr;
    }
  }
  return new_item;
}
std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDefFile(
    const string& id, const string& meta_graph_file, const ItemConfig& cfg) {
  MetaGraphDef meta_graph;
  if (!ReadMetaGraphDefFromFile(meta_graph_file, &meta_graph).ok()) {
    LOG(ERROR) << "Failed to read " << meta_graph_file;
    return nullptr;
  }
  return GrapplerItemFromMetaGraphDef(id, meta_graph, cfg);
}
}  
}  