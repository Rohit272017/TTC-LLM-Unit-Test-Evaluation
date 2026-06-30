#include "tensorflow/tools/proto_splitter/cc/saved_model_splitter.h"
#include <vector>
#include "absl/status/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/tools/proto_splitter/cc/graph_def_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/large_node_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
namespace tensorflow {
namespace tools::proto_splitter {
using namespace std::string_literals;  
absl::Status SavedModelSplitter::BuildChunks() {
  TF_RETURN_IF_ERROR(SetMessageAsBaseChunk());
  SavedModel* sm = tsl::protobuf::DynamicCastToGenerated<SavedModel>(message());
  int max_size = GetMaxSize();
  if (GetInitialSize() < max_size) return absl::OkStatus();
  std::vector<FieldType> fields_to_graph_def = {"meta_graphs"s, 0,
                                                "graph_def"s};
  GraphDefSplitter graph_def_splitter(
      sm->mutable_meta_graphs(0)->mutable_graph_def(), this,
      &fields_to_graph_def);
  TF_RETURN_IF_ERROR(graph_def_splitter.BuildChunks());
  if (sm->ByteSizeLong() < max_size) return absl::OkStatus();
  LargeNodeSplitter<GraphDef> entire_graph_splitter(
      sm->mutable_meta_graphs(0)->mutable_graph_def(), this,
      &fields_to_graph_def);
  int index = 1;
  entire_graph_splitter.SetChunkIndex(&index);
  TF_RETURN_IF_ERROR(entire_graph_splitter.BuildChunks());
  return absl::OkStatus();
}
}  
}  