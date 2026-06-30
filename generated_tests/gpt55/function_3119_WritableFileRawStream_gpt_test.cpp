#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;

class StringWritableFile : public WritableFile {
 public:
  explicit StringWritableFile(bool fail_on_append = false)
      : fail_on_append_(fail_on_append) {}

  Status Append(StringPiece data) override {
    if (fail_on_append_) {
      return errors::Internal("forced append failure");
    }
    data_.append(data.data(), data.size());
    return absl::OkStatus();
  }

  Status Close() override {
    closed_ = true;
    return absl::OkStatus();
  }

  Status Flush() override {
    flushed_ = true;
    return absl::OkStatus();
  }

  Status Sync() override {
    synced_ = true;
    return absl::OkStatus();
  }

  const std::string& data() const { return data_; }
  bool closed() const { return closed_; }
  bool flushed() const { return flushed_; }
  bool synced() const { return synced_; }

 private:
  bool fail_on_append_;
  bool closed_ = false;
  bool flushed_ = false;
  bool synced_ = false;
  std::string data_;
};

class DumpTextualIRToFileTest : public ::testing::Test {
 protected:
  MlirDumpConfig TfgConfig() {
    MlirDumpConfig config;
    config.dialect = MlirDumpConfig::Dialect::kTFG;
    return config;
  }

  std::unique_ptr<Graph> MakeEmptyGraph() {
    return std::make_unique<Graph>(OpRegistry::Global());
  }

  std::unique_ptr<Graph> MakeGraphWithSingleConstNode() {
    auto graph = std::make_unique<Graph>(OpRegistry::Global());

    Tensor value(DT_FLOAT, TensorShape({}));
    value.scalar<float>()() = 1.0f;

    Node* node = nullptr;
    TF_CHECK_OK(NodeBuilder("const_node", "Const")
                    .Attr("dtype", DT_FLOAT)
                    .Attr("value", value)
                    .Finalize(graph.get(), &node));

    return graph;
  }

  std::unique_ptr<Graph> MakeGraphWithPlaceholderAndIdentity() {
    auto graph = std::make_unique<Graph>(OpRegistry::Global());

    Node* placeholder = nullptr;
    TF_CHECK_OK(NodeBuilder("input", "Placeholder")
                    .Attr("dtype", DT_FLOAT)
                    .Finalize(graph.get(), &placeholder));

    Node* identity = nullptr;
    TF_CHECK_OK(NodeBuilder("identity", "Identity")
                    .Input(placeholder)
                    .Attr("T", DT_FLOAT)
                    .Finalize(graph.get(), &identity));

    return graph;
  }
};

TEST_F(DumpTextualIRToFileTest, EmptyGraphDumpsSuccessfully) {
  auto graph = MakeEmptyGraph();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("module"), std::string::npos);
}

TEST_F(DumpTextualIRToFileTest, GraphWithSingleConstNodeDumpsSuccessfully) {
  auto graph = MakeGraphWithSingleConstNode();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("const_node"), std::string::npos);
}

TEST_F(DumpTextualIRToFileTest, GraphWithPlaceholderAndIdentityDumpsSuccessfully) {
  auto graph = MakeGraphWithPlaceholderAndIdentity();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("input"), std::string::npos);
  EXPECT_NE(file.data().find("identity"), std::string::npos);
}

TEST_F(DumpTextualIRToFileTest, NullFunctionLibraryDefinitionUsesGraphLibrary) {
  auto graph = MakeGraphWithSingleConstNode();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
}

TEST_F(DumpTextualIRToFileTest, NonNullFunctionLibraryDefinitionUsesGraphLibrary) {
  auto graph = MakeGraphWithSingleConstNode();
  FunctionDefLibrary external_library;
  FunctionLibraryDefinition external_flib(OpRegistry::Global(),
                                          external_library);
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph, &external_flib,
                                      &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("const_node"), std::string::npos);
}

TEST_F(DumpTextualIRToFileTest, AppendFailureDoesNotCrashAndReturnsOkAfterConversion) {
  auto graph = MakeGraphWithSingleConstNode();
  StringWritableFile file(/*fail_on_append=*/true);

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_TRUE(file.data().empty());
}

TEST_F(DumpTextualIRToFileTest, NullWritableFileDoesNotCrash) {
  auto graph = MakeGraphWithSingleConstNode();

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr,
                                      /*file=*/nullptr);

  EXPECT_THAT(status, IsOk());
}

TEST_F(DumpTextualIRToFileTest, RepeatedDumpProducesStableOutputForSameGraph) {
  auto graph = MakeGraphWithPlaceholderAndIdentity();
  StringWritableFile file1;
  StringWritableFile file2;

  Status status1 = DumpTextualIRToFile(TfgConfig(), *graph,
                                       /*flib_def=*/nullptr, &file1);
  Status status2 = DumpTextualIRToFile(TfgConfig(), *graph,
                                       /*flib_def=*/nullptr, &file2);

  EXPECT_THAT(status1, IsOk());
  EXPECT_THAT(status2, IsOk());
  EXPECT_EQ(file1.data(), file2.data());
}

TEST_F(DumpTextualIRToFileTest, DumpDoesNotCloseFlushOrSyncWritableFile) {
  auto graph = MakeGraphWithSingleConstNode();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.closed());
  EXPECT_FALSE(file.flushed());
  EXPECT_FALSE(file.synced());
}

TEST_F(DumpTextualIRToFileTest, DumpLargeGraphSuccessfully) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());

  Tensor value(DT_INT32, TensorShape({}));
  value.scalar<int32>()() = 1;

  Node* previous = nullptr;
  TF_CHECK_OK(NodeBuilder("const_0", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", value)
                  .Finalize(graph.get(), &previous));

  for (int i = 1; i <= 64; ++i) {
    Node* next = nullptr;
    TF_CHECK_OK(NodeBuilder(absl::StrCat("identity_", i), "Identity")
                    .Input(previous)
                    .Attr("T", DT_INT32)
                    .Finalize(graph.get(), &next));
    previous = next;
  }

  StringWritableFile file;

  Status status = DumpTextualIRToFile(TfgConfig(), *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("const_0"), std::string::npos);
  EXPECT_NE(file.data().find("identity_1"), std::string::npos);
  EXPECT_NE(file.data().find("identity_64"), std::string::npos);
}

TEST_F(DumpTextualIRToFileTest, UseMlirForGraphDumpInstallsMlirGraphDumper) {
  MlirDumpConfig config = TfgConfig();

  UseMlirForGraphDump(config);

  auto graph = MakeGraphWithSingleConstNode();
  StringWritableFile file;

  Status status = DumpTextualIRToFile(config, *graph,
                                      /*flib_def=*/nullptr, &file);

  EXPECT_THAT(status, IsOk());
  EXPECT_FALSE(file.data().empty());
  EXPECT_NE(file.data().find("const_node"), std::string::npos);
}

}  // namespace
}  // namespace tensorflow