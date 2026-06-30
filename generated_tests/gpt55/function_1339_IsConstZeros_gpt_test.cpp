#include "tensorflow/lite/delegates/gpu/common/transformations/make_padding.h"

#include <gtest/gtest.h>

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

Value* AddTensor(GraphFloat32* graph, const std::string& name,
                 const BHWC& shape) {
  Value* value = graph->NewValue();
  value->tensor.type = DataType::FLOAT32;
  value->tensor.shape = shape;
  value->tensor.ref = -1;
  value->tensor.name = name;
  return value;
}

Node* AddConstNode(GraphFloat32* graph, const std::string& name,
                   Value* output, const BHWC& shape,
                   const std::vector<float>& data) {
  Node* node = graph->NewNode();
  node->operation.type = ToString(OperationType::CONSTANT);
  node->operation.attributes =
      ConstTensorAttributes{Tensor<Linear, DataType::FLOAT32>(
          Linear(data.size()), data)};
  node->operation.attributes =
      ConstTensorAttributes{Tensor<BHWC, DataType::FLOAT32>(shape, data)};
  node->operation.name = name;
  graph->AddConsumer(node->id, output->id);
  return node;
}

Node* AddConcatNode(GraphFloat32* graph, const std::string& name,
                    Value* input0, Value* input1, Value* output, Axis axis) {
  Node* node = graph->NewNode();
  ConcatAttributes attr;
  attr.axis = axis;
  node->operation.type = ToString(OperationType::CONCAT);
  node->operation.attributes = attr;
  node->operation.name = name;

  graph->AddConsumer(node->id, input0->id);
  graph->AddConsumer(node->id, input1->id);
  graph->SetProducer(node->id, output->id);
  return node;
}

Node* AddIdentityProducer(GraphFloat32* graph, const std::string& name,
                          Value* output) {
  Node* node = graph->NewNode();
  node->operation.type = ToString(OperationType::RESHAPE);
  node->operation.name = name;
  graph->SetProducer(node->id, output->id);
  return node;
}

TEST(MakePaddingFromConcatTest, NewTransformationReturnsNonNull) {
  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  ASSERT_NE(transformation, nullptr);
}

TEST(MakePaddingFromConcatTest, NonConcatNodeIsSkipped) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  node->operation.type = ToString(OperationType::ADD);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(node, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
  EXPECT_EQ(result.message, "");
}

TEST(MakePaddingFromConcatTest, ConcatWithNoInputsIsSkipped) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;
  node->operation.type = ToString(OperationType::CONCAT);
  node->operation.attributes = attr;

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(node, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
}

TEST(MakePaddingFromConcatTest, ConcatWithOneInputIsSkipped) {
  GraphFloat32 graph;
  Value* input = AddTensor(&graph, "input", BHWC(1, 2, 3, 4));
  Node* node = graph.NewNode();
  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;
  node->operation.type = ToString(OperationType::CONCAT);
  node->operation.attributes = attr;
  graph.AddConsumer(node->id, input->id);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(node, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
}

TEST(MakePaddingFromConcatTest, ConcatWithThreeInputsIsSkipped) {
  GraphFloat32 graph;
  Value* input0 = AddTensor(&graph, "input0", BHWC(1, 2, 3, 4));
  Value* input1 = AddTensor(&graph, "input1", BHWC(1, 2, 3, 4));
  Value* input2 = AddTensor(&graph, "input2", BHWC(1, 2, 3, 4));

  Node* node = graph.NewNode();
  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;
  node->operation.type = ToString(OperationType::CONCAT);
  node->operation.attributes = attr;
  graph.AddConsumer(node->id, input0->id);
  graph.AddConsumer(node->id, input1->id);
  graph.AddConsumer(node->id, input2->id);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(node, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
}

TEST(MakePaddingFromConcatTest, ConcatWithNoConstZeroInputIsSkipped) {
  GraphFloat32 graph;
  Value* input0 = AddTensor(&graph, "input0", BHWC(1, 2, 3, 4));
  Value* input1 = AddTensor(&graph, "input1", BHWC(1, 2, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 4, 3, 4));

  AddIdentityProducer(&graph, "producer0", input0);
  AddIdentityProducer(&graph, "producer1", input1);
  Node* concat =
      AddConcatNode(&graph, "concat", input0, input1, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::CONCAT));
}

TEST(MakePaddingFromConcatTest, ConcatWithNonZeroConstantInputIsSkipped) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "const", BHWC(1, 2, 3, 4));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 7, 3, 4));

  AddConstNode(&graph, "const_non_zero", zeros, BHWC(1, 2, 3, 4),
               std::vector<float>(24, 1.0f));
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::SKIPPED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::CONCAT));
}

TEST(MakePaddingFromConcatTest, FirstZeroInputHeightConcatBecomesPrependedPad) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 2, 3, 4));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 7, 3, 4));

  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 2, 3, 4),
               std::vector<float>(24, 0.0f));
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(result.message, "Replaced concat with padding");
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.type, PaddingContentType::ZEROS);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 2, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 0));
}

TEST(MakePaddingFromConcatTest, SecondZeroInputHeightConcatBecomesAppendedPad) {
  GraphFloat32 graph;
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 2, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 7, 3, 4));

  AddIdentityProducer(&graph, "producer", input);
  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 2, 3, 4),
               std::vector<float>(24, 0.0f));

  Node* concat =
      AddConcatNode(&graph, "concat", input, zeros, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.type, PaddingContentType::ZEROS);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 2, 0, 0));
}

TEST(MakePaddingFromConcatTest, FirstZeroInputWidthConcatBecomesPrependedPad) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 5, 2, 4));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 5, 5, 4));

  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 5, 2, 4),
               std::vector<float>(40, 0.0f));
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::WIDTH);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 2, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 0));
}

TEST(MakePaddingFromConcatTest, SecondZeroInputWidthConcatBecomesAppendedPad) {
  GraphFloat32 graph;
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 5, 2, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 5, 5, 4));

  AddIdentityProducer(&graph, "producer", input);
  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 5, 2, 4),
               std::vector<float>(40, 0.0f));

  Node* concat =
      AddConcatNode(&graph, "concat", input, zeros, output, Axis::WIDTH);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 2, 0));
}

TEST(MakePaddingFromConcatTest, FirstZeroInputChannelsConcatBecomesPrependedPad) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 5, 3, 2));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 5, 3, 6));

  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 5, 3, 2),
               std::vector<float>(30, 0.0f));
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::CHANNELS);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 0, 2));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 0));
}

TEST(MakePaddingFromConcatTest, SecondZeroInputChannelsConcatBecomesAppendedPad) {
  GraphFloat32 graph;
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 5, 3, 2));
  Value* output = AddTensor(&graph, "output", BHWC(1, 5, 3, 6));

  AddIdentityProducer(&graph, "producer", input);
  AddConstNode(&graph, "const_zeros", zeros, BHWC(1, 5, 3, 2),
               std::vector<float>(30, 0.0f));

  Node* concat =
      AddConcatNode(&graph, "concat", input, zeros, output, Axis::CHANNELS);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 2));
}

TEST(MakePaddingFromConcatTest, ZeroSizedConstTensorIsTreatedAsZeros) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "zeros", BHWC(1, 0, 3, 4));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 5, 3, 4));

  AddConstNode(&graph, "empty_const", zeros, BHWC(1, 0, 3, 4), {});
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 0, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 0));
}

TEST(MakePaddingFromConcatTest, UnsupportedBatchAxisDeclines) {
  GraphFloat32 graph;
  Value* zeros = AddTensor(&graph, "zeros", BHWC(2, 5, 3, 4));
  Value* input = AddTensor(&graph, "input", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(3, 5, 3, 4));

  AddConstNode(&graph, "const_zeros", zeros, BHWC(2, 5, 3, 4),
               std::vector<float>(120, 0.0f));
  AddIdentityProducer(&graph, "producer", input);

  Node* concat =
      AddConcatNode(&graph, "concat", zeros, input, output, Axis::BATCH);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::DECLINED);
  EXPECT_NE(result.message.find("Padding for concat axis is unsupported"),
            std::string::npos);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::CONCAT));
}

TEST(MakePaddingFromConcatTest, FirstZeroInputIsUsedWhenBothInputsAreZeroConstants) {
  GraphFloat32 graph;
  Value* zeros0 = AddTensor(&graph, "zeros0", BHWC(1, 2, 3, 4));
  Value* zeros1 = AddTensor(&graph, "zeros1", BHWC(1, 5, 3, 4));
  Value* output = AddTensor(&graph, "output", BHWC(1, 7, 3, 4));

  AddConstNode(&graph, "const_zeros0", zeros0, BHWC(1, 2, 3, 4),
               std::vector<float>(24, 0.0f));
  AddConstNode(&graph, "const_zeros1", zeros1, BHWC(1, 5, 3, 4),
               std::vector<float>(60, 0.0f));

  Node* concat =
      AddConcatNode(&graph, "concat", zeros0, zeros1, output, Axis::HEIGHT);

  std::unique_ptr<NodeTransformation> transformation =
      NewMakePaddingFromConcat();

  TransformResult result = transformation->ApplyToNode(concat, &graph);

  EXPECT_EQ(result.status, TransformStatus::APPLIED);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::PAD));

  const auto& pad_attr =
      std::any_cast<const PadAttributes&>(concat->operation.attributes);
  EXPECT_EQ(pad_attr.prepended, BHWC(0, 2, 0, 0));
  EXPECT_EQ(pad_attr.appended, BHWC(0, 0, 0, 0));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite