#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/lut.h"
#include "tensorflow/lite/kernels/register.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace tflite {
namespace ops {
namespace custom {
namespace table {
namespace {

class TableOpTest : public ::testing::Test {
 protected:
  static TfLiteIntArray* Dims(std::initializer_list<int> dims) {
    TfLiteIntArray* arr = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (int dim : dims) {
      arr->data[i++] = dim;
    }
    return arr;
  }

  static TfLiteTensor Tensor(TfLiteType type, TfLiteIntArray* dims,
                             void* data = nullptr, int zero_point = 0) {
    TfLiteTensor tensor = {};
    tensor.type = type;
    tensor.dims = dims;
    tensor.data.raw = static_cast<char*>(data);
    tensor.params.zero_point = zero_point;
    return tensor;
  }

  static TfLiteNode Node(TfLiteIntArray* inputs, TfLiteIntArray* outputs) {
    TfLiteNode node = {};
    node.inputs = inputs;
    node.outputs = outputs;
    return node;
  }

  static TfLiteStatus ResizeTensor(TfLiteContext* context,
                                   TfLiteTensor* tensor,
                                   TfLiteIntArray* new_size) {
    if (tensor->dims != nullptr) {
      TfLiteIntArrayFree(tensor->dims);
    }
    tensor->dims = new_size;
    return kTfLiteOk;
  }

  void SetContext(std::vector<TfLiteTensor>* tensors) {
    context_ = {};
    context_.tensors = tensors->data();
    context_.tensors_size = tensors->size();
    context_.ResizeTensor = ResizeTensor;
  }

  TfLiteContext context_ = {};
};

TEST_F(TableOpTest, PrepareInt8ValidResizesOutputToInputShape) {
  std::vector<int8_t> input_data(6);
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>());
  std::vector<int8_t> output_data(6);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({2, 3}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({reference_integer_ops::LUTSize<int8_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt8, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteOk);
  ASSERT_EQ(tensors[2].dims->size, 2);
  EXPECT_EQ(tensors[2].dims->data[0], 2);
  EXPECT_EQ(tensors[2].dims->data[1], 3);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareInt16ValidResizesOutputToInputShape) {
  std::vector<int16_t> input_data(4);
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>());
  std::vector<int16_t> output_data(4);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt16, Dims({4}), input_data.data(), 0),
      Tensor(kTfLiteInt16, Dims({reference_integer_ops::LUTSize<int16_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({1}), output_data.data(), 0),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteOk);
  ASSERT_EQ(tensors[2].dims->size, 1);
  EXPECT_EQ(tensors[2].dims->data[0], 4);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenInputCountIsNotTwo) {
  std::vector<TfLiteTensor> tensors(3);
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenOutputCountIsNotOne) {
  std::vector<TfLiteTensor> tensors(3);
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2, 2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsForUnsupportedInputType) {
  std::vector<float> input_data(1);
  std::vector<float> table_data(1);
  std::vector<float> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteFloat32, Dims({1}), input_data.data()),
      Tensor(kTfLiteFloat32, Dims({1}), table_data.data()),
      Tensor(kTfLiteFloat32, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenOutputTypeDiffersFromInputType) {
  std::vector<int8_t> input_data(1);
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>());
  std::vector<int16_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({1}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({reference_integer_ops::LUTSize<int8_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenTableTypeDiffersFromOutputType) {
  std::vector<int8_t> input_data(1);
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>());
  std::vector<int8_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({1}), input_data.data()),
      Tensor(kTfLiteInt16, Dims({reference_integer_ops::LUTSize<int16_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt8, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsForInt16InputNonZeroZeroPoint) {
  std::vector<int16_t> input_data(1);
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>());
  std::vector<int16_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt16, Dims({1}), input_data.data(), 1),
      Tensor(kTfLiteInt16, Dims({reference_integer_ops::LUTSize<int16_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({1}), output_data.data(), 0),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsForInt16OutputNonZeroZeroPoint) {
  std::vector<int16_t> input_data(1);
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>());
  std::vector<int16_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt16, Dims({1}), input_data.data(), 0),
      Tensor(kTfLiteInt16, Dims({reference_integer_ops::LUTSize<int16_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({1}), output_data.data(), 1),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenTableIsNotOneDimensional) {
  std::vector<int8_t> input_data(1);
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>());
  std::vector<int8_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({1}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({16, 16}), table_data.data()),
      Tensor(kTfLiteInt8, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenInt8TableSizeIsTooSmall) {
  std::vector<int8_t> input_data(1);
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>() - 1);
  std::vector<int8_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({1}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({reference_integer_ops::LUTSize<int8_t>() - 1}),
             table_data.data()),
      Tensor(kTfLiteInt8, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenInt8TableSizeIsTooLarge) {
  std::vector<int8_t> input_data(1);
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>() + 1);
  std::vector<int8_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({1}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({reference_integer_ops::LUTSize<int8_t>() + 1}),
             table_data.data()),
      Tensor(kTfLiteInt8, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, PrepareFailsWhenInt16TableSizeIsWrong) {
  std::vector<int16_t> input_data(1);
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>() - 1);
  std::vector<int16_t> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt16, Dims({1}), input_data.data(), 0),
      Tensor(kTfLiteInt16,
             Dims({reference_integer_ops::LUTSize<int16_t>() - 1}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({1}), output_data.data(), 0),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Prepare(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, EvalInt8UsesLookupTable) {
  std::vector<int8_t> input_data = {
      static_cast<int8_t>(-128), static_cast<int8_t>(-1),
      static_cast<int8_t>(0), static_cast<int8_t>(127)};
  std::vector<int8_t> table_data(reference_integer_ops::LUTSize<int8_t>());
  std::vector<int8_t> output_data(input_data.size(), 0);

  for (int i = 0; i < static_cast<int>(table_data.size()); ++i) {
    table_data[i] = static_cast<int8_t>(127 - i);
  }

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt8, Dims({4}), input_data.data()),
      Tensor(kTfLiteInt8, Dims({reference_integer_ops::LUTSize<int8_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt8, Dims({4}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Eval(&context_, &node), kTfLiteOk);
  EXPECT_EQ(output_data[0], table_data[0]);
  EXPECT_EQ(output_data[1], table_data[127]);
  EXPECT_EQ(output_data[2], table_data[128]);
  EXPECT_EQ(output_data[3], table_data[255]);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, EvalInt16UsesLookupTable) {
  std::vector<int16_t> input_data = {
      static_cast<int16_t>(-32768), static_cast<int16_t>(-1),
      static_cast<int16_t>(0), static_cast<int16_t>(32767)};
  std::vector<int16_t> table_data(reference_integer_ops::LUTSize<int16_t>());
  std::vector<int16_t> output_data(input_data.size(), 0);

  table_data[0] = 11;
  table_data[32767] = 22;
  table_data[32768] = 33;
  table_data[65535] = 44;

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteInt16, Dims({4}), input_data.data(), 0),
      Tensor(kTfLiteInt16, Dims({reference_integer_ops::LUTSize<int16_t>()}),
             table_data.data()),
      Tensor(kTfLiteInt16, Dims({4}), output_data.data(), 0),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Eval(&context_, &node), kTfLiteOk);
  EXPECT_EQ(output_data[0], 11);
  EXPECT_EQ(output_data[1], 22);
  EXPECT_EQ(output_data[2], 33);
  EXPECT_EQ(output_data[3], 44);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST_F(TableOpTest, EvalUnsupportedTypeReturnsError) {
  std::vector<float> input_data(1);
  std::vector<float> table_data(1);
  std::vector<float> output_data(1);

  std::vector<TfLiteTensor> tensors = {
      Tensor(kTfLiteFloat32, Dims({1}), input_data.data()),
      Tensor(kTfLiteFloat32, Dims({1}), table_data.data()),
      Tensor(kTfLiteFloat32, Dims({1}), output_data.data()),
  };
  SetContext(&tensors);

  TfLiteIntArray* inputs = Dims({0, 1});
  TfLiteIntArray* outputs = Dims({2});
  TfLiteNode node = Node(inputs, outputs);

  EXPECT_EQ(Eval(&context_, &node), kTfLiteError);

  TfLiteIntArrayFree(inputs);
  TfLiteIntArrayFree(outputs);
}

TEST(RegisterTableTest, RegistrationContainsPrepareAndEval) {
  TfLiteRegistration* registration = Register_TABLE();

  ASSERT_NE(registration, nullptr);
  EXPECT_EQ(registration->init, nullptr);
  EXPECT_EQ(registration->free, nullptr);
  EXPECT_NE(registration->prepare, nullptr);
  EXPECT_NE(registration->invoke, nullptr);
}

}  // namespace
}  // namespace table
}  // namespace custom
}  // namespace ops
}  // namespace tflite