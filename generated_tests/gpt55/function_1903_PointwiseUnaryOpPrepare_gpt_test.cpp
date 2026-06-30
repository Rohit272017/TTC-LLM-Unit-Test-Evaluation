#include "tensorflow/lite/kernels/custom_ops_register.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sign {
namespace {

TfLiteIntArray* MakeDims(std::initializer_list<int> dims) {
  TfLiteIntArray* array = TfLiteIntArrayCreate(dims.size());
  int i = 0;
  for (int dim : dims) {
    array->data[i++] = dim;
  }
  return array;
}

template <typename T>
TfLiteTensor MakeTensor(TfLiteType type, TfLiteIntArray* dims, T* data) {
  TfLiteTensor tensor{};
  tensor.type = type;
  tensor.dims = dims;
  tensor.data.data = reinterpret_cast<char*>(data);
  return tensor;
}

TfLiteStatus ResizeTensorStub(TfLiteContext* context,
                              TfLiteTensor* tensor,
                              TfLiteIntArray* new_size) {
  if (tensor->dims != nullptr) {
    TfLiteIntArrayFree(tensor->dims);
  }
  tensor->dims = new_size;
  return kTfLiteOk;
}

class SignCustomOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = {};
    context_.ResizeTensor = ResizeTensorStub;

    node_ = {};
    inputs_ = TfLiteIntArrayCreate(1);
    inputs_->data[0] = 0;
    outputs_ = TfLiteIntArrayCreate(1);
    outputs_->data[0] = 1;

    node_.inputs = inputs_;
    node_.outputs = outputs_;

    tensors_.resize(2);
    context_.tensors = tensors_.data();
    context_.tensors_size = tensors_.size();
  }

  void TearDown() override {
    for (auto& tensor : tensors_) {
      if (tensor.dims != nullptr) {
        TfLiteIntArrayFree(tensor.dims);
        tensor.dims = nullptr;
      }
    }

    TfLiteIntArrayFree(inputs_);
    TfLiteIntArrayFree(outputs_);
  }

  TfLiteContext context_{};
  TfLiteNode node_{};
  TfLiteIntArray* inputs_ = nullptr;
  TfLiteIntArray* outputs_ = nullptr;
  std::vector<TfLiteTensor> tensors_;
};

TEST_F(SignCustomOpTest, RegisterSignReturnsValidRegistration) {
  TfLiteRegistration* registration = Register_SIGN();

  ASSERT_NE(registration, nullptr);
  EXPECT_EQ(registration->init, nullptr);
  EXPECT_EQ(registration->free, nullptr);
  EXPECT_NE(registration->prepare, nullptr);
  EXPECT_NE(registration->invoke, nullptr);
}

TEST_F(SignCustomOpTest, ECP_PrepareWithFloat32MatchingTypesCopiesInputShape) {
  float input_data[6] = {};
  float output_data[6] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({2, 3}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  TfLiteStatus status = PointwiseUnaryOpPrepare(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 2);
  EXPECT_EQ(tensors_[1].dims->data[0], 2);
  EXPECT_EQ(tensors_[1].dims->data[1], 3);
}

TEST_F(SignCustomOpTest, ECP_PrepareWithFloat64MatchingTypesCopiesInputShape) {
  double input_data[4] = {};
  double output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({4}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  TfLiteStatus status = PointwiseUnaryOpPrepare(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 4);
}

TEST_F(SignCustomOpTest, BVA_PrepareScalarInputCopiesZeroRankShape) {
  float input_data[1] = {};
  float output_data[1] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  TfLiteStatus status = PointwiseUnaryOpPrepare(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  EXPECT_EQ(tensors_[1].dims->size, 0);
}

TEST_F(SignCustomOpTest, BVA_PrepareZeroElementTensorCopiesShape) {
  float input_data[1] = {};
  float output_data[1] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({0}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  TfLiteStatus status = PointwiseUnaryOpPrepare(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 0);
}

TEST_F(SignCustomOpTest, Invalid_PrepareFailsWhenInputCountIsNotOne) {
  float input_data[1] = {};
  float output_data[1] = {};

  TfLiteIntArrayFree(inputs_);
  inputs_ = TfLiteIntArrayCreate(2);
  inputs_->data[0] = 0;
  inputs_->data[1] = 0;
  node_.inputs = inputs_;

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(PointwiseUnaryOpPrepare(&context_, &node_), kTfLiteError);
}

TEST_F(SignCustomOpTest, Invalid_PrepareFailsWhenInputAndOutputTypesDiffer) {
  float input_data[1] = {};
  double output_data[1] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  EXPECT_EQ(PointwiseUnaryOpPrepare(&context_, &node_), kTfLiteError);
}

TEST_F(SignCustomOpTest, ECP_EvalFloat32NegativeZeroPositiveValues) {
  float input_data[] = {-3.5f, -1.0f, -0.0f, 0.0f, 2.25f, 100.0f};
  float output_data[6] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({6}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({6}), output_data);

  TfLiteStatus status = PointwiseUnaryOpEval<Sign>(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], -1.0f);
  EXPECT_FLOAT_EQ(output_data[1], -1.0f);
  EXPECT_FLOAT_EQ(output_data[2], 0.0f);
  EXPECT_FLOAT_EQ(output_data[3], 0.0f);
  EXPECT_FLOAT_EQ(output_data[4], 1.0f);
  EXPECT_FLOAT_EQ(output_data[5], 1.0f);
}

TEST_F(SignCustomOpTest, ECP_EvalFloat64NegativeZeroPositiveValues) {
  double input_data[] = {-3.5, -1.0, -0.0, 0.0, 2.25, 100.0};
  double output_data[6] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({6}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({6}), output_data);

  TfLiteStatus status = PointwiseUnaryOpEval<Sign>(&context_, &node_);

  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_DOUBLE_EQ(output_data[0], -1.0);
  EXPECT_DOUBLE_EQ(output_data[1], -1.0);
  EXPECT_DOUBLE_EQ(output_data[2], 0.0);
  EXPECT_DOUBLE_EQ(output_data[3], 0.0);
  EXPECT_DOUBLE_EQ(output_data[4], 1.0);
  EXPECT_DOUBLE_EQ(output_data[5], 1.0);
}

TEST_F(SignCustomOpTest, BVA_EvalFloat32SmallestPositiveSubnormalReturnsOne) {
  float input_data[] = {std::numeric_limits<float>::denorm_min()};
  float output_data[] = {0.0f};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], 1.0f);
}

TEST_F(SignCustomOpTest, BVA_EvalFloat32NegativeSmallestSubnormalReturnsMinusOne) {
  float input_data[] = {-std::numeric_limits<float>::denorm_min()};
  float output_data[] = {0.0f};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], -1.0f);
}

TEST_F(SignCustomOpTest, BVA_EvalFloat64SmallestPositiveSubnormalReturnsOne) {
  double input_data[] = {std::numeric_limits<double>::denorm_min()};
  double output_data[] = {0.0};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_DOUBLE_EQ(output_data[0], 1.0);
}

TEST_F(SignCustomOpTest, BVA_EvalFloat64NegativeSmallestSubnormalReturnsMinusOne) {
  double input_data[] = {-std::numeric_limits<double>::denorm_min()};
  double output_data[] = {0.0};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_DOUBLE_EQ(output_data[0], -1.0);
}

TEST_F(SignCustomOpTest, Edge_EvalFloat32InfinityValues) {
  float input_data[] = {-std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity()};
  float output_data[2] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({2}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({2}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], -1.0f);
  EXPECT_FLOAT_EQ(output_data[1], 1.0f);
}

TEST_F(SignCustomOpTest, Edge_EvalFloat64InfinityValues) {
  double input_data[] = {-std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
  double output_data[2] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({2}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({2}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_DOUBLE_EQ(output_data[0], -1.0);
  EXPECT_DOUBLE_EQ(output_data[1], 1.0);
}

TEST_F(SignCustomOpTest, Edge_EvalFloat32NaNReturnsZero) {
  float input_data[] = {std::numeric_limits<float>::quiet_NaN()};
  float output_data[] = {123.0f};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], 0.0f);
}

TEST_F(SignCustomOpTest, Edge_EvalFloat64NaNReturnsZero) {
  double input_data[] = {std::numeric_limits<double>::quiet_NaN()};
  double output_data[] = {123.0};

  tensors_[0] = MakeTensor(kTfLiteFloat64, MakeDims({1}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_DOUBLE_EQ(output_data[0], 0.0);
}

TEST_F(SignCustomOpTest, Edge_EvalZeroElementTensorDoesNotModifyOutput) {
  float input_data[1] = {99.0f};
  float output_data[1] = {77.0f};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({0}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({0}), output_data);

  ASSERT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], 77.0f);
}

TEST_F(SignCustomOpTest, Invalid_EvalUnsupportedOutputTypeReturnsError) {
  int32_t input_data[] = {-1, 0, 1};
  int32_t output_data[3] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({3}), input_data);
  tensors_[1] = MakeTensor(kTfLiteInt32, MakeDims({3}), output_data);

  EXPECT_EQ(PointwiseUnaryOpEval<Sign>(&context_, &node_), kTfLiteError);
}

TEST_F(SignCustomOpTest, Integration_RegisterInvokeFloat32Works) {
  float input_data[] = {-2.0f, 0.0f, 2.0f};
  float output_data[3] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({3}), input_data);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({3}), output_data);

  TfLiteRegistration* registration = Register_SIGN();

  ASSERT_NE(registration, nullptr);
  ASSERT_NE(registration->prepare, nullptr);
  ASSERT_NE(registration->invoke, nullptr);

  ASSERT_EQ(registration->prepare(&context_, &node_), kTfLiteOk);
  ASSERT_EQ(registration->invoke(&context_, &node_), kTfLiteOk);

  EXPECT_FLOAT_EQ(output_data[0], -1.0f);
  EXPECT_FLOAT_EQ(output_data[1], 0.0f);
  EXPECT_FLOAT_EQ(output_data[2], 1.0f);
}

}  // namespace
}  // namespace sign
}  // namespace custom
}  // namespace ops
}  // namespace tflite