#include "tensorflow/lite/kernels/custom_ops_register.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace ops {
namespace custom {
namespace random_standard_normal {
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
TfLiteTensor MakeTensor(TfLiteType type,
                        TfLiteIntArray* dims,
                        T* data,
                        TfLiteAllocationType allocation_type = kTfLiteArenaRw) {
  TfLiteTensor tensor{};
  tensor.type = type;
  tensor.dims = dims;
  tensor.data.data = reinterpret_cast<char*>(data);
  tensor.allocation_type = allocation_type;
  return tensor;
}

TfLiteStatus ResizeTensorStub(TfLiteContext* context,
                              TfLiteTensor* tensor,
                              TfLiteIntArray* new_size) {
  if (tensor->dims != nullptr) {
    TfLiteIntArrayFree(tensor->dims);
  }
  tensor->dims = new_size;
  tensor->allocation_type = kTfLiteArenaRw;
  return kTfLiteOk;
}

class RandomStandardNormalTest : public ::testing::Test {
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

    node_.user_data = Init(&context_, nullptr, 0);
  }

  void TearDown() override {
    Free(&context_, node_.user_data);
    node_.user_data = nullptr;

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

TEST_F(RandomStandardNormalTest, RegisterReturnsValidRegistration) {
  TfLiteRegistration* registration = Register_RANDOM_STANDARD_NORMAL();

  ASSERT_NE(registration, nullptr);
  EXPECT_NE(registration->init, nullptr);
  EXPECT_NE(registration->free, nullptr);
  EXPECT_NE(registration->prepare, nullptr);
  EXPECT_NE(registration->invoke, nullptr);
}

TEST_F(RandomStandardNormalTest, InitReturnsNonNullOpData) {
  void* data = Init(&context_, nullptr, 0);

  EXPECT_NE(data, nullptr);

  Free(&context_, data);
}

TEST_F(RandomStandardNormalTest, FreeAcceptsNullBuffer) {
  EXPECT_NO_FATAL_FAILURE(Free(&context_, nullptr));
}

TEST_F(RandomStandardNormalTest, ECP_PrepareConstantShapeResizesFloat32Output) {
  int32_t shape_data[] = {2, 3};
  float output_data[6] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({2}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(Prepare(&context_, &node_), kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 2);
  EXPECT_EQ(tensors_[1].dims->data[0], 2);
  EXPECT_EQ(tensors_[1].dims->data[1], 3);
}

TEST_F(RandomStandardNormalTest, ECP_PrepareConstantShapeResizesFloat64Output) {
  int32_t shape_data[] = {4};
  double output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  ASSERT_EQ(Prepare(&context_, &node_), kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 4);
}

TEST_F(RandomStandardNormalTest, BVA_PrepareScalarOutputShape) {
  int32_t shape_data[] = {};
  float output_data[1] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({0}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(Prepare(&context_, &node_), kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  EXPECT_EQ(tensors_[1].dims->size, 0);
}

TEST_F(RandomStandardNormalTest, BVA_PrepareZeroElementOutputShape) {
  int32_t shape_data[] = {0};
  float output_data[1] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(Prepare(&context_, &node_), kTfLiteOk);
  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 0);
}

TEST_F(RandomStandardNormalTest, ECP_PrepareNonConstantShapeMarksOutputDynamic) {
  int32_t shape_data[] = {2, 2};
  float output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({2}), shape_data,
                           kTfLiteArenaRw);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(Prepare(&context_, &node_), kTfLiteOk);
  EXPECT_EQ(tensors_[1].allocation_type, kTfLiteDynamic);
}

TEST_F(RandomStandardNormalTest, Invalid_PrepareFailsWhenInputCountIsNotOne) {
  int32_t shape_data[] = {2};
  float output_data[2] = {};

  TfLiteIntArrayFree(inputs_);
  inputs_ = TfLiteIntArrayCreate(2);
  inputs_->data[0] = 0;
  inputs_->data[1] = 0;
  node_.inputs = inputs_;

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(Prepare(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, Invalid_PrepareFailsWhenOutputCountIsNotOne) {
  int32_t shape_data[] = {2};
  float output_data[2] = {};

  TfLiteIntArrayFree(outputs_);
  outputs_ = TfLiteIntArrayCreate(2);
  outputs_->data[0] = 1;
  outputs_->data[1] = 1;
  node_.outputs = outputs_;

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(Prepare(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, Invalid_PrepareFailsWhenShapeTensorIsNotInt32) {
  float shape_data[] = {2.0f};
  float output_data[2] = {};

  tensors_[0] = MakeTensor(kTfLiteFloat32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(Prepare(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, Invalid_PrepareFailsWhenShapeTensorIsNotRankOne) {
  int32_t shape_data[] = {2, 2};
  float output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1, 2}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(Prepare(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, ECP_EvalFloat32GeneratesFiniteValues) {
  int32_t shape_data[] = {8};
  float output_data[8] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({8}), output_data);

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);

  for (float value : output_data) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST_F(RandomStandardNormalTest, ECP_EvalFloat64GeneratesFiniteValues) {
  int32_t shape_data[] = {8};
  double output_data[8] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({8}), output_data);

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);

  for (double value : output_data) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST_F(RandomStandardNormalTest, BVA_EvalSingleElementFloat32) {
  int32_t shape_data[] = {1};
  float output_data[1] = {0.0f};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);
  EXPECT_TRUE(std::isfinite(output_data[0]));
}

TEST_F(RandomStandardNormalTest, BVA_EvalSingleElementFloat64) {
  int32_t shape_data[] = {1};
  double output_data[1] = {0.0};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);
  EXPECT_TRUE(std::isfinite(output_data[0]));
}

TEST_F(RandomStandardNormalTest, BVA_EvalZeroElementTensorDoesNotModifyOutput) {
  int32_t shape_data[] = {0};
  float output_data[1] = {123.0f};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({0}), output_data);

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);
  EXPECT_FLOAT_EQ(output_data[0], 123.0f);
}

TEST_F(RandomStandardNormalTest, ECP_EvalDynamicOutputResizesBeforeSampling) {
  int32_t shape_data[] = {2, 2};
  float output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({2}), shape_data,
                           kTfLiteArenaRw);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);
  tensors_[1].allocation_type = kTfLiteDynamic;

  ASSERT_EQ(Eval(&context_, &node_), kTfLiteOk);

  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 2);
  EXPECT_EQ(tensors_[1].dims->data[0], 2);
  EXPECT_EQ(tensors_[1].dims->data[1], 2);

  for (float value : output_data) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST_F(RandomStandardNormalTest, Invalid_EvalFailsWhenUserDataIsNull) {
  int32_t shape_data[] = {1};
  float output_data[1] = {};

  Free(&context_, node_.user_data);
  node_.user_data = nullptr;

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  EXPECT_EQ(Eval(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, Invalid_EvalUnsupportedOutputTypeReturnsError) {
  int32_t shape_data[] = {3};
  int32_t output_data[3] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteInt32, MakeDims({3}), output_data);

  EXPECT_EQ(Eval(&context_, &node_), kTfLiteError);
}

TEST_F(RandomStandardNormalTest, Integration_RegisterPrepareAndInvokeFloat32Works) {
  int32_t shape_data[] = {4};
  float output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat32, MakeDims({1}), output_data);

  TfLiteRegistration* registration = Register_RANDOM_STANDARD_NORMAL();

  ASSERT_NE(registration, nullptr);
  ASSERT_NE(registration->prepare, nullptr);
  ASSERT_NE(registration->invoke, nullptr);

  ASSERT_EQ(registration->prepare(&context_, &node_), kTfLiteOk);
  ASSERT_EQ(registration->invoke(&context_, &node_), kTfLiteOk);

  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 4);

  for (float value : output_data) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST_F(RandomStandardNormalTest, Integration_RegisterPrepareAndInvokeFloat64Works) {
  int32_t shape_data[] = {4};
  double output_data[4] = {};

  tensors_[0] = MakeTensor(kTfLiteInt32, MakeDims({1}), shape_data,
                           kTfLiteMmapRo);
  tensors_[1] = MakeTensor(kTfLiteFloat64, MakeDims({1}), output_data);

  TfLiteRegistration* registration = Register_RANDOM_STANDARD_NORMAL();

  ASSERT_NE(registration, nullptr);
  ASSERT_NE(registration->prepare, nullptr);
  ASSERT_NE(registration->invoke, nullptr);

  ASSERT_EQ(registration->prepare(&context_, &node_), kTfLiteOk);
  ASSERT_EQ(registration->invoke(&context_, &node_), kTfLiteOk);

  ASSERT_NE(tensors_[1].dims, nullptr);
  ASSERT_EQ(tensors_[1].dims->size, 1);
  EXPECT_EQ(tensors_[1].dims->data[0], 4);

  for (double value : output_data) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

}  // namespace
}  // namespace random_standard_normal
}  // namespace custom
}  // namespace ops
}  // namespace tflite