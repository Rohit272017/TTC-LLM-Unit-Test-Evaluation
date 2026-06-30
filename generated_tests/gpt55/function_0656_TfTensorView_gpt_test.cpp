#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tflite {
namespace shim {
namespace {

template <typename T>
::tensorflow::Tensor MakeTensor(::tensorflow::DataType dtype,
                                std::initializer_list<int64_t> dims) {
  return ::tensorflow::Tensor(dtype, ::tensorflow::TensorShape(dims));
}

TEST(TfTensorViewTest, CreatesViewForBoolTensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<bool>(::tensorflow::DT_BOOL, {2, 3});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_BOOL);
  EXPECT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 2);
  EXPECT_EQ(result->shape()[1], 3);
}

TEST(TfTensorViewTest, CreatesViewForUint8Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<uint8_t>(::tensorflow::DT_UINT8, {1});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_UINT8);
  ASSERT_EQ(result->shape().size(), 1);
  EXPECT_EQ(result->shape()[0], 1);
}

TEST(TfTensorViewTest, CreatesViewForUint64Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<uint64_t>(::tensorflow::DT_UINT64, {4});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_UINT64);
  ASSERT_EQ(result->shape().size(), 1);
  EXPECT_EQ(result->shape()[0], 4);
}

TEST(TfTensorViewTest, CreatesViewForInt8Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<int8_t>(::tensorflow::DT_INT8, {5});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_INT8);
  ASSERT_EQ(result->shape().size(), 1);
  EXPECT_EQ(result->shape()[0], 5);
}

TEST(TfTensorViewTest, CreatesViewForInt16Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<int16_t>(::tensorflow::DT_INT16, {2, 2});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_INT16);
  ASSERT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 2);
  EXPECT_EQ(result->shape()[1], 2);
}

TEST(TfTensorViewTest, CreatesViewForInt32Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<int32_t>(::tensorflow::DT_INT32, {3, 4});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_INT32);
  ASSERT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 3);
  EXPECT_EQ(result->shape()[1], 4);
}

TEST(TfTensorViewTest, CreatesViewForInt64Tensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<int64_t>(::tensorflow::DT_INT64, {6});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_INT64);
  ASSERT_EQ(result->shape().size(), 1);
  EXPECT_EQ(result->shape()[0], 6);
}

TEST(TfTensorViewTest, CreatesViewForFloatTensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<float>(::tensorflow::DT_FLOAT, {2, 3, 4});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_FLOAT);
  ASSERT_EQ(result->shape().size(), 3);
  EXPECT_EQ(result->shape()[0], 2);
  EXPECT_EQ(result->shape()[1], 3);
  EXPECT_EQ(result->shape()[2], 4);
}

TEST(TfTensorViewTest, CreatesViewForDoubleTensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<double>(::tensorflow::DT_DOUBLE, {1, 2});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_DOUBLE);
  ASSERT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 1);
  EXPECT_EQ(result->shape()[1], 2);
}

TEST(TfTensorViewTest, CreatesViewForStringTensor) {
  ::tensorflow::Tensor tensor =
      MakeTensor<::tensorflow::tstring>(::tensorflow::DT_STRING, {2});

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_STRING);
  ASSERT_EQ(result->shape().size(), 1);
  EXPECT_EQ(result->shape()[0], 2);
}

TEST(TfTensorViewTest, ScalarTensorHasEmptyShape) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_FLOAT,
                              ::tensorflow::TensorShape({}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_FLOAT);
  EXPECT_TRUE(result->shape().empty());
}

TEST(TfTensorViewTest, ZeroSizedDimensionIsPreserved) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_INT32,
                              ::tensorflow::TensorShape({0, 3}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 0);
  EXPECT_EQ(result->shape()[1], 3);
}

TEST(TfTensorViewTest, HighRankShapeIsPreserved) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_FLOAT,
                              ::tensorflow::TensorShape({1, 2, 3, 4, 5, 6}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->shape().size(), 6);
  EXPECT_EQ(result->shape()[0], 1);
  EXPECT_EQ(result->shape()[1], 2);
  EXPECT_EQ(result->shape()[2], 3);
  EXPECT_EQ(result->shape()[3], 4);
  EXPECT_EQ(result->shape()[4], 5);
  EXPECT_EQ(result->shape()[5], 6);
}

TEST(TfTensorViewTest, UnsupportedDataTypeReturnsUnimplemented) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_COMPLEX64,
                              ::tensorflow::TensorShape({2}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(result.status().message().find("Unsupported data type"),
            std::string::npos);
}

TEST(TfTensorViewTest, ConstTensorCreatesConstView) {
  const ::tensorflow::Tensor tensor(::tensorflow::DT_FLOAT,
                                    ::tensorflow::TensorShape({2, 2}));

  auto result = TensorView::New<const ::tensorflow::Tensor>(&tensor);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->dtype(), ::tensorflow::DT_FLOAT);
  ASSERT_EQ(result->shape().size(), 2);
  EXPECT_EQ(result->shape()[0], 2);
  EXPECT_EQ(result->shape()[1], 2);
}

TEST(TfTensorViewTest, ConstUnsupportedDataTypeReturnsUnimplemented) {
  const ::tensorflow::Tensor tensor(::tensorflow::DT_COMPLEX128,
                                    ::tensorflow::TensorShape({1}));

  auto result = TensorView::New<const ::tensorflow::Tensor>(&tensor);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(result.status().message().find("Unsupported data type"),
            std::string::npos);
}

TEST(TfTensorViewTest, CopyConstructorPreservesShapeAndDtype) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_INT32,
                              ::tensorflow::TensorShape({2, 3}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);
  ASSERT_TRUE(result.ok());

  TfTensorView copied(*result);

  EXPECT_EQ(copied.dtype(), result->dtype());
  ASSERT_EQ(copied.shape().size(), 2);
  EXPECT_EQ(copied.shape()[0], 2);
  EXPECT_EQ(copied.shape()[1], 3);
}

TEST(TfTensorViewTest, CopyAssignmentPreservesShapeAndDtype) {
  ::tensorflow::Tensor source_tensor(::tensorflow::DT_DOUBLE,
                                     ::tensorflow::TensorShape({4, 5}));
  ::tensorflow::Tensor target_tensor(::tensorflow::DT_FLOAT,
                                     ::tensorflow::TensorShape({1}));

  auto source = TensorView::New<::tensorflow::Tensor>(&source_tensor);
  auto target = TensorView::New<::tensorflow::Tensor>(&target_tensor);
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(target.ok());

  *target = *source;

  EXPECT_EQ(target->dtype(), ::tensorflow::DT_DOUBLE);
  ASSERT_EQ(target->shape().size(), 2);
  EXPECT_EQ(target->shape()[0], 4);
  EXPECT_EQ(target->shape()[1], 5);
}

TEST(TfTensorViewTest, SelfCopyAssignmentKeepsObjectValid) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_UINT8,
                              ::tensorflow::TensorShape({7}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);
  ASSERT_TRUE(result.ok());

  TfTensorView& view = *result;
  view = view;

  EXPECT_EQ(view.dtype(), ::tensorflow::DT_UINT8);
  ASSERT_EQ(view.shape().size(), 1);
  EXPECT_EQ(view.shape()[0], 7);
}

TEST(TfTensorViewTest, MoveConstructorPreservesShapeAndDtype) {
  ::tensorflow::Tensor tensor(::tensorflow::DT_INT64,
                              ::tensorflow::TensorShape({8, 9}));

  auto result = TensorView::New<::tensorflow::Tensor>(&tensor);
  ASSERT_TRUE(result.ok());

  TfTensorView moved(std::move(*result));

  EXPECT_EQ(moved.dtype(), ::tensorflow::DT_INT64);
  ASSERT_EQ(moved.shape().size(), 2);
  EXPECT_EQ(moved.shape()[0], 8);
  EXPECT_EQ(moved.shape()[1], 9);
}

TEST(TfTensorViewTest, MoveAssignmentPreservesShapeAndDtype) {
  ::tensorflow::Tensor source_tensor(::tensorflow::DT_BOOL,
                                     ::tensorflow::TensorShape({3}));
  ::tensorflow::Tensor target_tensor(::tensorflow::DT_FLOAT,
                                     ::tensorflow::TensorShape({1, 1}));

  auto source = TensorView::New<::tensorflow::Tensor>(&source_tensor);
  auto target = TensorView::New<::tensorflow::Tensor>(&target_tensor);
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(target.ok());

  *target = std::move(*source);

  EXPECT_EQ(target->dtype(), ::tensorflow::DT_BOOL);
  ASSERT_EQ(target->shape().size(), 1);
  EXPECT_EQ(target->shape()[0], 3);
}

}  // namespace
}  // namespace shim
}  // namespace tflite