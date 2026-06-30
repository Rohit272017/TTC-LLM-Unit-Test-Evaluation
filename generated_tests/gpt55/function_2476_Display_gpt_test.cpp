#include "tensorflow/core/profiler/internal/tfprof_tensor.h"

#include <gtest/gtest.h>

#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace tfprof {
namespace {

TEST(TFProfTensorTest, ECP_DisplayFloatTensorBuildsProtoAndFormattedString) {
  Tensor tensor(DT_FLOAT, TensorShape({3}));
  auto flat = tensor.flat<float>();
  flat(0) = -1.5f;
  flat(1) = 0.0f;
  flat(2) = 2.5f;

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_FLOAT);
  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, ECP_DisplayDoubleTensorBuildsProto) {
  Tensor tensor(DT_DOUBLE, TensorShape({2}));
  auto flat = tensor.flat<double>();
  flat(0) = -10.25;
  flat(1) = 99.75;

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_DOUBLE);
}

TEST(TFProfTensorTest, ECP_DisplayInt32TensorBuildsProto) {
  Tensor tensor(DT_INT32, TensorShape({3}));
  auto flat = tensor.flat<int32>();
  flat(0) = -1;
  flat(1) = 0;
  flat(2) = 1;

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_INT32);
}

TEST(TFProfTensorTest, ECP_DisplayInt64TensorBuildsProto) {
  Tensor tensor(DT_INT64, TensorShape({3}));
  auto flat = tensor.flat<int64_t>();
  flat(0) = -10000000000LL;
  flat(1) = 0LL;
  flat(2) = 10000000000LL;

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_INT64);
}

TEST(TFProfTensorTest, ECP_DisplayStringTensorBuildsProto) {
  Tensor tensor(DT_STRING, TensorShape({3}));
  auto flat = tensor.flat<tstring>();
  flat(0) = "abc";
  flat(1) = "";
  flat(2) = "xyz";

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_STRING);
  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, BVA_DisplayScalarFloatTensor) {
  Tensor tensor(DT_FLOAT, TensorShape({}));
  tensor.scalar<float>()() = 1.25f;

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_FLOAT);
  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, BVA_DisplayEmptyFloatTensor) {
  Tensor tensor(DT_FLOAT, TensorShape({0}));

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_FLOAT);
}

TEST(TFProfTensorTest, BVA_DisplayInt32MinAndMaxValues) {
  Tensor tensor(DT_INT32, TensorShape({2}));
  auto flat = tensor.flat<int32>();
  flat(0) = std::numeric_limits<int32>::min();
  flat(1) = std::numeric_limits<int32>::max();

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_INT32);
}

TEST(TFProfTensorTest, BVA_DisplayInt64MinAndMaxValues) {
  Tensor tensor(DT_INT64, TensorShape({2}));
  auto flat = tensor.flat<int64_t>();
  flat(0) = std::numeric_limits<int64_t>::min();
  flat(1) = std::numeric_limits<int64_t>::max();

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_INT64);
}

TEST(TFProfTensorTest, BVA_DisplayDoubleLowestAndMaxValues) {
  Tensor tensor(DT_DOUBLE, TensorShape({2}));
  auto flat = tensor.flat<double>();
  flat(0) = std::numeric_limits<double>::lowest();
  flat(1) = std::numeric_limits<double>::max();

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_DOUBLE);
}

TEST(TFProfTensorTest, Edge_DisplayFloatSpecialValues) {
  Tensor tensor(DT_FLOAT, TensorShape({3}));
  auto flat = tensor.flat<float>();
  flat(0) = std::numeric_limits<float>::infinity();
  flat(1) = -std::numeric_limits<float>::infinity();
  flat(2) = std::numeric_limits<float>::quiet_NaN();

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_FLOAT);
}

TEST(TFProfTensorTest, Edge_DisplayStringWithWhitespaceAndEmbeddedNull) {
  Tensor tensor(DT_STRING, TensorShape({2}));
  auto flat = tensor.flat<tstring>();
  flat(0) = "  hello world  ";
  flat(1) = tstring("abc\0def", 7);

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_STRING);
  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, Edge_DisplayLongStringTensor) {
  Tensor tensor(DT_STRING, TensorShape({1}));
  tensor.flat<tstring>()(0) = tstring(4096, 'x');

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  TFProfTensorProto proto;
  prof_tensor.Display(&formatted, &proto);

  EXPECT_EQ(proto.dtype(), DT_STRING);
  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, Invalid_UnsupportedBoolTensorDoesNotCrashAndSetsDtype) {
  Tensor tensor(DT_BOOL, TensorShape({2}));
  auto flat = tensor.flat<bool>();
  flat(0) = true;
  flat(1) = false;

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  EXPECT_NO_FATAL_FAILURE(prof_tensor.Display(nullptr, &proto));

  EXPECT_EQ(proto.dtype(), DT_BOOL);
}

TEST(TFProfTensorTest, Edge_DisplayAllowsBothOutputArgumentsNull) {
  Tensor tensor(DT_FLOAT, TensorShape({1}));
  tensor.flat<float>()(0) = 1.0f;

  TFProfTensor prof_tensor(&tensor);

  EXPECT_NO_FATAL_FAILURE(prof_tensor.Display(nullptr, nullptr));
}

TEST(TFProfTensorTest, Edge_DisplayOnlyFormattedString) {
  Tensor tensor(DT_INT32, TensorShape({1}));
  tensor.flat<int32>()(0) = 42;

  TFProfTensor prof_tensor(&tensor);

  string formatted;
  prof_tensor.Display(&formatted, nullptr);

  EXPECT_FALSE(formatted.empty());
}

TEST(TFProfTensorTest, Edge_DisplayOnlyProto) {
  Tensor tensor(DT_INT32, TensorShape({1}));
  tensor.flat<int32>()(0) = 42;

  TFProfTensor prof_tensor(&tensor);

  TFProfTensorProto proto;
  prof_tensor.Display(nullptr, &proto);

  EXPECT_EQ(proto.dtype(), DT_INT32);
}

TEST(TFProfTensorTest, Edge_RepeatedDisplayReturnsStableResults) {
  Tensor tensor(DT_DOUBLE, TensorShape({2}));
  tensor.flat<double>()(0) = 1.0;
  tensor.flat<double>()(1) = 2.0;

  TFProfTensor prof_tensor(&tensor);

  string first_formatted;
  string second_formatted;
  TFProfTensorProto first_proto;
  TFProfTensorProto second_proto;

  prof_tensor.Display(&first_formatted, &first_proto);
  prof_tensor.Display(&second_formatted, &second_proto);

  EXPECT_EQ(first_formatted, second_formatted);
  EXPECT_EQ(first_proto.dtype(), second_proto.dtype());
}

}  // namespace
}  // namespace tfprof
}  // namespace tensorflow