#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"

#include <gtest/gtest.h>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace {

TEST(RngConverterUtilsTest, ECP_ToTensorflowAlgorithmPhilox) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_PHILOX),
            RNG_ALG_PHILOX);
}

TEST(RngConverterUtilsTest, ECP_ToTensorflowAlgorithmThreeFry) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_THREE_FRY),
            RNG_ALG_THREEFRY);
}

TEST(RngConverterUtilsTest, ECP_ToTensorflowAlgorithmDefault) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_DEFAULT),
            RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, Invalid_ToTensorflowAlgorithmUnknownValueReturnsAutoSelect) {
  auto unknown_alg = static_cast<xla::RandomAlgorithm>(-1);

  EXPECT_EQ(ToTensorflowAlgorithm(unknown_alg), RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, BVA_ToTensorflowAlgorithmZeroValueReturnsAutoSelect) {
  auto zero_alg = static_cast<xla::RandomAlgorithm>(0);

  EXPECT_EQ(ToTensorflowAlgorithm(zero_alg), RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, Invalid_ToTensorflowAlgorithmLargeUnknownValueReturnsAutoSelect) {
  auto unknown_alg = static_cast<xla::RandomAlgorithm>(9999);

  EXPECT_EQ(ToTensorflowAlgorithm(unknown_alg), RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, ECP_DefaultRngAlgForGpuXlaJitReturnsPhilox) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(DEVICE_GPU_XLA_JIT),
            xla::RandomAlgorithm::RNG_PHILOX);
}

TEST(RngConverterUtilsTest, ECP_DefaultRngAlgForCpuXlaJitReturnsPhilox) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(DEVICE_CPU_XLA_JIT),
            xla::RandomAlgorithm::RNG_PHILOX);
}

TEST(RngConverterUtilsTest, ECP_DefaultRngAlgForUnknownDeviceReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType("UNKNOWN_DEVICE"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, BVA_DefaultRngAlgForEmptyDeviceStringReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(""),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForGpuStringWithDifferentCaseReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType("xla_gpu_jit"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForCpuStringWithDifferentCaseReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType("xla_cpu_jit"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForGpuWithLeadingWhitespaceReturnsDefault) {
  std::string device = std::string(" ") + std::string(DEVICE_GPU_XLA_JIT);

  EXPECT_EQ(DefaultRngAlgForDeviceType(device),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForGpuWithTrailingWhitespaceReturnsDefault) {
  std::string device = std::string(DEVICE_GPU_XLA_JIT) + std::string(" ");

  EXPECT_EQ(DefaultRngAlgForDeviceType(device),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForCpuWithTrailingWhitespaceReturnsDefault) {
  std::string device = std::string(DEVICE_CPU_XLA_JIT) + std::string(" ");

  EXPECT_EQ(DefaultRngAlgForDeviceType(device),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForSimilarGpuNameReturnsDefault) {
  std::string device = std::string(DEVICE_GPU_XLA_JIT) + "_EXTRA";

  EXPECT_EQ(DefaultRngAlgForDeviceType(device),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForSimilarCpuNameReturnsDefault) {
  std::string device = std::string(DEVICE_CPU_XLA_JIT) + "_EXTRA";

  EXPECT_EQ(DefaultRngAlgForDeviceType(device),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForNonXlaGpuDeviceReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType("GPU"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, Edge_DefaultRngAlgForNonXlaCpuDeviceReturnsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType("CPU"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

}  // namespace
}  // namespace tensorflow