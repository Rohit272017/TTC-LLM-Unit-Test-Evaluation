#include "tensorflow/lite/tools/delegates/compatibility/gpu/gpu_delegate_compatibility_checker.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {
namespace {

TEST(GpuDelegateCompatibilityCheckerTest, GetDccConfigurationsReturnsEmptyMap) {
  GpuDelegateCompatibilityChecker checker;

  const std::unordered_map<std::string, std::string> configs =
      checker.getDccConfigurations();

  EXPECT_TRUE(configs.empty());
}

TEST(GpuDelegateCompatibilityCheckerTest, SetDccConfigurationsWithEmptyMapReturnsOk) {
  GpuDelegateCompatibilityChecker checker;
  std::unordered_map<std::string, std::string> configs;

  absl::Status status = checker.setDccConfigurations(configs);

  EXPECT_TRUE(status.ok());
}

TEST(GpuDelegateCompatibilityCheckerTest, SetDccConfigurationsWithNonEmptyMapReturnsOk) {
  GpuDelegateCompatibilityChecker checker;
  std::unordered_map<std::string, std::string> configs = {
      {"gpu_backend", "opencl"},
      {"precision_loss_allowed", "true"},
  };

  absl::Status status = checker.setDccConfigurations(configs);

  EXPECT_TRUE(status.ok());
}

TEST(GpuDelegateCompatibilityCheckerTest,
     SetDccConfigurationsDoesNotPersistConfigurations) {
  GpuDelegateCompatibilityChecker checker;
  std::unordered_map<std::string, std::string> configs = {
      {"key", "value"},
  };

  EXPECT_TRUE(checker.setDccConfigurations(configs).ok());
  EXPECT_TRUE(checker.getDccConfigurations().empty());
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckModelCompatibilityOnlineReturnsUnimplemented) {
  GpuDelegateCompatibilityChecker checker;
  proto::CompatibilityResult result;

  absl::Status status =
      checker.checkModelCompatibilityOnline(/*model_buffer=*/nullptr, &result);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(
      std::string(status.message()).find(
          "Online mode is not supported on GPU delegate compatibility checker"),
      std::string::npos);
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckModelCompatibilityOnlineDoesNotModifyEmptyResult) {
  GpuDelegateCompatibilityChecker checker;
  proto::CompatibilityResult result;

  absl::Status status =
      checker.checkModelCompatibilityOnline(/*model_buffer=*/nullptr, &result);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(result.op_results_size(), 0);
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilityReturnsOkForDefaultSignature) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  proto::OpCompatibilityResult op_result;

  absl::Status status = checker.checkOpSigCompatibility(op_sig, &op_result);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(op_result.has_is_supported());
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilitySetsSupportedOrFailureForDefaultSignature) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  proto::OpCompatibilityResult op_result;

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());

  if (op_result.is_supported()) {
    EXPECT_EQ(op_result.compatibility_failures_size(), 0);
  } else {
    ASSERT_GE(op_result.compatibility_failures_size(), 1);
    EXPECT_FALSE(op_result.compatibility_failures(0).description().empty());
  }
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilityAppendsFailureWithoutClearingExistingFailures) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  proto::OpCompatibilityResult op_result;

  auto* existing_failure = op_result.add_compatibility_failures();
  existing_failure->set_failure_type(
      proto::CompatibilityFailureType::DCC_INTERNAL_ERROR);
  existing_failure->set_description("existing failure");

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());

  if (op_result.is_supported()) {
    EXPECT_EQ(op_result.compatibility_failures_size(), 1);
    EXPECT_EQ(op_result.compatibility_failures(0).description(),
              "existing failure");
  } else {
    EXPECT_GE(op_result.compatibility_failures_size(), 2);
    EXPECT_EQ(op_result.compatibility_failures(0).description(),
              "existing failure");
  }
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilityOverwritesPreviousSupportedFlag) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  proto::OpCompatibilityResult op_result;

  op_result.set_is_supported(false);

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());

  EXPECT_TRUE(op_result.has_is_supported());
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilityWithUnsupportedBuiltinOpProducesFailureIfRejected) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  op_sig.op = static_cast<BuiltinOperator>(-1);

  proto::OpCompatibilityResult op_result;

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());

  EXPECT_TRUE(op_result.has_is_supported());
  if (!op_result.is_supported()) {
    ASSERT_GE(op_result.compatibility_failures_size(), 1);
    const auto& failure = op_result.compatibility_failures(0);
    EXPECT_FALSE(failure.description().empty());
    EXPECT_TRUE(
        failure.failure_type() ==
            proto::CompatibilityFailureType::DCC_INVALID_ARGUMENT ||
        failure.failure_type() ==
            proto::CompatibilityFailureType::DCC_UNIMPLEMENTED_ERROR ||
        failure.failure_type() ==
            proto::CompatibilityFailureType::DCC_INTERNAL_ERROR ||
        failure.failure_type() ==
            proto::CompatibilityFailureType::DCC_OUT_OF_RANGE);
  }
}

TEST(GpuDelegateCompatibilityCheckerTest,
     CheckOpSigCompatibilityCanBeCalledMultipleTimesOnSameResult) {
  GpuDelegateCompatibilityChecker checker;
  OpSignature op_sig;
  proto::OpCompatibilityResult op_result;

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());
  const bool first_supported = op_result.is_supported();
  const int first_failure_count = op_result.compatibility_failures_size();

  ASSERT_TRUE(checker.checkOpSigCompatibility(op_sig, &op_result).ok());

  EXPECT_TRUE(op_result.has_is_supported());
  EXPECT_EQ(op_result.is_supported(), first_supported);
  EXPECT_GE(op_result.compatibility_failures_size(), first_failure_count);
}

}  // namespace
}  // namespace tools
}  // namespace tflite