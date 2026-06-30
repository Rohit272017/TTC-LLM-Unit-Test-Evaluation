#include "xla/tests/literal_test_util.h"
#include "absl/strings/str_format.h"
#include "xla/literal_comparison.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"
namespace xla {
namespace {
void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& name) {
  std::string outdir;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    outdir = tsl::testing::TmpDir();
  }
  auto* env = tsl::Env::Default();
  std::string filename = tsl::io::JoinPath(
      outdir, absl::StrFormat("tempfile-%d-%s", env->NowMicros(), name));
  TF_CHECK_OK(tsl::WriteBinaryProto(env, absl::StrCat(filename, ".pb"),
                                    literal.ToProto()));
  TF_CHECK_OK(tsl::WriteStringToFile(env, absl::StrCat(filename, ".txt"),
                                     literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " file: " << filename
             << ".{pb,txt}";
}
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& ,
                  const literal_comparison::ErrorBuckets& ) {
  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}
::testing::AssertionResult StatusToAssertion(const absl::Status& s) {
  if (s.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << s.message();
}
}  
 ::testing::AssertionResult LiteralTestUtil::EqualShapes(
    const Shape& expected, const Shape& actual) {
  return StatusToAssertion(literal_comparison::EqualShapes(expected, actual));
}
 ::testing::AssertionResult LiteralTestUtil::EqualShapesAndLayouts(
    const Shape& expected, const Shape& actual) {
  if (expected.ShortDebugString() != actual.ShortDebugString()) {
    return ::testing::AssertionFailure()
           << "want: " << expected.ShortDebugString()
           << " got: " << actual.ShortDebugString();
  }
  return ::testing::AssertionSuccess();
}
 ::testing::AssertionResult LiteralTestUtil::Equal(
    const LiteralSlice& expected, const LiteralSlice& actual) {
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}
 ::testing::AssertionResult LiteralTestUtil::Near(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const ErrorSpec& error_spec, std::optional<bool> detailed_message) {
  return StatusToAssertion(literal_comparison::Near(
      expected, actual, error_spec, detailed_message, &OnMiscompare));
}
 ::testing::AssertionResult LiteralTestUtil::NearOrEqual(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const std::optional<ErrorSpec>& error) {
  if (error.has_value()) {
    VLOG(1) << "Expects near";
    return StatusToAssertion(literal_comparison::Near(
        expected, actual, *error, std::nullopt,
        &OnMiscompare));
  }
  VLOG(1) << "Expects equal";
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}
}  