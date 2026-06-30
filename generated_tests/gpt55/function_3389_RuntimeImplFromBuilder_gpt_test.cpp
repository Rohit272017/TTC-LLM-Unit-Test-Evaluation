#include "runtime/reference_resolver.h"

#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "runtime/runtime_builder.h"
#include "tsl/platform/status_matchers.h"

namespace cel {
namespace {

using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(EnableReferenceResolverTest, CheckedExpressionOnlyIsAccepted) {
  RuntimeBuilder builder;

  absl::Status status = EnableReferenceResolver(
      builder, ReferenceResolverEnabled::kCheckedExpressionOnly);

  EXPECT_THAT(status, IsOk());
}

TEST(EnableReferenceResolverTest, AlwaysIsAccepted) {
  RuntimeBuilder builder;

  absl::Status status =
      EnableReferenceResolver(builder, ReferenceResolverEnabled::kAlways);

  EXPECT_THAT(status, IsOk());
}

TEST(EnableReferenceResolverTest, CanEnableCheckedExpressionOnlyMultipleTimes) {
  RuntimeBuilder builder;

  EXPECT_THAT(EnableReferenceResolver(
                  builder, ReferenceResolverEnabled::kCheckedExpressionOnly),
              IsOk());
  EXPECT_THAT(EnableReferenceResolver(
                  builder, ReferenceResolverEnabled::kCheckedExpressionOnly),
              IsOk());
}

TEST(EnableReferenceResolverTest, CanEnableAlwaysMultipleTimes) {
  RuntimeBuilder builder;

  EXPECT_THAT(EnableReferenceResolver(builder,
                                      ReferenceResolverEnabled::kAlways),
              IsOk());
  EXPECT_THAT(EnableReferenceResolver(builder,
                                      ReferenceResolverEnabled::kAlways),
              IsOk());
}

TEST(EnableReferenceResolverTest, CanEnableDifferentModesOnSameBuilder) {
  RuntimeBuilder builder;

  EXPECT_THAT(EnableReferenceResolver(
                  builder, ReferenceResolverEnabled::kCheckedExpressionOnly),
              IsOk());
  EXPECT_THAT(EnableReferenceResolver(builder,
                                      ReferenceResolverEnabled::kAlways),
              IsOk());
}

#ifndef NDEBUG

TEST(EnableReferenceResolverDeathTest, InvalidEnumeratorCausesFatalFailure) {
  RuntimeBuilder builder;

  EXPECT_DEATH(
      static_cast<void>(EnableReferenceResolver(
          builder, static_cast<ReferenceResolverEnabled>(-1))),
      "unsupported ReferenceResolverEnabled enumerator");
}

TEST(EnableReferenceResolverDeathTest, LargeInvalidEnumeratorCausesFatalFailure) {
  RuntimeBuilder builder;

  EXPECT_DEATH(
      static_cast<void>(EnableReferenceResolver(
          builder, static_cast<ReferenceResolverEnabled>(100))),
      "unsupported ReferenceResolverEnabled enumerator");
}

#endif  // NDEBUG

}  // namespace
}  // namespace cel