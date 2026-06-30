#include "absl/base/log_severity.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace {

template <typename T>
std::string StreamToString(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

TEST(LogSeverityStreamTest, ECP_LogSeverityValidValuesPrintNames) {
  EXPECT_EQ(StreamToString(absl::LogSeverity::kInfo), "INFO");
  EXPECT_EQ(StreamToString(absl::LogSeverity::kWarning), "WARNING");
  EXPECT_EQ(StreamToString(absl::LogSeverity::kError), "ERROR");
  EXPECT_EQ(StreamToString(absl::LogSeverity::kFatal), "FATAL");
}

TEST(LogSeverityStreamTest, BVA_LogSeverityLowestValidValuePrintsInfo) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(0)), "INFO");
}

TEST(LogSeverityStreamTest, BVA_LogSeverityHighestValidValuePrintsFatal) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(3)), "FATAL");
}

TEST(LogSeverityStreamTest, Invalid_LogSeverityBelowValidRangePrintsNumericWrapper) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(-1)),
            "absl::LogSeverity(-1)");
}

TEST(LogSeverityStreamTest, Invalid_LogSeverityAboveValidRangePrintsNumericWrapper) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(4)),
            "absl::LogSeverity(4)");
}

TEST(LogSeverityStreamTest, Invalid_LogSeverityLargePositiveValuePrintsNumericWrapper) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(999)),
            "absl::LogSeverity(999)");
}

TEST(LogSeverityStreamTest, Invalid_LogSeverityLargeNegativeValuePrintsNumericWrapper) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverity>(-999)),
            "absl::LogSeverity(-999)");
}

TEST(LogSeverityAtLeastStreamTest, ECP_ValidThresholdsPrintGreaterThanOrEqualPrefix) {
  EXPECT_EQ(StreamToString(absl::LogSeverityAtLeast::kInfo), ">=INFO");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtLeast::kWarning), ">=WARNING");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtLeast::kError), ">=ERROR");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtLeast::kFatal), ">=FATAL");
}

TEST(LogSeverityAtLeastStreamTest, ECP_InfinityPrintsInfinity) {
  EXPECT_EQ(StreamToString(absl::LogSeverityAtLeast::kInfinity), "INFINITY");
}

TEST(LogSeverityAtLeastStreamTest, BVA_LowestFiniteThresholdPrintsInfo) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverityAtLeast>(0)),
            ">=INFO");
}

TEST(LogSeverityAtLeastStreamTest, BVA_HighestFiniteThresholdPrintsFatal) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverityAtLeast>(3)),
            ">=FATAL");
}

TEST(LogSeverityAtLeastStreamTest, Invalid_UnknownThresholdDoesNotModifyStream) {
  std::ostringstream os;
  os << "prefix";
  os << static_cast<absl::LogSeverityAtLeast>(123);

  EXPECT_EQ(os.str(), "prefix");
}

TEST(LogSeverityAtLeastStreamTest, Invalid_NegativeUnknownThresholdDoesNotModifyStream) {
  std::ostringstream os;
  os << "prefix";
  os << static_cast<absl::LogSeverityAtLeast>(-123);

  EXPECT_EQ(os.str(), "prefix");
}

TEST(LogSeverityAtMostStreamTest, ECP_ValidThresholdsPrintLessThanOrEqualPrefix) {
  EXPECT_EQ(StreamToString(absl::LogSeverityAtMost::kInfo), "<=INFO");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtMost::kWarning), "<=WARNING");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtMost::kError), "<=ERROR");
  EXPECT_EQ(StreamToString(absl::LogSeverityAtMost::kFatal), "<=FATAL");
}

TEST(LogSeverityAtMostStreamTest, ECP_NegativeInfinityPrintsNegativeInfinity) {
  EXPECT_EQ(StreamToString(absl::LogSeverityAtMost::kNegativeInfinity),
            "NEGATIVE_INFINITY");
}

TEST(LogSeverityAtMostStreamTest, BVA_LowestFiniteThresholdPrintsInfo) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverityAtMost>(0)),
            "<=INFO");
}

TEST(LogSeverityAtMostStreamTest, BVA_HighestFiniteThresholdPrintsFatal) {
  EXPECT_EQ(StreamToString(static_cast<absl::LogSeverityAtMost>(3)),
            "<=FATAL");
}

TEST(LogSeverityAtMostStreamTest, Invalid_UnknownThresholdDoesNotModifyStream) {
  std::ostringstream os;
  os << "prefix";
  os << static_cast<absl::LogSeverityAtMost>(123);

  EXPECT_EQ(os.str(), "prefix");
}

TEST(LogSeverityAtMostStreamTest, Invalid_NegativeUnknownThresholdDoesNotModifyStream) {
  std::ostringstream os;
  os << "prefix";
  os << static_cast<absl::LogSeverityAtMost>(-123);

  EXPECT_EQ(os.str(), "prefix");
}

TEST(LogSeverityStreamTest, Edge_ChainedStreamingPreservesOrder) {
  std::ostringstream os;

  os << absl::LogSeverity::kInfo << "|"
     << absl::LogSeverityAtLeast::kWarning << "|"
     << absl::LogSeverityAtMost::kError;

  EXPECT_EQ(os.str(), "INFO|>=WARNING|<=ERROR");
}

TEST(LogSeverityStreamTest, Edge_OperatorReturnsOriginalStreamReference) {
  std::ostringstream os;

  std::ostream& returned = (os << absl::LogSeverity::kInfo);

  EXPECT_EQ(&returned, &os);
  EXPECT_EQ(os.str(), "INFO");
}

}  // namespace
ABSL_NAMESPACE_END
}  // namespace absl