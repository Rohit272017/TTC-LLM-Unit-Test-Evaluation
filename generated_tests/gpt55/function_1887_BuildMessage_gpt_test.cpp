#include "tensorflow/c/logging.h"

#include <gtest/gtest.h>

#include <cstdarg>
#include <string>

namespace {

TEST(TFLogTest, ECP_InfoLevelWithPlainMessageDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, "info message"));
}

TEST(TFLogTest, ECP_WarningLevelWithPlainMessageDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_WARNING, "warning message"));
}

TEST(TFLogTest, ECP_ErrorLevelWithPlainMessageDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_ERROR, "error message"));
}

TEST(TFLogTest, BVA_MinValidLogLevelInfoDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, "minimum valid level"));
}

TEST(TFLogTest, BVA_MaxNonFatalLogLevelErrorDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_ERROR, "maximum non fatal level"));
}

TEST(TFLogTest, Invalid_LevelBelowInfoReturnsWithoutLoggingOrCrash) {
  TF_LogLevel invalid_level = static_cast<TF_LogLevel>(TF_INFO - 1);

  EXPECT_NO_FATAL_FAILURE(TF_Log(invalid_level, "invalid low level"));
}

TEST(TFLogTest, Invalid_LevelAboveFatalReturnsWithoutLoggingOrCrash) {
  TF_LogLevel invalid_level = static_cast<TF_LogLevel>(TF_FATAL + 1);

  EXPECT_NO_FATAL_FAILURE(TF_Log(invalid_level, "invalid high level"));
}

TEST(TFLogTest, ECP_FormatStringWithIntegerArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, "value=%d", 42));
}

TEST(TFLogTest, ECP_FormatStringWithStringArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_WARNING, "name=%s", "TensorFlow"));
}

TEST(TFLogTest, ECP_FormatStringWithMultipleArgumentsDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(
      TF_Log(TF_ERROR, "id=%d name=%s ratio=%.2f", 7, "op", 3.14));
}

TEST(TFLogTest, BVA_EmptyFormatStringDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, ""));
}

TEST(TFLogTest, Edge_FormatStringWithPercentLiteralDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, "progress=100%%"));
}

TEST(TFLogTest, Edge_LongMessageDoesNotCrash) {
  std::string long_message(4096, 'x');

  EXPECT_NO_FATAL_FAILURE(TF_Log(TF_INFO, "%s", long_message.c_str()));
}

TEST(TFLogDeathTest, FatalLevelTerminatesProcess) {
  EXPECT_DEATH(TF_Log(TF_FATAL, "fatal message"), "fatal message");
}

TEST(TFVLogTest, ECP_VLogLevelZeroDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(0, "vlog level zero"));
}

TEST(TFVLogTest, ECP_VLogPositiveLevelDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, "vlog level one"));
}

TEST(TFVLogTest, ECP_VLogNegativeLevelDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(-1, "vlog negative level"));
}

TEST(TFVLogTest, ECP_VLogFormatStringWithIntegerArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, "value=%d", 123));
}

TEST(TFVLogTest, ECP_VLogFormatStringWithStringArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, "name=%s", "test"));
}

TEST(TFVLogTest, BVA_VLogEmptyFormatStringDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, ""));
}

TEST(TFVLogTest, Edge_VLogPercentLiteralDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, "progress=100%%"));
}

TEST(TFVLogTest, Edge_VLogLongMessageDoesNotCrash) {
  std::string long_message(4096, 'v');

  EXPECT_NO_FATAL_FAILURE(TF_VLog(1, "%s", long_message.c_str()));
}

TEST(TFDVLogTest, ECP_DVLogLevelZeroDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(0, "dvlog level zero"));
}

TEST(TFDVLogTest, ECP_DVLogPositiveLevelDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, "dvlog level one"));
}

TEST(TFDVLogTest, ECP_DVLogNegativeLevelDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(-1, "dvlog negative level"));
}

TEST(TFDVLogTest, ECP_DVLogFormatStringWithIntegerArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, "value=%d", 123));
}

TEST(TFDVLogTest, ECP_DVLogFormatStringWithStringArgumentDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, "name=%s", "test"));
}

TEST(TFDVLogTest, BVA_DVLogEmptyFormatStringDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, ""));
}

TEST(TFDVLogTest, Edge_DVLogPercentLiteralDoesNotCrash) {
  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, "progress=100%%"));
}

TEST(TFDVLogTest, Edge_DVLogLongMessageDoesNotCrash) {
  std::string long_message(4096, 'd');

  EXPECT_NO_FATAL_FAILURE(TF_DVLog(1, "%s", long_message.c_str()));
}

}  // namespace