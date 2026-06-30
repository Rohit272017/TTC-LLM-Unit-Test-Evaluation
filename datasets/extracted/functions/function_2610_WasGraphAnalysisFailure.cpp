#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_TEST_MATCHERS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_TEST_MATCHERS_H_
#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tsl/platform/statusor.h"
template <typename T>
bool WasGraphAnalysisFailure(const absl::StatusOr<T>& status) {
  return (status.status() ==
          tensorflow::CompileToHloGraphAnalysisFailedError());
}
MATCHER(IsOkOrFiltered,
        "Status was OK or equal to the Graph Analysis failure") {
  bool is_ok = arg.ok();
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  return testing::ExplainMatchResult(
      testing::IsTrue(), is_ok || graph_analysis_failure, result_listener);
}
MATCHER_P2(IncrementedOrFiltered, metric, value,
           "Metric was incremented by value or Status equal to the Graph "
           "Analysis failure") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  return testing::ExplainMatchResult(testing::Eq(metric), value,
                                     result_listener);
}
MATCHER_P(ComputationProtoContains, regex,
          "If not a Graph Analysis failure then matches the computation result "
          "with the regex") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto proto = arg.value().computation->proto().DebugString();
  return testing::ExplainMatchResult(testing::ContainsRegex(regex), proto,
                                     result_listener);
}
MATCHER_P(XlaComputationProtoContains, regex,
          "If not a Graph Analysis failure then matches the computation result "
          "with the regex") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto proto = arg.value().proto().DebugString();
  return testing::ExplainMatchResult(testing::ContainsRegex(regex), proto,
                                     result_listener);
}
MATCHER_P(
    HasMlirModuleWith, expected,
    "If not a Graph Analysis failure then matches the mlir module result") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto actual = arg.value();
  return testing::ExplainMatchResult(testing::ContainsRegex(expected), actual,
                                     result_listener);
}
#endif  