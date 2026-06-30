#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include <algorithm>
#include <ostream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
namespace tensorflow {
namespace profiler {
namespace {
std::pair<TfFunctionExecutionMode, TfFunctionCompiler> Decode(
    absl::string_view function_name, absl::string_view mode) {
  if (mode == "eager") return {EAGER_MODE, INVALID_COMPILER};
  if (mode == "concrete") return {CONCRETE_MODE, INVALID_COMPILER};
  if (mode == "traced-xla") return {TRACED_MODE, XLA_COMPILER};
  if (mode == "traced-nonXla") return {TRACED_MODE, OTHER_COMPILER};
  if (mode == "notTraced-xla") return {NOT_TRACED_MODE, XLA_COMPILER};
  if (mode == "notTraced-nonXla") return {NOT_TRACED_MODE, OTHER_COMPILER};
  LOG(ERROR) << absl::StrCat("tf-function '", function_name,
                             "' has an unexpected execution mode '", mode, "'")
             << std::endl;
  return {INVALID_MODE, INVALID_COMPILER};
  DCHECK(false);
}
double ComputeExpensiveCallPercent(const TfFunction& tf_function) {
  uint64 total_call_time_ps = 0;
  uint64 expensive_call_time_ps = 0;
  for (const auto& mode_metrics : tf_function.metrics()) {
    const auto mode = mode_metrics.first;
    const auto& metrics = mode_metrics.second;
    total_call_time_ps += metrics.self_time_ps();
    if (mode == TRACED_MODE || mode == EAGER_MODE) {
      expensive_call_time_ps += metrics.self_time_ps();
    }
  }
  return tsl::profiler::SafeDivide(100.0 * expensive_call_time_ps,
                                   total_call_time_ps);
}
struct ActivationRecord {
  std::string function_name;               
  tsl::profiler::Timespan timespan;        
  TfFunctionExecutionMode execution_mode;  
  TfFunctionCompiler compiler;             
  int64_t tracing_count;  
  uint64 children_duration_ps;  
  ActivationRecord()
      : function_name(""),
        execution_mode(INVALID_MODE),
        compiler(INVALID_COMPILER),
        tracing_count(0),
        children_duration_ps(0) {}
  ActivationRecord(absl::string_view name,
                   const tsl::profiler::Timespan& timespan,
                   TfFunctionExecutionMode exe_mode,
                   TfFunctionCompiler compiler, int64_t tracing_cnt)
      : function_name(std::string(name)),
        timespan(timespan),
        execution_mode(exe_mode),
        compiler(compiler),
        tracing_count(tracing_cnt),
        children_duration_ps(0) {}
  std::string DebugString() const {
    return absl::StrCat("{", function_name, ", ",
                        TfFunctionExecutionMode_Name(execution_mode), ", ",
                        TfFunctionCompiler_Name(compiler),
                        ", tracing_count:", tracing_count,
                        ", children_duration:", children_duration_ps,
                        " ps, timespan:", timespan.DebugString(), "}");
  }
};
struct EntryOrExit {
  bool is_entry;        
  int64_t index;        
  uint64 timestamp_ps;  
  EntryOrExit() : is_entry(false), index(-1), timestamp_ps(0) {}
  EntryOrExit(bool is_entry, int64_t index, uint64 timestamp_ps)
      : is_entry(is_entry), index(index), timestamp_ps(timestamp_ps) {}
  std::string DebugString() const {
    std::string entry_or_exit = is_entry ? "entry, " : "exit,  ";
    return absl::StrCat("{", entry_or_exit, "idx:", index,
                        ", timestamp:", timestamp_ps, "}");
  }
};
TfFunctionCompiler CombineCompilers(TfFunctionCompiler a,
                                    TfFunctionCompiler b) {
  if (a == INVALID_COMPILER) return b;
  if (b == INVALID_COMPILER) return a;
  if (a == b) return a;
  return MIXED_COMPILER;
}
void CombineTfFunctionMetrics(const TfFunctionMetrics& src,
                              TfFunctionMetrics* dst) {
  dst->set_count(src.count() + dst->count());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
}
void CombineTfFunction(const TfFunction& src, TfFunction* dst) {
  dst->set_total_tracing_count(
      std::max(src.total_tracing_count(), dst->total_tracing_count()));
  dst->set_compiler(CombineCompilers(src.compiler(), dst->compiler()));
  for (const auto& mode_metrics : src.metrics()) {
    int32_t execution_mode = mode_metrics.first;
    const TfFunctionMetrics& src_metrics = mode_metrics.second;
    TfFunctionMetrics* dst_metrics =
        gtl::FindOrNull(*dst->mutable_metrics(), execution_mode);
    if (dst_metrics == nullptr) {
      (*dst->mutable_metrics())[execution_mode] = src_metrics;
    } else {
      CombineTfFunctionMetrics(src_metrics, dst_metrics);
    }
  }
  dst->set_expensive_call_percent(ComputeExpensiveCallPercent(*dst));
}
class TfFunctionExecutions {
 public:
  explicit TfFunctionExecutions(const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      absl::string_view mode;
      int64_t tracing_count = 0;
      event.ForEachStat([&mode, &tracing_count](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kTfFunctionCall:
            mode = stat.StrOrRefValue();
            break;
          case StatType::kTfFunctionTracingCount:
            tracing_count = stat.IntValue();
            break;
        }
      });
      if (mode.empty()) return;
      int64_t index = activations_.size();
      auto timespan = event.GetTimespan();
      auto mode_compiler = Decode(event.Name(), mode);
      ActivationRecord activation_record =
          ActivationRecord(event.Name(), timespan, mode_compiler.first,
                           mode_compiler.second, tracing_count);
      activations_.push_back(activation_record);
      EntryOrExit entry_point =
          EntryOrExit(true, index, timespan.begin_ps());
      EntryOrExit exit_point =
          EntryOrExit(false, index, timespan.end_ps());
      points_.push_back(entry_point);
      points_.push_back(exit_point);
    });
    auto ascending_in_timestamp = [](const EntryOrExit& a,
                                     const EntryOrExit& b) {
      return a.timestamp_ps < b.timestamp_ps;
    };
    absl::c_sort(points_, ascending_in_timestamp);
    CalculateChildrenDurations();
  }
  std::string DebugString() const {
    std::string result = "\nActivations:\n";
    for (int i = 0, end = activations_.size(); i < end; i++) {
      absl::StrAppend(&result, "[", i, "] ", activations_[i].DebugString(),
                      "\n");
    }
    absl::StrAppend(&result, "tf-function Entry/Exit Points:\n");
    for (const auto& pt : points_) {
      absl::StrAppend(&result, pt.DebugString(), "\n");
    }
    return result;
  }
  TfFunctionDb ConvertToTfFunctionDb() {
    TfFunctionDb result;
    for (const auto& record : activations_) {
      TfFunction* fun = &(*result.mutable_tf_functions())[record.function_name];
      fun->set_total_tracing_count(
          std::max(static_cast<int64_t>(fun->total_tracing_count()),
                   record.tracing_count));
      fun->set_compiler(CombineCompilers(fun->compiler(), record.compiler));
      uint64 self_time_ps =
          record.timespan.duration_ps() - record.children_duration_ps;
      TfFunctionMetrics* metrics =
          &(*fun->mutable_metrics())[record.execution_mode];
      metrics->set_count(metrics->count() + 1);
      metrics->set_self_time_ps(metrics->self_time_ps() + self_time_ps);
    }
    for (auto& name_fun : *result.mutable_tf_functions()) {
      TfFunction& fun = name_fun.second;
      fun.set_expensive_call_percent(ComputeExpensiveCallPercent(fun));
    }
    return result;
  }
  void CalculateChildrenDurations() {
    std::stack<int64_t> call_stack;
    for (const auto& pt : points_) {
      if (pt.is_entry) {
        call_stack.push(pt.index);
      } else {
        DCHECK(call_stack.top() == pt.index);  
        uint64 call_duration = activations_[pt.index].timespan.duration_ps();
        call_stack.pop();
        if (!call_stack.empty()) {
          activations_[call_stack.top()].children_duration_ps += call_duration;
        }
      }
    }
  }
 private:
  std::vector<ActivationRecord> activations_;
  std::vector<EntryOrExit> points_;
};
}  
std::string DebugString(const TfFunctionDb& tf_function_db) {
  std::string str;
  protobuf::TextFormat::PrintToString(tf_function_db, &str);
  return str;
}
void CombineTfFunctionDb(const TfFunctionDb& src, TfFunctionDb* dst) {
  for (const auto& name_function : src.tf_functions()) {
    const auto& name = name_function.first;
    const auto& src_fun = name_function.second;
    TfFunction* dst_fun = gtl::FindOrNull(*dst->mutable_tf_functions(), name);
    if (dst_fun == nullptr) {
      (*dst->mutable_tf_functions())[name] = src_fun;
    } else {
      CombineTfFunction(src_fun, dst_fun);
    }
  }
}
TfFunctionDb ConvertHostThreadsXLineToTfFunctionDb(const XLineVisitor& line) {
  TfFunctionExecutions tf_function_executions = TfFunctionExecutions(line);
  return tf_function_executions.ConvertToTfFunctionDb();
}
}  
}  