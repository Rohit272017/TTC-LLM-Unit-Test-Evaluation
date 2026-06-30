#include "tensorflow/core/util/stat_summarizer.h"
#include <iomanip>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/match.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
using Detail = StatsCalculator::Detail;
StatSummarizer::StatSummarizer(const StatSummarizerOptions& options)
    : stats_calculator_(new StatsCalculator(options)) {}
StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph)
    : stats_calculator_(new StatsCalculator(StatSummarizerOptions())) {}
StatSummarizer::~StatSummarizer() = default;
void StatSummarizer::Validate(const std::vector<TensorDescription>* outputs,
                              const NodeExecStats& ns) const {
  if (outputs->size() != ns.output_size()) {
    LOG(WARNING) << "Number of outputs changed between runs for '"
                 << ns.node_name() << "' - was " << outputs->size() << ", now "
                 << ns.output_size();
  } else {
    for (const auto& output : ns.output()) {
      const int32_t slot = output.slot();
      if ((slot < 0) || (slot >= ns.output_size())) {
        continue;
      }
      const auto& stored = (*outputs)[slot];
      const auto& current = output.tensor_description();
      bool do_tensors_match =
          (stored.dtype() == current.dtype()) &&
          (stored.shape().dim_size() == current.shape().dim_size());
      if (do_tensors_match) {
        for (int i = 0; i < stored.shape().dim_size(); ++i) {
          if (stored.shape().dim(i).size() != current.shape().dim(i).size()) {
            do_tensors_match = false;
            break;
          }
        }
      }
      if (!do_tensors_match) {
        LOG(WARNING) << "Output tensor changed between runs for '"
                     << ns.node_name();
      }
    }
  }
}
void StatSummarizer::PrintStepStats() const {
  string output = GetOutputString();
  std::istringstream iss(output);
  for (std::string line; std::getline(iss, line);) {
    LOG(INFO) << line;
  }
}
namespace {
std::string OpType(const DeviceStepStats& ds, const NodeExecStats& ns) {
  if (absl::StrContains(ds.device(), "/stream") ||
      absl::StrContains(ds.device(), "/memcpy")) {
    return "<>";
  }
  const std::string sep(" = ");
  const std::string& label = ns.timeline_label();
  std::string::size_type start = label.find(sep);
  if (start == std::string::npos) return "<>";
  start += sep.size();
  std::string::size_type end = label.find('(', start);
  if (end == std::string::npos) return "<>";
  return label.substr(start, end - start);
}
}  
void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  int64_t curr_total_us = 0;
  int64_t mem_total = 0;
  int node_num = 0;
  for (const auto& ds : step_stats.dev_stats()) {
    for (const auto& ns : ds.node_stats()) {
      if (absl::StrContains(ds.device(), "/stream") &&
          !absl::StrContains(ds.device(), "/stream:all")) {
        continue;
      }
      if (absl::StrContains(ds.device(), "/host:CPU")) {
        continue;
      }
      std::string name = ns.node_name();
      std::string op_type = "<>";
      if (absl::StrContains(ds.device(), "/stream")) {
        auto parts = str_util::Split(ns.node_name(), ':');
        if (parts.size() == 2) {
          name = parts[0] + " [Kernel]";
          op_type = "gpu:" + parts[1];
        }
      } else if (absl::StrContains(ds.device(), "/memcpy")) {
        auto parts = str_util::Split(ns.node_name(), ':');
        if (parts.size() == 2 || parts.size() == 3) {
          name = parts.front() + " [MemCpy]";
          op_type = "gpu:" + parts.back();
        }
      } else {
        op_type = OpType(ds, ns);
      }
      ++node_num;
      const int64_t curr_time = ns.all_end_rel_micros();
      curr_total_us += curr_time;
      auto output_result =
          outputs_.emplace(name, std::vector<TensorDescription>());
      std::vector<TensorDescription>* outputs = &(output_result.first->second);
      int64_t rel_end_us = curr_time;
      if (output_result.second) {
        outputs->resize(ns.output_size());
        for (const auto& output : ns.output()) {
          const int32_t slot = output.slot();
          if ((slot < 0) || (slot >= ns.output_size())) {
            continue;
          }
          (*outputs)[slot] = output.tensor_description();
        }
      }
      int64_t curr_node_mem = 0;
      for (const auto& mem : ns.memory()) {
        const int64_t mem_usage = mem.total_bytes();
        curr_node_mem += mem_usage;
      }
      stats_calculator_->AddNodeStats(name, op_type, node_num, rel_end_us,
                                      curr_node_mem);
      mem_total += curr_node_mem;
      Validate(outputs, ns);
    }
  }
  stats_calculator_->UpdateRunTotalUs(curr_total_us);
  stats_calculator_->UpdateMemoryUsed(mem_total);
}
void StatSummarizer::PrintOutputs() const {
  std::priority_queue<
      std::pair<int64_t, const std::pair<const std::string, Detail>*>>
      timings;
  for (const auto& entry : stats_calculator_->GetDetails()) {
    timings.emplace(-entry.second.run_order, &entry);
  }
  LOG(INFO) << "============ Node output tensor sizes in run order ========";
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();
    std::stringstream stream;
    const auto detail_outputs = outputs_.at(entry.second->first);
    stream << entry.second->first << "\t" << detail_outputs.size();
    for (const auto& tensor : detail_outputs) {
      stream << "\t" << DataTypeString(tensor.dtype());
      stream << "\t" << tensor.shape().dim_size();
      for (const auto& d : tensor.shape().dim()) {
        stream << "\t" << d.size();
      }
    }
    LOG(INFO) << stream.str();
  }
}
}  