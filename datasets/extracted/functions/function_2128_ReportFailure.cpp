#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/latency_benchmark.h"
#include <errno.h>
#include <sys/stat.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto/delegate_performance.pb.h"
namespace tflite {
namespace benchmark {
namespace latency {
namespace {
static constexpr char kBenchmarkToolName[] = "(BenchmarkModelAndroid)";
class DelegatePerformanceReportingListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& unused) override {
    results_proto_.set_event_type(proto::benchmark::BENCHMARK_EVENT_TYPE_START);
  }
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    ReportResult(results);
  }
  void ReportFailure(TfLiteStatus status) {
    std::string status_msg =
        status == kTfLiteError
            ? "TFLite error"
            : (status == kTfLiteDelegateError ? "TFLite delegate error"
                                              : "unexpected TFLite status");
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                    "Benchmark failed due to %s with status code %d.",
                    status_msg.c_str(), status);
    results_proto_.set_event_type(proto::benchmark::BENCHMARK_EVENT_TYPE_ERROR);
    results_proto_.mutable_error()->mutable_error_code()->set_tflite_error(
        status);
    results_proto_.mutable_error()->set_error_message(status_msg);
  }
  const proto::benchmark::LatencyResults& GetResults() {
    return results_proto_;
  }
 private:
  void ReportResult(const BenchmarkResults& results) {
    tensorflow::Stat<int64_t> warmup_us = results.warmup_time_us();
    tensorflow::Stat<int64_t> inference_us = results.inference_time_us();
    profiling::memory::MemoryUsage init_mem_usage = results.init_mem_usage();
    profiling::memory::MemoryUsage overall_mem_usage =
        results.overall_mem_usage();
    if (results.model_size_mb() > 0) {
      AddMetric("model_size_megabyte",
                results.model_size_mb());
    }
    AddMetric("initialization_latency_us",
              results.startup_latency_us());
    AddMetric("warmup_latency_average_us", warmup_us.avg());
    AddMetric("warmup_latency_min_us", warmup_us.min());
    AddMetric("warmup_latency_max_us", warmup_us.max());
    AddMetric("warmup_latency_standard_deviation_us",
              warmup_us.std_deviation());
    AddMetric("inference_latency_average_us",
              inference_us.avg());
    AddMetric("inference_latency_min_us",
              inference_us.min());
    AddMetric("inference_latency_max_us",
              inference_us.max());
    AddMetric("inference_latency_standard_deviation_us",
              inference_us.std_deviation());
    AddMetric("initialization_memory_max_rss_mebibyte",
              init_mem_usage.mem_footprint_kb / 1024.0);
    AddMetric("initialization_memory_total_non_mmapped_heap_mebibyte",
              init_mem_usage.total_allocated_bytes / 1024.0 / 1024.0);
    AddMetric(
        "initialization_memory_in_use_heap_mebibyte",
        init_mem_usage.in_use_allocated_bytes / 1024.0 / 1024.0);
    AddMetric("overall_memory_max_rss_mebibyte",
              overall_mem_usage.mem_footprint_kb / 1024.0);
    AddMetric(
        "overall_memory_total_non_mmapped_heap_mebibyte",
        overall_mem_usage.total_allocated_bytes / 1024.0 / 1024.0);
    AddMetric(
        "overall_memory_in_use_heap_mebibyte",
        overall_mem_usage.in_use_allocated_bytes / 1024.0 / 1024.0);
    results_proto_.set_event_type(proto::benchmark::BENCHMARK_EVENT_TYPE_END);
    TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Benchmark finished.");
  }
  void AddMetric(std::string name, float value) {
    proto::benchmark::BenchmarkMetric* metric = results_proto_.add_metrics();
    metric->set_name(name);
    metric->set_value(value);
  }
  proto::benchmark::LatencyResults results_proto_;
};
std::vector<std::string> ParseArgumentsFromTfLiteSettings(
    const TFLiteSettings& tflite_settings,
    const std::string& tflite_settings_path) {
  std::vector<std::string> args;
  if (tflite_settings_path.empty()) {
    return args;
  }
  if (tflite_settings.stable_delegate_loader_settings()) {
    args.push_back(absl::StrFormat("--stable_delegate_settings_file=%s",
                                   tflite_settings_path));
    return args;
  }
  switch (tflite_settings.delegate()) {
    case Delegate_XNNPACK: {
      args.push_back("--use_xnnpack=true");
      if (tflite_settings.xnnpack_settings()) {
        if (tflite_settings.xnnpack_settings()->num_threads()) {
          args.push_back(absl::StrFormat(
              "--num_threads=%d",
              tflite_settings.xnnpack_settings()->num_threads()));
        }
        if (tflite_settings.xnnpack_settings()->flags() ==
            XNNPackFlags_TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16) {
          args.push_back("--xnnpack_force_fp16=true");
        }
      }
      return args;
    }
    case Delegate_GPU: {
      args.push_back("--use_gpu=true");
      const tflite::GPUSettings* gpu_settings = tflite_settings.gpu_settings();
      if (gpu_settings) {
        if (gpu_settings->is_precision_loss_allowed()) {
          args.push_back("--gpu_precision_loss_allowed=true");
        }
        if (gpu_settings->enable_quantized_inference()) {
          args.push_back("--gpu_experimental_enable_quant=true");
        }
        if (gpu_settings->inference_preference() ==
            GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED) {
          args.push_back("--gpu_inference_for_sustained_speed=true");
        }
        if (gpu_settings->force_backend() == GPUBackend_OPENCL) {
          args.push_back("--gpu_backend=cl");
        } else if (gpu_settings->force_backend() == GPUBackend_OPENGL) {
          args.push_back("--gpu_backend=gl");
        }
        if (gpu_settings->cache_directory()) {
          args.push_back(
              absl::StrFormat("--delegate_serialize_dir=%s",
                              gpu_settings->cache_directory()->c_str()));
        }
        if (gpu_settings->model_token()) {
          args.push_back(absl::StrFormat("--delegate_serialize_token=%s",
                                         gpu_settings->model_token()->c_str()));
        }
      }
      break;
    }
    case Delegate_EDGETPU: {
      args.push_back("--use_edgetpu=true");
      break;
    }
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "Delegate type %s is not enabled by the latency module.",
                      EnumNameDelegate(tflite_settings.delegate()));
      break;
  }
  if (tflite_settings.disable_default_delegates()) {
    args.push_back("--use_xnnpack=false");
  }
  return args;
}
}  
proto::benchmark::LatencyResults Benchmark(
    const TFLiteSettings& tflite_settings,
    const std::string& tflite_settings_path, int model_fd, size_t model_offset,
    size_t model_size, const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(kBenchmarkToolName));
  std::string arg_graph =
      absl::StrCat("--graph=fd:", model_fd, ":", model_offset, ":", model_size);
  argv.push_back(const_cast<char*>(arg_graph.data()));
  std::vector<std::string> args_from_tflite_settings =
      ParseArgumentsFromTfLiteSettings(tflite_settings, tflite_settings_path);
  for (const std::string& arg : args_from_tflite_settings) {
    argv.push_back(const_cast<char*>(arg.data()));
  }
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.data()));
  }
  BenchmarkTfLiteModel benchmark;
  DelegatePerformanceReportingListener delegatePerformanceReporting;
  benchmark.AddListener(&delegatePerformanceReporting);
  TfLiteStatus status = benchmark.Run(argv.size(), argv.data());
  if (status != kTfLiteOk) {
    delegatePerformanceReporting.ReportFailure(status);
  }
  return delegatePerformanceReporting.GetResults();
}
}  
}  
}  