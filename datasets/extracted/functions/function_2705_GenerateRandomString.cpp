#include "tensorflow/lite/experimental/acceleration/mini_benchmark/blocking_validator_runner.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "flatbuffers/flatbuffer_builder.h"  
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_options.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
namespace tflite {
namespace acceleration {
namespace {
using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::GetRoot;
constexpr absl::Duration kWaitBetweenRefresh = absl::Milliseconds(20);
std::string GenerateRandomString() {
  static const char charset[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  const int size = 10;
  std::string result;
  result.resize(size);
  for (int i = 0; i < size; ++i) {
    result[i] = charset[rand() % (sizeof(charset) - 1)];
  }
  return result;
}
}  
BlockingValidatorRunner::BlockingValidatorRunner(
    const ValidatorRunnerOptions& options)
    : per_test_timeout_ms_(options.per_test_timeout_ms),
      storage_path_base_(options.storage_path) {
  validator_runner_impl_ = std::make_unique<ValidatorRunnerImpl>(
      CreateModelLoaderPath(options), options.storage_path,
      options.data_directory_path, options.per_test_timeout_ms,
      options.custom_input_data.empty()
          ? nullptr
          : std::make_unique<CustomValidationEmbedder>(
                options.custom_input_batch_size, options.custom_input_data,
                options.error_reporter),
      options.error_reporter, options.nnapi_sl, options.gpu_plugin_handle,
      options.validation_entrypoint_name, options.benchmark_result_evaluator);
}
MinibenchmarkStatus BlockingValidatorRunner::Init() {
  return validator_runner_impl_->Init();
}
std::vector<FlatBufferBuilder> BlockingValidatorRunner::TriggerValidation(
    const std::vector<const TFLiteSettings*>& for_settings) {
  if (for_settings.empty()) {
    return {};
  }
  std::string storage_path =
      absl::StrCat(storage_path_base_, ".", GenerateRandomString());
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Validation storage path: %s",
                  storage_path.c_str());
  std::vector<flatbuffers::FlatBufferBuilder> to_be_run;
  std::vector<TFLiteSettingsT> for_settings_obj;
  for_settings_obj.reserve(for_settings.size());
  for (auto settings : for_settings) {
    TFLiteSettingsT tflite_settings;
    settings->UnPackTo(&tflite_settings);
    flatbuffers::FlatBufferBuilder copy;
    copy.Finish(CreateTFLiteSettings(copy, &tflite_settings));
    to_be_run.emplace_back(std::move(copy));
    for_settings_obj.emplace_back(tflite_settings);
  }
  validator_runner_impl_->TriggerValidationAsync(std::move(to_be_run),
                                                 storage_path);
  int64_t total_timeout_ms = per_test_timeout_ms_ * (1 + for_settings.size());
  int64_t deadline_us = Validator::BootTimeMicros() + total_timeout_ms * 1000;
  bool within_timeout = true;
  while ((validator_runner_impl_->GetNumCompletedResults()) <
             for_settings.size() &&
         (within_timeout = Validator::BootTimeMicros() < deadline_us)) {
    usleep(absl::ToInt64Microseconds(kWaitBetweenRefresh));
  }
  std::vector<FlatBufferBuilder> results =
      validator_runner_impl_->GetCompletedResults();
  if (!within_timeout) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_WARNING,
        "Validation timed out after %ld ms. Return before all tests finished.",
        total_timeout_ms);
  } else if (for_settings.size() != results.size()) {
    TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                    "Validation completed.Started benchmarking for %d "
                    "TFLiteSettings, received %d results.",
                    for_settings.size(), results.size());
  }
  std::vector<TFLiteSettingsT> result_settings;
  result_settings.reserve(results.size());
  for (auto& result : results) {
    const BenchmarkEvent* event =
        GetRoot<BenchmarkEvent>(result.GetBufferPointer());
    TFLiteSettingsT event_settings;
    event->tflite_settings()->UnPackTo(&event_settings);
    result_settings.emplace_back(std::move(event_settings));
  }
  for (auto& settings_obj : for_settings_obj) {
    auto result_it =
        std::find(result_settings.begin(), result_settings.end(), settings_obj);
    if (result_it == result_settings.end()) {
      FlatBufferBuilder fbb;
      fbb.Finish(CreateBenchmarkEvent(
          fbb, CreateTFLiteSettings(fbb, &settings_obj),
          BenchmarkEventType_ERROR,  0,
          CreateBenchmarkError(fbb, BenchmarkStage_UNKNOWN,
                                0,  0,
                                0,
                               kMinibenchmarkCompletionEventMissing),
          Validator::BootTimeMicros(), Validator::WallTimeMicros()));
      results.emplace_back(std::move(fbb));
    }
  }
  (void)unlink(storage_path.c_str());
  return results;
}
}  
}  