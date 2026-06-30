#include "tensorflow/lite/testing/kernel_test/diff_analyzer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/testing/split.h"
namespace tflite {
namespace testing {
namespace {
float CalculateNormalizedMaxDiff(const std::vector<float>& base,
                                 const std::vector<float>& test) {
  float diff = 0;
  float base_max = 1e-6;
  for (int i = 0; i < base.size(); i++) {
    diff = std::max(diff, std::abs(base[i] - test[i]));
    base_max = std::max(base_max, base[i]);
  }
  return diff / base_max;
}
float CalculateNormalizedL2Norm(const std::vector<float>& base,
                                const std::vector<float>& test) {
  float l2_error = 0;
  float base_max = 1e-6;
  for (int i = 0; i < base.size(); i++) {
    float diff = base[i] - test[i];
    l2_error += diff * diff;
    base_max = std::max(base_max, base[i]);
  }
  l2_error /= base.size();
  return std::sqrt(l2_error) / base_max;
}
TfLiteStatus Populate(const string& filename,
                      std::unordered_map<string, std::vector<float>>* tensors) {
  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }
  std::ifstream file(filename);
  string content;
  while (std::getline(file, content, '\n')) {
    auto parts = Split<string>(content, ":");
    if (parts.size() != 2) {
      fprintf(stderr, "Expected <name>:<value>, got %s", content.c_str());
      return kTfLiteError;
    }
    tensors->insert(std::make_pair(parts[0], Split<float>(parts[1], ",")));
  }
  file.close();
  return kTfLiteOk;
}
}  
TfLiteStatus DiffAnalyzer::ReadFiles(const string& base, const string& test) {
  TF_LITE_ENSURE_STATUS(Populate(base, &base_tensors_));
  TF_LITE_ENSURE_STATUS(Populate(test, &test_tensors_));
  if (base_tensors_.size() != test_tensors_.size()) {
    fprintf(stderr, "Golden and test tensor dimensions don't match.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}
TfLiteStatus DiffAnalyzer::WriteReport(const string& filename) {
  if (filename.empty()) {
    fprintf(stderr, "Empty output file name.");
    return kTfLiteError;
  }
  std::ofstream output_file;
  output_file.open(filename, std::fstream::out | std::fstream::trunc);
  if (!output_file) {
    fprintf(stderr, "Failed to open output file %s.", filename.c_str());
    return kTfLiteError;
  }
  output_file << "Normalized L2 Error"
              << ","
              << "Normalized Max Diff"
              << "\n";
  for (const auto& item : base_tensors_) {
    const auto& name = item.first;
    if (!test_tensors_.count(name)) {
      fprintf(stderr, "Missing tensor %s in test tensors.", name.c_str());
      continue;
    }
    float l2_error =
        CalculateNormalizedL2Norm(base_tensors_[name], test_tensors_[name]);
    float max_diff =
        CalculateNormalizedMaxDiff(base_tensors_[name], test_tensors_[name]);
    output_file << name << ":" << l2_error << "," << max_diff << "\n";
  }
  output_file.close();
  return kTfLiteOk;
}
}  
}  