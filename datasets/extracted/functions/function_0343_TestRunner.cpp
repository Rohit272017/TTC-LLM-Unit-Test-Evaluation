#ifndef TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
#define TENSORFLOW_LITE_TESTING_TEST_RUNNER_H_
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "tensorflow/lite/string_type.h"
namespace tflite {
namespace testing {
class TestRunner {
 public:
  TestRunner() {}
  virtual ~TestRunner() {}
  virtual void LoadModel(const string& bin_file_path) = 0;
  virtual void LoadModel(const string& bin_file_path,
                         const string& signature) = 0;
  virtual void ReshapeTensor(const string& name, const string& csv_values) = 0;
  virtual void ResetTensor(const std::string& name) = 0;
  virtual string ReadOutput(const string& name) = 0;
  virtual void Invoke(const std::vector<std::pair<string, string>>& inputs) = 0;
  virtual bool CheckResults(
      const std::vector<std::pair<string, string>>& expected_outputs,
      const std::vector<std::pair<string, string>>& expected_output_shapes) = 0;
  virtual std::vector<string> GetOutputNames() = 0;
  virtual void AllocateTensors() = 0;
  void SetModelBaseDir(const string& path) {
    model_base_dir_ = path;
    if (path[path.length() - 1] != '/') {
      model_base_dir_ += "/";
    }
  }
  string GetFullPath(const string& path) { return model_base_dir_ + path; }
  void SetInvocationId(const string& id) { invocation_id_ = id; }
  const string& GetInvocationId() const { return invocation_id_; }
  void Invalidate(const string& error_message) {
    std::cerr << error_message << std::endl;
    error_message_ = error_message;
  }
  bool IsValid() const { return error_message_.empty(); }
  const string& GetErrorMessage() const { return error_message_; }
  void SetOverallSuccess(bool value) { overall_success_ = value; }
  bool GetOverallSuccess() const { return overall_success_; }
 protected:
  template <typename T>
  bool CheckSizes(size_t tensor_bytes, size_t num_values) {
    size_t num_tensor_elements = tensor_bytes / sizeof(T);
    if (num_tensor_elements != num_values) {
      Invalidate("Expected '" + std::to_string(num_tensor_elements) +
                 "' elements for a tensor, but only got '" +
                 std::to_string(num_values) + "'");
      return false;
    }
    return true;
  }
 private:
  string model_base_dir_;
  string invocation_id_;
  bool overall_success_ = true;
  string error_message_;
};
}  
}  
#endif  