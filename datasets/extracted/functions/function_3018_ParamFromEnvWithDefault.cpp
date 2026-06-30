#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_util.h"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/str_util.h"
namespace tfrt {
namespace tf {
double ParamFromEnvWithDefault(const char* var_name, double default_value) {
  const char* val = std::getenv(var_name);
  double num;
  return (val && tensorflow::strings::safe_strtod(val, &num)) ? num
                                                              : default_value;
}
std::vector<double> ParamFromEnvWithDefault(const char* var_name,
                                            std::vector<double> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<std::string> splits = tensorflow::str_util::Split(val, ",");
  std::vector<double> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    double num;
    if (tensorflow::strings::safe_strtod(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}
std::vector<int> ParamFromEnvWithDefault(const char* var_name,
                                         std::vector<int> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<std::string> splits = tensorflow::str_util::Split(val, ",");
  std::vector<int> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    int num;
    if (tensorflow::strings::safe_strto32(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}
bool ParamFromEnvBoolWithDefault(const char* var_name, bool default_value) {
  const char* val = std::getenv(var_name);
  return (val) ? absl::AsciiStrToLower(val) == "true" : default_value;
}
}  
}  