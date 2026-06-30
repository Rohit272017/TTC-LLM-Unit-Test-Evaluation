#ifndef TENSORFLOW_LITE_TESTING_JOIN_H_
#define TENSORFLOW_LITE_TESTING_JOIN_H_
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include "tensorflow/lite/string_type.h"
namespace tflite {
namespace testing {
template <typename T>
string JoinDefault(T* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << data[0];
  for (int i = 1; i < len; i++) {
    result << delimiter << data[i];
  }
  return result.str();
}
template <typename T>
string Join(T* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << std::setprecision(9) << data[0];
  for (int i = 1; i < len; i++) {
    result << std::setprecision(9) << delimiter << data[i];
  }
  return result.str();
}
template <>
inline string Join<uint8_t>(uint8_t* data, size_t len,
                            const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << static_cast<int>(data[0]);
  for (int i = 1; i < len; i++) {
    result << delimiter << static_cast<int>(data[i]);
  }
  return result.str();
}
template <>
inline string Join<int8_t>(int8_t* data, size_t len, const string& delimiter) {
  if (len == 0 || data == nullptr) {
    return "";
  }
  std::stringstream result;
  result << static_cast<int>(data[0]);
  for (int i = 1; i < len; i++) {
    result << delimiter << static_cast<int>(data[i]);
  }
  return result.str();
}
}  
}  
#endif  