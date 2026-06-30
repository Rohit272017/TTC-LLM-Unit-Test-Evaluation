#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"
#include <ctype.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
namespace tflite {
void ReadAccelerationConfig(
    const char* config,
    const std::function<void(std::string, std::string, bool)>& consumer) {
  if (config) {
    std::istringstream istream{config};
    std::string curr_config_line;
    while (std::getline(istream, curr_config_line)) {
      curr_config_line.erase(
          curr_config_line.begin(),
          std::find_if_not(curr_config_line.begin(), curr_config_line.end(),
                           [](int ch) { return std::isspace(ch); }));
      if (curr_config_line.empty() || curr_config_line.at(0) == '#') {
        continue;
      }
      auto first_sep_pos =
          std::find(curr_config_line.begin(), curr_config_line.end(), ',');
      bool is_denylist = false;
      std::string key = curr_config_line;
      std::string value{};
      if (first_sep_pos != curr_config_line.end()) {
        key = std::string(curr_config_line.begin(), first_sep_pos);
        value = std::string(first_sep_pos + 1, curr_config_line.end());
      }
      if (key[0] == '-') {
        key = key.substr(1);
        is_denylist = true;
      }
      consumer(key, value, is_denylist);
    }
  }
}
}  