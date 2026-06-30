#include "testdata_source.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
namespace i18n {
namespace addressinput {
const char kDataFileName[] = TEST_DATA_DIR "/countryinfo.txt";
namespace {
const char kNormalPrefix = '-';
const char kAggregatePrefix = '+';
const char kDataKeyPrefix[] = "data/";
const size_t kDataKeyPrefixLength = sizeof kDataKeyPrefix - 1;
const size_t kCldrRegionCodeLength = 2;
const size_t kAggregateDataKeyLength =
    kDataKeyPrefixLength + kCldrRegionCodeLength;
std::map<std::string, std::string> InitData(const std::string& src_path) {
  std::map<std::string, std::string> data;
  std::ifstream file(src_path);
  if (!file.is_open()) {
    std::cerr << "Error opening \"" << src_path << "\"." << '\n';
    std::exit(EXIT_FAILURE);
  }
  const std::string normal_prefix(1, kNormalPrefix);
  const std::string aggregate_prefix(1, kAggregatePrefix);
  std::string key;
  std::string value;
  auto last_data_it = data.end();
  auto aggregate_data_it = data.end();
  while (file.good()) {
    std::getline(file, key, '=');
    if (!key.empty()) {
      std::getline(file, value, '\n');
      last_data_it =
          data.emplace_hint(last_data_it, normal_prefix + key, value);
      if (key.compare(0,
                      kDataKeyPrefixLength,
                      kDataKeyPrefix,
                      kDataKeyPrefixLength) == 0) {
        if (aggregate_data_it != data.end() &&
            key.compare(0,
                        kAggregateDataKeyLength,
                        aggregate_data_it->first,
                        sizeof kAggregatePrefix,
                        kAggregateDataKeyLength) == 0) {
          aggregate_data_it->second.append(", \"" + key + "\": " + value);
        } else {
          assert(key.size() == kAggregateDataKeyLength);
          if (aggregate_data_it != data.end()) {
            aggregate_data_it->second.push_back('}');
          }
          const std::string& aggregate_key =
              aggregate_prefix + key.substr(0, kAggregateDataKeyLength);
          aggregate_data_it = data.emplace_hint(
              aggregate_data_it, aggregate_key, "{\"" + key + "\": " + value);
        }
      }
    }
  }
  file.close();
  return data;
}
const std::map<std::string, std::string>& GetData(const std::string& src_path) {
  static const std::map<std::string, std::string> kData(InitData(src_path));
  return kData;
}
}  
TestdataSource::TestdataSource(bool aggregate, const std::string& src_path)
    : aggregate_(aggregate), src_path_(src_path) {}
TestdataSource::TestdataSource(bool aggregate)
    : aggregate_(aggregate), src_path_(kDataFileName) {}
TestdataSource::~TestdataSource() = default;
void TestdataSource::Get(const std::string& key,
                         const Callback& data_ready) const {
  std::string prefixed_key(1, aggregate_ ? kAggregatePrefix : kNormalPrefix);
  prefixed_key += key;
  auto data_it = GetData(src_path_).find(prefixed_key);
  bool success = data_it != GetData(src_path_).end();
  std::string* data = nullptr;
  if (success) {
    data = new std::string(data_it->second);
  } else {
    success = true;
    data = new std::string("{}");
  }
  data_ready(success, key, data);
}
}  
}  