#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include <string>
#include "flatbuffers/idl.h"  
#include "tensorflow/lite/acceleration/configuration/configuration_fbs_contents-inl.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/tools/logging.h"
namespace tflite {
namespace delegates {
namespace utils {
TfLiteSettingsJsonParser::TfLiteSettingsJsonParser() {
  TFLITE_DCHECK(parser_.Parse(configuration_fbs_contents) &&
                parser_.SetRootType("TFLiteSettings"));
}
const TFLiteSettings* TfLiteSettingsJsonParser::Parse(
    const std::string& json_file_path) {
  if (!LoadFromJsonFile(json_file_path) || buffer_pointer_ == nullptr) {
    return nullptr;
  }
  return flatbuffers::GetRoot<TFLiteSettings>(buffer_pointer_);
}
const uint8_t* TfLiteSettingsJsonParser::GetBufferPointer() {
  return buffer_pointer_;
}
flatbuffers::uoffset_t TfLiteSettingsJsonParser::GetBufferSize() {
  return buffer_size_;
}
bool TfLiteSettingsJsonParser::LoadFromJsonFile(
    const std::string& json_file_path) {
  buffer_size_ = 0;
  buffer_pointer_ = nullptr;
  if (json_file_path.empty()) {
    TFLITE_LOG(ERROR) << "Invalid JSON file path.";
    return false;
  }
  std::string json_file;
  if (!flatbuffers::LoadFile(json_file_path.c_str(), false, &json_file)) {
    TFLITE_LOG(ERROR) << "Failed to load the delegate settings file ("
                      << json_file_path << ").";
    return false;
  }
  if (!parser_.Parse(json_file.c_str())) {
    TFLITE_LOG(ERROR) << "Failed to parse the delegate settings file ("
                      << json_file_path << ").";
    return false;
  }
  buffer_size_ = parser_.builder_.GetSize();
  buffer_pointer_ = parser_.builder_.GetBufferPointer();
  return true;
}
}  
}  
}  