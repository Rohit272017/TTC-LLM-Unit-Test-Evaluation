#include "tensorflow/c/tf_status_helper.h"
#include <string>
#include "tensorflow/c/tf_status.h"
#include "xla/tsl/c/tsl_status_helper.h"
namespace tsl {
void Set_TF_Status_from_Status(TF_Status* tf_status,
                               const absl::Status& status) {
  TF_SetStatus(tf_status, TSLCodeFromStatusCode(status.code()),
               absl::StatusMessageAsCStr(status));
  status.ForEachPayload(
      [tf_status](absl::string_view key, const absl::Cord& value) {
        std::string key_str(key);
        std::string value_str(value);
        TF_SetPayload(tf_status, key_str.c_str(), value_str.c_str());
      });
}
absl::Status StatusFromTF_Status(const TF_Status* tf_status) {
  absl::Status status(StatusCodeFromTSLCode(TF_GetCode(tf_status)),
                      TF_Message(tf_status));
  TF_ForEachPayload(
      tf_status,
      [](const char* key, const char* value, void* capture) {
        absl::Status* status = static_cast<absl::Status*>(capture);
        status->SetPayload(key, absl::Cord(absl::string_view(value)));
      },
      &status);
  return status;
}
}  