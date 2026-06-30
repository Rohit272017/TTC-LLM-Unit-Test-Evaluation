#include "xla/service/custom_call_status_internal.h"
namespace xla {
std::optional<absl::string_view> CustomCallStatusGetMessage(
    const XlaCustomCallStatus* status) {
  return status->message;
}
}  
void XlaCustomCallStatusSetSuccess(XlaCustomCallStatus* status) {
  status->message = std::nullopt;
}
void XlaCustomCallStatusSetFailure(XlaCustomCallStatus* status,
                                   const char* message, size_t message_len) {
  status->message = std::string(message, 0, message_len);
}