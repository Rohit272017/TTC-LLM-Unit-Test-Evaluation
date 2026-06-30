#include "tsl/platform/cloud/gcs_throttle.h"
#include <algorithm>
namespace tsl {
namespace {
EnvTime* get_default_env_time() {
  static EnvTime* default_env_time = new EnvTime;
  return default_env_time;
}
}  
GcsThrottle::GcsThrottle(EnvTime* env_time)
    : last_updated_secs_(env_time ? env_time->GetOverridableNowSeconds()
                                  : EnvTime::NowSeconds()),
      available_tokens_(0),
      env_time_(env_time ? env_time : get_default_env_time()) {}
bool GcsThrottle::AdmitRequest() {
  mutex_lock l(mu_);
  UpdateState();
  if (available_tokens_ < config_.tokens_per_request) {
    return false || !config_.enabled;
  }
  available_tokens_ -= config_.tokens_per_request;
  return true;
}
void GcsThrottle::RecordResponse(size_t num_bytes) {
  mutex_lock l(mu_);
  UpdateState();
  available_tokens_ -= request_bytes_to_tokens(num_bytes);
}
void GcsThrottle::SetConfig(GcsThrottleConfig config) {
  mutex_lock l(mu_);
  config_ = config;
  available_tokens_ = config.initial_tokens;
  last_updated_secs_ = env_time_->GetOverridableNowSeconds();
}
void GcsThrottle::UpdateState() {
  int64_t now = env_time_->GetOverridableNowSeconds();
  uint64 delta_secs =
      std::max(int64_t{0}, now - static_cast<int64_t>(last_updated_secs_));
  available_tokens_ += delta_secs * config_.token_rate;
  available_tokens_ = std::min(available_tokens_, config_.bucket_size);
  last_updated_secs_ = now;
}
}  