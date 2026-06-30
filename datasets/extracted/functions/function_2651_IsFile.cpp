#include "tsl/platform/cloud/google_auth_provider.h"
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#else
#include <sys/types.h>
#endif
#include <fstream>
#include <utility>
#include "absl/strings/match.h"
#include "json/json.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/retrying_utils.h"
namespace tsl {
namespace {
constexpr char kGoogleApplicationCredentials[] =
    "GOOGLE_APPLICATION_CREDENTIALS";
constexpr char kGoogleAuthTokenForTesting[] = "GOOGLE_AUTH_TOKEN_FOR_TESTING";
constexpr char kCloudSdkConfig[] = "CLOUDSDK_CONFIG";
constexpr char kNoGceCheck[] = "NO_GCE_CHECK";
constexpr char kGCloudConfigFolder[] = ".config/gcloud/";
constexpr char kWellKnownCredentialsFile[] =
    "application_default_credentials.json";
constexpr int kExpirationTimeMarginSec = 60;
constexpr char kOAuthV3Url[] = "https:
constexpr char kOAuthV4Url[] = "https:
constexpr char kGceTokenPath[] = "instance/service-accounts/default/token";
constexpr char kOAuthScope[] = "https:
bool IsFile(const string& filename) {
  std::ifstream fstream(filename.c_str());
  return fstream.good();
}
absl::Status GetEnvironmentVariableFileName(string* filename) {
  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  const char* result = std::getenv(kGoogleApplicationCredentials);
  if (!result || !IsFile(result)) {
    return errors::NotFound(strings::StrCat("$", kGoogleApplicationCredentials,
                                            " is not set or corrupt."));
  }
  *filename = result;
  return absl::OkStatus();
}
absl::Status GetWellKnownFileName(string* filename) {
  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  string config_dir;
  const char* config_dir_override = std::getenv(kCloudSdkConfig);
  if (config_dir_override) {
    config_dir = config_dir_override;
  } else {
    const char* home_dir = std::getenv("HOME");
    if (!home_dir) {
      return errors::FailedPrecondition("Could not read $HOME.");
    }
    config_dir = io::JoinPath(home_dir, kGCloudConfigFolder);
  }
  auto result = io::JoinPath(config_dir, kWellKnownCredentialsFile);
  if (!IsFile(result)) {
    return errors::NotFound(
        "Could not find the credentials file in the standard gcloud location.");
  }
  *filename = result;
  return absl::OkStatus();
}
}  
GoogleAuthProvider::GoogleAuthProvider(
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client)
    : GoogleAuthProvider(std::unique_ptr<OAuthClient>(new OAuthClient()),
                         std::move(compute_engine_metadata_client),
                         Env::Default()) {}
GoogleAuthProvider::GoogleAuthProvider(
    std::unique_ptr<OAuthClient> oauth_client,
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client,
    Env* env)
    : oauth_client_(std::move(oauth_client)),
      compute_engine_metadata_client_(
          std::move(compute_engine_metadata_client)),
      env_(env) {}
absl::Status GoogleAuthProvider::GetToken(string* t) {
  mutex_lock lock(mu_);
  const uint64 now_sec = env_->NowSeconds();
  if (now_sec + kExpirationTimeMarginSec < expiration_timestamp_sec_) {
    *t = current_token_;
    return absl::OkStatus();
  }
  if (GetTokenForTesting().ok()) {
    *t = current_token_;
    return absl::OkStatus();
  }
  auto token_from_files_status = GetTokenFromFiles();
  if (token_from_files_status.ok()) {
    *t = current_token_;
    return absl::OkStatus();
  }
  char* no_gce_check_var = std::getenv(kNoGceCheck);
  bool skip_gce_check = no_gce_check_var != nullptr &&
                        absl::EqualsIgnoreCase(no_gce_check_var, "true");
  absl::Status token_from_gce_status;
  if (skip_gce_check) {
    token_from_gce_status =
        absl::Status(absl::StatusCode::kCancelled,
                     strings::StrCat("GCE check skipped due to presence of $",
                                     kNoGceCheck, " environment variable."));
  } else {
    token_from_gce_status = GetTokenFromGce();
  }
  if (token_from_gce_status.ok()) {
    *t = current_token_;
    return absl::OkStatus();
  }
  if (skip_gce_check) {
    LOG(INFO)
        << "Attempting an empty bearer token since no token was retrieved "
        << "from files, and GCE metadata check was skipped.";
  } else {
    LOG(WARNING)
        << "All attempts to get a Google authentication bearer token failed, "
        << "returning an empty token. Retrieving token from files failed with "
           "\""
        << token_from_files_status.ToString() << "\"."
        << " Retrieving token from GCE failed with \""
        << token_from_gce_status.ToString() << "\".";
  }
  *t = "";
  if (skip_gce_check) {
    expiration_timestamp_sec_ = 0;
  } else {
    expiration_timestamp_sec_ = UINT64_MAX;
  }
  current_token_ = "";
  return absl::OkStatus();
}
absl::Status GoogleAuthProvider::GetTokenFromFiles() {
  string credentials_filename;
  if (!GetEnvironmentVariableFileName(&credentials_filename).ok() &&
      !GetWellKnownFileName(&credentials_filename).ok()) {
    return errors::NotFound("Could not locate the credentials file.");
  }
  Json::Value json;
  Json::Reader reader;
  std::ifstream credentials_fstream(credentials_filename);
  if (!reader.parse(credentials_fstream, json)) {
    return errors::FailedPrecondition(
        "Couldn't parse the JSON credentials file.");
  }
  if (json.isMember("refresh_token")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromRefreshTokenJson(
        json, kOAuthV3Url, &current_token_, &expiration_timestamp_sec_));
  } else if (json.isMember("private_key")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromServiceAccountJson(
        json, kOAuthV4Url, kOAuthScope, &current_token_,
        &expiration_timestamp_sec_));
  } else {
    return errors::FailedPrecondition(
        "Unexpected content of the JSON credentials file.");
  }
  return absl::OkStatus();
}
absl::Status GoogleAuthProvider::GetTokenFromGce() {
  std::vector<char> response_buffer;
  const uint64 request_timestamp_sec = env_->NowSeconds();
  TF_RETURN_IF_ERROR(compute_engine_metadata_client_->GetMetadata(
      kGceTokenPath, &response_buffer));
  absl::string_view response =
      absl::string_view(&response_buffer[0], response_buffer.size());
  TF_RETURN_IF_ERROR(oauth_client_->ParseOAuthResponse(
      response, request_timestamp_sec, &current_token_,
      &expiration_timestamp_sec_));
  return absl::OkStatus();
}
absl::Status GoogleAuthProvider::GetTokenForTesting() {
  const char* token = std::getenv(kGoogleAuthTokenForTesting);
  if (!token) {
    return errors::NotFound("The env variable for testing was not set.");
  }
  expiration_timestamp_sec_ = UINT64_MAX;
  current_token_ = token;
  return absl::OkStatus();
}
}  