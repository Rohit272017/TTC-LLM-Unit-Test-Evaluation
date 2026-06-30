#include "tsl/platform/cloud/curl_http_request.h"
#include <algorithm>
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/scanner.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/types.h"
#define CHECK_CURL_OK(expr) CHECK_EQ(expr, CURLE_OK)
namespace tsl {
namespace {
constexpr uint64 kVerboseOutput = 0;
class LibCurlProxy : public LibCurl {
 public:
  static LibCurlProxy* Load() {
    static LibCurlProxy* libcurl = []() -> LibCurlProxy* {
      curl_global_init(CURL_GLOBAL_ALL);
      return new LibCurlProxy;
    }();
    return libcurl;
  }
  CURL* curl_easy_init() override { return ::curl_easy_init(); }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            int (*param)(void* clientp, curl_off_t dltotal,
                                         curl_off_t dlnow, curl_off_t ultotal,
                                         curl_off_t ulnow)) override {
    return ::curl_easy_setopt(curl, option, param);
  }
  CURLcode curl_easy_perform(CURL* curl) override {
    return ::curl_easy_perform(curl);
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
    return ::curl_easy_getinfo(curl, info, value);
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
    return ::curl_easy_getinfo(curl, info, value);
  }
  void curl_easy_cleanup(CURL* curl) override {
    return ::curl_easy_cleanup(curl);
  }
  char* curl_easy_escape(CURL* curl, const char* str, int length) override {
    return ::curl_easy_escape(curl, str, length);
  }
  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
    return ::curl_slist_append(list, str);
  }
  void curl_slist_free_all(curl_slist* list) override {
    return ::curl_slist_free_all(list);
  }
  void curl_free(void* p) override { ::curl_free(p); }
};
}  
CurlHttpRequest::CurlHttpRequest() : CurlHttpRequest(LibCurlProxy::Load()) {}
CurlHttpRequest::CurlHttpRequest(LibCurl* libcurl, Env* env)
    : libcurl_(libcurl), env_(env) {
  default_response_buffer_.reserve(CURL_MAX_WRITE_SIZE);
  curl_ = libcurl_->curl_easy_init();
  CHECK(curl_ != nullptr) << "Couldn't initialize a curl session.";
  std::string value = "";
  TF_CHECK_OK(ReadStringFromEnvVar("CURL_CA_BUNDLE", "", &value));
  if (!value.empty()) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_CAINFO, value.c_str()));
  }
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_VERBOSE, kVerboseOutput));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_USERAGENT, "TSL"));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTP_VERSION,
                                           CURL_HTTP_VERSION_1_1));
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_NOPROGRESS, uint64{0}));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFODATA, this));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFOFUNCTION,
                                           &CurlHttpRequest::ProgressCallback));
  SetResultBuffer(&default_response_buffer_);
}
CurlHttpRequest::~CurlHttpRequest() {
  if (curl_headers_) {
    libcurl_->curl_slist_free_all(curl_headers_);
  }
  if (resolve_list_) {
    libcurl_->curl_slist_free_all(resolve_list_);
  }
  if (put_body_) {
    if (fclose(put_body_) != 0) {
      LOG(ERROR) << "fclose() failed: " << strerror(errno);
    }
  }
  if (curl_) {
    libcurl_->curl_easy_cleanup(curl_);
  }
}
string CurlHttpRequest::EscapeString(const string& str) {
  char* out_char_str = libcurl_->curl_easy_escape(curl_, str.c_str(), 0);
  string out_str(out_char_str);
  libcurl_->curl_free(out_char_str);
  return out_str;
}
void CurlHttpRequest::SetUri(const string& uri) {
  CheckNotSent();
  is_uri_set_ = true;
  uri_ = uri;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_URL, uri.c_str()));
}
void CurlHttpRequest::SetRange(uint64 start, uint64 end) {
  CheckNotSent();
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(
      curl_, CURLOPT_RANGE, strings::StrCat(start, "-", end).c_str()));
}
void CurlHttpRequest::AddHeader(const string& name, const string& value) {
  CheckNotSent();
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat(name, ": ", value).c_str());
}
void CurlHttpRequest::AddResolveOverride(const string& hostname, int64_t port,
                                         const string& ip_addr) {
  CheckNotSent();
  resolve_list_ = libcurl_->curl_slist_append(
      resolve_list_,
      strings::StrCat(hostname, ":", port, ":", ip_addr).c_str());
}
void CurlHttpRequest::AddAuthBearerHeader(const string& auth_token) {
  CheckNotSent();
  if (!auth_token.empty()) {
    AddHeader("Authorization", strings::StrCat("Bearer ", auth_token));
  }
}
void CurlHttpRequest::SetRequestStats(RequestStats* stats) {
  CheckNotSent();
  CHECK(stats_ == nullptr) << "SetRequestStats already called";
  stats_ = stats;
}
void CurlHttpRequest::SetDeleteRequest() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kDelete;
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE"));
}
absl::Status CurlHttpRequest::SetPutFromFile(const string& body_filepath,
                                             size_t offset) {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPut;
  if (put_body_) {
    if (fclose(put_body_) != 0) {
      LOG(ERROR) << "fclose() failed: " << strerror(errno);
    }
  }
  put_body_ = fopen(body_filepath.c_str(), "r");
  if (!put_body_) {
    return errors::InvalidArgument("Couldn't open the specified file: " +
                                   body_filepath);
  }
  fseek(put_body_, 0, SEEK_END);
  const auto size = ftell(put_body_) - offset;
  fseek(put_body_, offset, SEEK_SET);
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(put_body_)));
  return absl::OkStatus();
}
void CurlHttpRequest::SetPutEmptyBody() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPut;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1));
  AddHeader("Content-Length", "0");
  AddHeader("Transfer-Encoding", "identity");
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
}
void CurlHttpRequest::SetPostFromBuffer(const char* buffer, size_t size) {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPost;
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
  post_body_buffer_ = absl::string_view(buffer, size);
}
void CurlHttpRequest::SetPostEmptyBody() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPost;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1));
  AddHeader("Content-Length", "0");
  AddHeader("Transfer-Encoding", "identity");
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
}
void CurlHttpRequest::SetResultBuffer(std::vector<char>* out_buffer) {
  CheckNotSent();
  CHECK(out_buffer != nullptr);
  out_buffer->clear();
  response_buffer_ = out_buffer;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                                           &CurlHttpRequest::WriteCallback));
}
void CurlHttpRequest::SetResultBufferDirect(char* buffer, size_t size) {
  CHECK(buffer != nullptr);
  CheckNotSent();
  direct_response_ = DirectResponseState{buffer, size, 0, 0};
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(
      curl_, CURLOPT_WRITEFUNCTION, &CurlHttpRequest::WriteCallbackDirect));
}
bool CurlHttpRequest::IsDirectResponse() const {
  return direct_response_.buffer_ != nullptr;
}
size_t CurlHttpRequest::WriteCallbackDirect(const void* ptr, size_t size,
                                            size_t nmemb, void* userdata) {
  CHECK(ptr != nullptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(userdata);
  DirectResponseState* state = &that->direct_response_;
  CHECK(state->buffer_ != nullptr);
  CHECK(state->bytes_transferred_ <= state->buffer_size_);
  size_t curl_bytes_received = size * nmemb;
  size_t user_buffer_bytes_available =
      state->buffer_size_ - state->bytes_transferred_;
  size_t bytes_to_copy =
      std::min<size_t>(curl_bytes_received, user_buffer_bytes_available);
  memcpy(&state->buffer_[state->bytes_transferred_], ptr, bytes_to_copy);
  state->bytes_transferred_ += bytes_to_copy;
  state->bytes_received_ += curl_bytes_received;
  return bytes_to_copy;
}
size_t CurlHttpRequest::GetResultBufferDirectBytesTransferred() {
  CHECK(direct_response_.buffer_ != nullptr);
  return direct_response_.bytes_transferred_;
}
void CurlHttpRequest::SetTimeouts(uint32 connection, uint32 inactivity,
                                  uint32 total) {
  CheckNotSent();
  connect_timeout_secs_ = connection;
  inactivity_timeout_secs_ = inactivity;
  request_timeout_secs_ = total;
}
size_t CurlHttpRequest::WriteCallback(const void* ptr, size_t size,
                                      size_t nmemb, void* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->response_buffer_);
  const size_t bytes_to_copy = size * nmemb;
  that->response_buffer_->insert(
      that->response_buffer_->end(), reinterpret_cast<const char*>(ptr),
      reinterpret_cast<const char*>(ptr) + bytes_to_copy);
  return bytes_to_copy;
}
size_t CurlHttpRequest::ReadCallback(void* ptr, size_t size, size_t nmemb,
                                     FILE* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->post_body_read_ <= that->post_body_buffer_.size());
  const size_t bytes_to_copy = std::min(
      size * nmemb, that->post_body_buffer_.size() - that->post_body_read_);
  memcpy(ptr, that->post_body_buffer_.data() + that->post_body_read_,
         bytes_to_copy);
  that->post_body_read_ += bytes_to_copy;
  return bytes_to_copy;
}
size_t CurlHttpRequest::HeaderCallback(const void* ptr, size_t size,
                                       size_t nmemb, void* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  absl::string_view header(reinterpret_cast<const char*>(ptr), size * nmemb);
  absl::string_view name, value;
  if (strings::Scanner(header)
          .ScanEscapedUntil(':')
          .StopCapture()
          .OneLiteral(": ")
          .GetResult(&value, &name)) {
    string str_value(value);
    absl::StripTrailingAsciiWhitespace(&str_value);
    that->response_headers_[string(name)] = str_value;
  }
  return size * nmemb;
}
absl::Status CurlHttpRequest::Send() {
  CheckNotSent();
  CHECK(is_uri_set_) << "URI has not been set.";
  is_sent_ = true;
  if (curl_headers_) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, curl_headers_));
  }
  if (resolve_list_) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_RESOLVE, resolve_list_));
  }
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION,
                                           &CurlHttpRequest::HeaderCallback));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_TIMEOUT,
                                           request_timeout_secs_));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT,
                                           connect_timeout_secs_));
  char error_buffer[CURL_ERROR_SIZE] = {0};
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_ERRORBUFFER, error_buffer));
  if (stats_ != nullptr) {
    stats_->RecordRequest(this, uri_, method_);
  }
  const CURLcode curl_result = libcurl_->curl_easy_perform(curl_);
  TF_RETURN_IF_ERROR(CURLcodeToStatus(curl_result, error_buffer));
  double written_size = 0;
  CHECK_CURL_OK(libcurl_->curl_easy_getinfo(curl_, CURLINFO_SIZE_DOWNLOAD,
                                            &written_size));
  CHECK_CURL_OK(libcurl_->curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE,
                                            &response_code_));
  auto get_error_message = [this]() -> string {
    string error_message = strings::StrCat(
        "Error executing an HTTP request: HTTP response code ", response_code_);
    absl::string_view body = GetResponse();
    if (!body.empty()) {
      return strings::StrCat(
          error_message, " with body '",
          body.substr(0, std::min(body.size(), response_to_error_limit_)), "'");
    }
    return error_message;
  };
  absl::Status result;
  switch (response_code_) {
    case 200:  
    case 201:  
    case 204:  
    case 206:  
      result = absl::OkStatus();
      break;
    case 416:  
      response_buffer_->clear();
      if (IsDirectResponse()) {
        direct_response_.bytes_transferred_ = 0;
      }
      result = absl::OkStatus();
      break;
    case 400:  
    case 406:  
    case 411:  
    case 414:  
      result = errors::InvalidArgument(get_error_message());
      break;
    case 401:  
    case 403:  
    case 407:  
      result = errors::PermissionDenied(get_error_message());
      break;
    case 404:  
    case 410:  
      result = errors::NotFound(get_error_message());
      break;
    case 302:  
    case 303:  
    case 304:  
    case 307:  
    case 412:  
    case 413:  
      result = errors::FailedPrecondition(get_error_message());
      break;
    case 308:  
    case 409:  
    case 429:  
    case 500:  
    case 502:  
    case 503:  
    default:   
      result = errors::Unavailable(get_error_message());
      break;
  }
  if (!result.ok()) {
    response_buffer_->clear();
  }
  if (stats_ != nullptr) {
    stats_->RecordResponse(this, uri_, method_, result);
  }
  return result;
}
void CurlHttpRequest::CheckMethodNotSet() const {
  CHECK(!is_method_set_) << "HTTP method has been already set.";
}
void CurlHttpRequest::CheckNotSent() const {
  CHECK(!is_sent_) << "The request has already been sent.";
}
absl::string_view CurlHttpRequest::GetResponse() const {
  absl::string_view response;
  if (IsDirectResponse()) {
    response = absl::string_view(direct_response_.buffer_,
                                 direct_response_.bytes_transferred_);
  } else {
    response =
        absl::string_view(response_buffer_->data(), response_buffer_->size());
  }
  return response;
}
string CurlHttpRequest::GetResponseHeader(const string& name) const {
  const auto& header = response_headers_.find(name);
  return header != response_headers_.end() ? header->second : "";
}
uint64 CurlHttpRequest::GetResponseCode() const { return response_code_; }
int CurlHttpRequest::ProgressCallback(void* this_object, curl_off_t dltotal,
                                      curl_off_t dlnow, curl_off_t ultotal,
                                      curl_off_t ulnow) {
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  const auto now = that->env_->NowSeconds();
  const auto current_progress = dlnow + ulnow;
  if (that->last_progress_timestamp_ == 0 ||
      current_progress > that->last_progress_bytes_) {
    that->last_progress_timestamp_ = now;
    that->last_progress_bytes_ = current_progress;
    return 0;
  }
  if (now - that->last_progress_timestamp_ > that->inactivity_timeout_secs_) {
    double lookup_time = -1;
    const auto lookup_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_NAMELOOKUP_TIME, &lookup_time);
    double connect_time = -1;
    const auto connect_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_CONNECT_TIME, &connect_time);
    double pretransfer_time = -1;
    const auto pretransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_PRETRANSFER_TIME, &pretransfer_time);
    double starttransfer_time = -1;
    const auto starttransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_STARTTRANSFER_TIME, &starttransfer_time);
    LOG(ERROR) << "The transmission  of request " << this_object
               << " (URI: " << that->uri_ << ") has been stuck at "
               << current_progress << " of " << dltotal + ultotal
               << " bytes for " << now - that->last_progress_timestamp_
               << " seconds and will be aborted. CURL timing information: "
               << "lookup time: " << lookup_time << " ("
               << curl_easy_strerror(lookup_time_status)
               << "), connect time: " << connect_time << " ("
               << curl_easy_strerror(connect_time_status)
               << "), pre-transfer time: " << pretransfer_time << " ("
               << curl_easy_strerror(pretransfer_time_status)
               << "), start-transfer time: " << starttransfer_time << " ("
               << curl_easy_strerror(starttransfer_time_status) << ")";
    return 1;  
  }
  return 0;
}
absl::Status CurlHttpRequest::CURLcodeToStatus(CURLcode code,
                                               const char* error_buffer) {
  if (code == CURLE_OK) {
    return absl::OkStatus();
  }
  string error_message = strings::StrCat(
      "Error executing an HTTP request: libcurl code ", code, " meaning '",
      curl_easy_strerror(code), "', error details: ");
  if (code == CURLE_WRITE_ERROR && IsDirectResponse() &&
      direct_response_.bytes_received_ > direct_response_.buffer_size_) {
    string overflow_message = strings::StrCat(
        "Received ", direct_response_.bytes_received_, " response bytes ",
        "for a ", direct_response_.buffer_size_, "-byte buffer");
    uint64 response_code = 0;
    const CURLcode get_response_result = libcurl_->curl_easy_getinfo(
        curl_, CURLINFO_RESPONSE_CODE, &response_code);
    if (get_response_result == CURLE_OK && response_code == 416) {
      return absl::OkStatus();
    }
    return errors::FailedPrecondition(
        strings::StrCat(error_message, overflow_message));
  }
  if (code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_SSL_CACERT_BADFILE) {
    return errors::FailedPrecondition(
        strings::StrCat(error_message, error_buffer));
  }
  return errors::Unavailable(
      strings::StrCat(error_message, *error_buffer ? error_buffer : "(none)"));
}
}  