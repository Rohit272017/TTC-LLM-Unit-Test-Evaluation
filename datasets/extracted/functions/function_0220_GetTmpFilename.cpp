#include "tsl/platform/cloud/gcs_file_system.h"
#include <stdio.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "tsl/platform/file_statistics.h"
#include "tsl/platform/strcat.h"
#ifdef _WIN32
#include <io.h>  
#endif
#include "absl/base/macros.h"
#include "json/json.h"
#include "tsl/platform/cloud/curl_http_request.h"
#include "tsl/platform/cloud/file_block_cache.h"
#include "tsl/platform/cloud/google_auth_provider.h"
#include "tsl/platform/cloud/ram_file_block_cache.h"
#include "tsl/platform/cloud/time_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/retrying_utils.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/stringprintf.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/lib/traceme.h"
#ifdef _WIN32
#ifdef DeleteFile
#undef DeleteFile
#endif
#endif
namespace tsl {
namespace {
constexpr char kGcsUriBase[] = "https:
constexpr char kGcsUploadUriBase[] =
    "https:
constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr char kBucketMetadataLocationKey[] = "location";
constexpr size_t kReadAppendableFileBufferSize = 1024 * 1024;  
constexpr int kGetChildrenDefaultPageSize = 1000;
constexpr uint64 HTTP_CODE_RESUME_INCOMPLETE = 308;
constexpr uint64 HTTP_CODE_PRECONDITION_FAILED = 412;
ABSL_DEPRECATED("Use GCS_READ_CACHE_BLOCK_SIZE_MB instead.")
constexpr char kReadaheadBufferSize[] = "GCS_READAHEAD_BUFFER_SIZE_BYTES";
constexpr char kStatCacheMaxAge[] = "GCS_STAT_CACHE_MAX_AGE";
constexpr uint64 kStatCacheDefaultMaxAge = 5;
constexpr char kStatCacheMaxEntries[] = "GCS_STAT_CACHE_MAX_ENTRIES";
constexpr size_t kStatCacheDefaultMaxEntries = 1024;
constexpr char kMatchingPathsCacheMaxAge[] = "GCS_MATCHING_PATHS_CACHE_MAX_AGE";
constexpr uint64 kMatchingPathsCacheDefaultMaxAge = 0;
constexpr char kMatchingPathsCacheMaxEntries[] =
    "GCS_MATCHING_PATHS_CACHE_MAX_ENTRIES";
constexpr size_t kMatchingPathsCacheDefaultMaxEntries = 1024;
constexpr size_t kBucketLocationCacheMaxEntries = 10;
constexpr size_t kCacheNeverExpire = std::numeric_limits<uint64>::max();
const FileStatistics DIRECTORY_STAT(0, 0, true);
constexpr char kResolveCacheSecs[] = "GCS_RESOLVE_REFRESH_SECS";
constexpr char kRequestConnectionTimeout[] =
    "GCS_REQUEST_CONNECTION_TIMEOUT_SECS";
constexpr char kRequestIdleTimeout[] = "GCS_REQUEST_IDLE_TIMEOUT_SECS";
constexpr char kMetadataRequestTimeout[] = "GCS_METADATA_REQUEST_TIMEOUT_SECS";
constexpr char kReadRequestTimeout[] = "GCS_READ_REQUEST_TIMEOUT_SECS";
constexpr char kWriteRequestTimeout[] = "GCS_WRITE_REQUEST_TIMEOUT_SECS";
constexpr char kAdditionalRequestHeader[] = "GCS_ADDITIONAL_REQUEST_HEADER";
constexpr char kThrottleRate[] = "GCS_THROTTLE_TOKEN_RATE";
constexpr char kThrottleBucket[] = "GCS_THROTTLE_BUCKET_SIZE";
constexpr char kTokensPerRequest[] = "GCS_TOKENS_PER_REQUEST";
constexpr char kInitialTokens[] = "GCS_INITIAL_TOKENS";
constexpr char kRetryConfigInitialDelayTimeUs[] =
    "GCS_RETRY_CONFIG_INIT_DELAY_TIME_US";
constexpr char kRetryConfigMaxDelayTimeUs[] =
    "GCS_RETRY_CONFIG_MAX_DELAY_TIME_US";
constexpr char kRetryConfigMaxRetries[] = "GCS_RETRY_CONFIG_MAX_RETRIES";
constexpr char kAllowedBucketLocations[] = "GCS_ALLOWED_BUCKET_LOCATIONS";
constexpr char kDetectZoneSentinelValue[] = "auto";
constexpr char kAppendMode[] = "GCS_APPEND_MODE";
constexpr char kComposeAppend[] = "compose";
absl::Status GetTmpFilename(string* filename) {
  *filename = io::GetTempFilename("");
  return absl::OkStatus();
}
string MaybeAppendSlash(const string& name) {
  if (name.empty()) {
    return "/";
  }
  if (name.back() != '/') {
    return strings::StrCat(name, "/");
  }
  return name;
}
string JoinGcsPath(const string& path, const string& subpath) {
  return strings::StrCat(MaybeAppendSlash(path), subpath);
}
std::set<string> AddAllSubpaths(const std::vector<string>& paths) {
  std::set<string> result;
  result.insert(paths.begin(), paths.end());
  for (const string& path : paths) {
    absl::string_view subpath = io::Dirname(path);
    while (!(subpath.empty() || subpath == "/")) {
      result.emplace(string(subpath));
      subpath = io::Dirname(subpath);
    }
  }
  return result;
}
absl::Status ParseJson(absl::string_view json, Json::Value* result) {
  Json::Reader reader;
  if (!reader.parse(json.data(), json.data() + json.size(), *result)) {
    return errors::Internal("Couldn't parse JSON response from GCS.");
  }
  return absl::OkStatus();
}
absl::Status ParseJson(const std::vector<char>& json, Json::Value* result) {
  return ParseJson(absl::string_view{json.data(), json.size()}, result);
}
absl::Status GetValue(const Json::Value& parent, const char* name,
                      Json::Value* result) {
  *result = parent.get(name, Json::Value::null);
  if (result->isNull()) {
    return errors::Internal("The field '", name,
                            "' was expected in the JSON response.");
  }
  return absl::OkStatus();
}
absl::Status GetStringValue(const Json::Value& parent, const char* name,
                            string* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isString()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a string.");
  }
  *result = result_value.asString();
  return absl::OkStatus();
}
absl::Status GetInt64Value(const Json::Value& parent, const char* name,
                           int64_t* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (result_value.isNumeric()) {
    *result = result_value.asInt64();
    return absl::OkStatus();
  }
  if (result_value.isString() &&
      strings::safe_strto64(result_value.asCString(), result)) {
    return absl::OkStatus();
  }
  return errors::Internal(
      "The field '", name,
      "' in the JSON response was expected to be a number.");
}
absl::Status GetBoolValue(const Json::Value& parent, const char* name,
                          bool* result) {
  Json::Value result_value;
  TF_RETURN_IF_ERROR(GetValue(parent, name, &result_value));
  if (!result_value.isBool()) {
    return errors::Internal(
        "The field '", name,
        "' in the JSON response was expected to be a boolean.");
  }
  *result = result_value.asBool();
  return absl::OkStatus();
}
RetryConfig GetGcsRetryConfig() {
  RetryConfig retryConfig(
       1000 * 1000,
       32 * 1000 * 1000,
       10);
  uint64 init_delay_time_us;
  if (GetEnvVar(kRetryConfigInitialDelayTimeUs, strings::safe_strtou64,
                &init_delay_time_us)) {
    retryConfig.init_delay_time_us = init_delay_time_us;
  }
  uint64 max_delay_time_us;
  if (GetEnvVar(kRetryConfigMaxDelayTimeUs, strings::safe_strtou64,
                &max_delay_time_us)) {
    retryConfig.max_delay_time_us = max_delay_time_us;
  }
  uint32 max_retries;
  if (GetEnvVar(kRetryConfigMaxRetries, strings::safe_strtou32, &max_retries)) {
    retryConfig.max_retries = max_retries;
  }
  VLOG(1) << "GCS RetryConfig: "
          << "init_delay_time_us = " << retryConfig.init_delay_time_us << " ; "
          << "max_delay_time_us = " << retryConfig.max_delay_time_us << " ; "
          << "max_retries = " << retryConfig.max_retries;
  return retryConfig;
}
class GcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn = std::function<absl::Status(
      const string& filename, uint64 offset, size_t n,
      absl::string_view* result, char* scratch)>;
  GcsRandomAccessFile(const string& filename, ReadFn read_fn)
      : filename_(filename), read_fn_(std::move(read_fn)) {}
  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }
  absl::Status Read(uint64 offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    return read_fn_(filename_, offset, n, result, scratch);
  }
 private:
  const string filename_;
  const ReadFn read_fn_;
};
class BufferedGcsRandomAccessFile : public RandomAccessFile {
 public:
  using ReadFn = std::function<absl::Status(
      const string& filename, uint64 offset, size_t n,
      absl::string_view* result, char* scratch)>;
  BufferedGcsRandomAccessFile(const string& filename, uint64 buffer_size,
                              ReadFn read_fn)
      : filename_(filename),
        read_fn_(std::move(read_fn)),
        buffer_size_(buffer_size),
        buffer_start_(0),
        buffer_end_is_past_eof_(false) {}
  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }
  absl::Status Read(uint64 offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    if (n > buffer_size_) {
      return read_fn_(filename_, offset, n, result, scratch);
    }
    {
      mutex_lock l(buffer_mutex_);
      size_t buffer_end = buffer_start_ + buffer_.size();
      size_t copy_size = 0;
      if (offset < buffer_end && offset >= buffer_start_) {
        copy_size = std::min(n, static_cast<size_t>(buffer_end - offset));
        memcpy(scratch, buffer_.data() + (offset - buffer_start_), copy_size);
        *result = absl::string_view(scratch, copy_size);
      }
      bool consumed_buffer_to_eof =
          offset + copy_size >= buffer_end && buffer_end_is_past_eof_;
      if (copy_size < n && !consumed_buffer_to_eof) {
        absl::Status status = FillBuffer(offset + copy_size);
        if (!status.ok() && !absl::IsOutOfRange(status)) {
          buffer_.resize(0);
          return status;
        }
        size_t remaining_copy = std::min(n - copy_size, buffer_.size());
        memcpy(scratch + copy_size, buffer_.data(), remaining_copy);
        copy_size += remaining_copy;
        *result = absl::string_view(scratch, copy_size);
      }
      if (copy_size < n) {
        buffer_end_is_past_eof_ = false;
        return errors::OutOfRange("EOF reached. Requested to read ", n,
                                  " bytes from ", offset, ".");
      }
    }
    return absl::OkStatus();
  }
 private:
  absl::Status FillBuffer(uint64 start) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(buffer_mutex_) {
    buffer_start_ = start;
    buffer_.resize(buffer_size_);
    absl::string_view str_piece;
    absl::Status status = read_fn_(filename_, buffer_start_, buffer_size_,
                                   &str_piece, &(buffer_[0]));
    buffer_end_is_past_eof_ = absl::IsOutOfRange(status);
    buffer_.resize(str_piece.size());
    return status;
  }
  const string filename_;
  const ReadFn read_fn_;
  const uint64 buffer_size_;
  mutable mutex buffer_mutex_;
  mutable uint64 buffer_start_ TF_GUARDED_BY(buffer_mutex_);
  mutable bool buffer_end_is_past_eof_ TF_GUARDED_BY(buffer_mutex_);
  mutable string buffer_ TF_GUARDED_BY(buffer_mutex_);
};
typedef std::function<absl::Status(
    uint64 start_offset, const std::string& object_to_upload,
    const std::string& bucket, uint64 file_size, const std::string& gcs_path,
    UploadSessionHandle* session_handle)>
    SessionCreator;
typedef std::function<absl::Status(
    const std::string& session_uri, uint64 start_offset,
    uint64 already_uploaded, const std::string& tmp_content_filename,
    uint64 file_size, const std::string& file_path)>
    ObjectUploader;
typedef std::function<absl::Status(const string& session_uri, uint64 file_size,
                                   const std::string& gcs_path, bool* completed,
                                   uint64* uploaded)>
    StatusPoller;
typedef std::function<absl::Status(const string& fname, const string& bucket,
                                   const string& object, int64_t* generation)>
    GenerationGetter;
class GcsWritableFile : public WritableFile {
 public:
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config, bool compose_append,
                  SessionCreator session_creator,
                  ObjectUploader object_uploader, StatusPoller status_poller,
                  GenerationGetter generation_getter)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config),
        compose_append_(compose_append),
        start_offset_(0),
        session_creator_(std::move(session_creator)),
        object_uploader_(std::move(object_uploader)),
        status_poller_(std::move(status_poller)),
        generation_getter_(std::move(generation_getter)) {
    VLOG(3) << "GcsWritableFile: " << GetGcsPath();
    if (GetTmpFilename(&tmp_content_filename_).ok()) {
      outfile_.open(tmp_content_filename_,
                    std::ofstream::binary | std::ofstream::app);
    }
  }
  GcsWritableFile(const string& bucket, const string& object,
                  GcsFileSystem* filesystem, const string& tmp_content_filename,
                  GcsFileSystem::TimeoutConfig* timeouts,
                  std::function<void()> file_cache_erase,
                  RetryConfig retry_config, bool compose_append,
                  SessionCreator session_creator,
                  ObjectUploader object_uploader, StatusPoller status_poller,
                  GenerationGetter generation_getter)
      : bucket_(bucket),
        object_(object),
        filesystem_(filesystem),
        timeouts_(timeouts),
        file_cache_erase_(std::move(file_cache_erase)),
        sync_needed_(true),
        retry_config_(retry_config),
        compose_append_(compose_append),
        start_offset_(0),
        session_creator_(std::move(session_creator)),
        object_uploader_(std::move(object_uploader)),
        status_poller_(std::move(status_poller)),
        generation_getter_(std::move(generation_getter)) {
    VLOG(3) << "GcsWritableFile: " << GetGcsPath() << "with existing file "
            << tmp_content_filename;
    tmp_content_filename_ = tmp_content_filename;
    outfile_.open(tmp_content_filename_,
                  std::ofstream::binary | std::ofstream::app);
  }
  ~GcsWritableFile() override {
    Close().IgnoreError();
    std::remove(tmp_content_filename_.c_str());
  }
  absl::Status Append(absl::string_view data) override {
    TF_RETURN_IF_ERROR(CheckWritable());
    VLOG(3) << "Append: " << GetGcsPath() << " size " << data.length();
    sync_needed_ = true;
    outfile_ << data;
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not append to the internal temporary file.");
    }
    return absl::OkStatus();
  }
  absl::Status Close() override {
    VLOG(3) << "Close:" << GetGcsPath();
    if (outfile_.is_open()) {
      absl::Status sync_status = Sync();
      if (sync_status.ok()) {
        outfile_.close();
      }
      return sync_status;
    }
    return absl::OkStatus();
  }
  absl::Status Flush() override {
    VLOG(3) << "Flush:" << GetGcsPath();
    return Sync();
  }
  absl::Status Name(absl::string_view* result) const override {
    *result = object_;
    return absl::OkStatus();
  }
  absl::Status Sync() override {
    VLOG(3) << "Sync started:" << GetGcsPath();
    TF_RETURN_IF_ERROR(CheckWritable());
    if (!sync_needed_) {
      return absl::OkStatus();
    }
    absl::Status status = SyncImpl();
    VLOG(3) << "Sync finished " << GetGcsPath();
    if (status.ok()) {
      sync_needed_ = false;
    }
    return status;
  }
  absl::Status Tell(int64_t* position) override {
    *position = outfile_.tellp();
    if (*position == -1) {
      return errors::Internal("tellp on the internal temporary file failed");
    }
    return absl::OkStatus();
  }
 private:
  absl::Status SyncImpl() {
    outfile_.flush();
    if (!outfile_.good()) {
      return errors::Internal(
          "Could not write to the internal temporary file.");
    }
    UploadSessionHandle session_handle;
    uint64 start_offset = 0;
    string object_to_upload = object_;
    bool should_compose = false;
    if (compose_append_) {
      start_offset = start_offset_;
      should_compose = start_offset > 0;
      if (should_compose) {
        object_to_upload =
            strings::StrCat(io::Dirname(object_), "/.tmpcompose/",
                            io::Basename(object_), ".", start_offset_);
      }
    }
    TF_RETURN_IF_ERROR(CreateNewUploadSession(start_offset, object_to_upload,
                                              &session_handle));
    uint64 already_uploaded = 0;
    bool first_attempt = true;
    const absl::Status upload_status = RetryingUtils::CallWithRetries(
        [&first_attempt, &already_uploaded, &session_handle, &start_offset,
         this]() {
          if (session_handle.resumable && !first_attempt) {
            bool completed;
            TF_RETURN_IF_ERROR(RequestUploadSessionStatus(
                session_handle.session_uri, &completed, &already_uploaded));
            LOG(INFO) << "### RequestUploadSessionStatus: completed = "
                      << completed
                      << ", already_uploaded = " << already_uploaded
                      << ", file = " << GetGcsPath();
            if (completed) {
              file_cache_erase_();
              return absl::OkStatus();
            }
          }
          first_attempt = false;
          return UploadToSession(session_handle.session_uri, start_offset,
                                 already_uploaded);
        },
        retry_config_);
    if (absl::IsNotFound(upload_status)) {
      return errors::Unavailable(
          strings::StrCat("Upload to gs:
                          " failed, caused by: ", upload_status.message()));
    }
    if (upload_status.ok()) {
      if (should_compose) {
        TF_RETURN_IF_ERROR(AppendObject(object_to_upload));
      }
      TF_RETURN_IF_ERROR(GetCurrentFileSize(&start_offset_));
    }
    return upload_status;
  }
  absl::Status CheckWritable() const {
    if (!outfile_.is_open()) {
      return errors::FailedPrecondition(
          "The internal temporary file is not writable.");
    }
    return absl::OkStatus();
  }
  absl::Status GetCurrentFileSize(uint64* size) {
    const auto tellp = outfile_.tellp();
    if (tellp == static_cast<std::streampos>(-1)) {
      return errors::Internal(
          "Could not get the size of the internal temporary file.");
    }
    *size = tellp;
    return absl::OkStatus();
  }
  absl::Status CreateNewUploadSession(uint64 start_offset,
                                      std::string object_to_upload,
                                      UploadSessionHandle* session_handle) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    return session_creator_(start_offset, object_to_upload, bucket_, file_size,
                            GetGcsPath(), session_handle);
  }
  absl::Status AppendObject(string append_object) {
    const string append_object_path = GetGcsPathWithObject(append_object);
    VLOG(3) << "AppendObject: " << append_object_path << " to " << GetGcsPath();
    int64_t generation = 0;
    TF_RETURN_IF_ERROR(
        generation_getter_(GetGcsPath(), bucket_, object_, &generation));
    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&append_object, &generation, this]() {
          std::unique_ptr<HttpRequest> request;
          TF_RETURN_IF_ERROR(filesystem_->CreateHttpRequest(&request));
          request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket_, "/o/",
                                          request->EscapeString(object_),
                                          "/compose"));
          const string request_body = strings::StrCat(
              "{'sourceObjects': [{'name': '", object_,
              "','objectPrecondition':{'ifGenerationMatch':", generation,
              "}},{'name': '", append_object, "'}]}");
          request->SetTimeouts(timeouts_->connect, timeouts_->idle,
                               timeouts_->metadata);
          request->AddHeader("content-type", "application/json");
          request->SetPostFromBuffer(request_body.c_str(), request_body.size());
          TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(),
                                          " when composing to ", GetGcsPath());
          return absl::OkStatus();
        },
        retry_config_));
    return RetryingUtils::DeleteWithRetries(
        [&append_object_path, this]() {
          return filesystem_->DeleteFile(append_object_path, nullptr);
        },
        retry_config_);
  }
  absl::Status RequestUploadSessionStatus(const string& session_uri,
                                          bool* completed, uint64* uploaded) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    return status_poller_(session_uri, file_size, GetGcsPath(), completed,
                          uploaded);
  }
  absl::Status UploadToSession(const string& session_uri, uint64 start_offset,
                               uint64 already_uploaded) {
    uint64 file_size;
    TF_RETURN_IF_ERROR(GetCurrentFileSize(&file_size));
    absl::Status status =
        object_uploader_(session_uri, start_offset, already_uploaded,
                         tmp_content_filename_, file_size, GetGcsPath());
    if (status.ok()) {
      file_cache_erase_();
    }
    return status;
  }
  string GetGcsPathWithObject(string object) const {
    return strings::StrCat("gs:
  }
  string GetGcsPath() const { return GetGcsPathWithObject(object_); }
  string bucket_;
  string object_;
  GcsFileSystem* const filesystem_;  
  string tmp_content_filename_;
  std::ofstream outfile_;
  GcsFileSystem::TimeoutConfig* timeouts_;
  std::function<void()> file_cache_erase_;
  bool sync_needed_;  
  RetryConfig retry_config_ = GetGcsRetryConfig();
  bool compose_append_;
  uint64 start_offset_;
  const SessionCreator session_creator_;
  const ObjectUploader object_uploader_;
  const StatusPoller status_poller_;
  const GenerationGetter generation_getter_;
};
class GcsReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  GcsReadOnlyMemoryRegion(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }
 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};
bool StringPieceIdentity(absl::string_view str, absl::string_view* value) {
  *value = str;
  return true;
}
bool SplitByCommaToLowercaseSet(absl::string_view list,
                                std::unordered_set<string>* set) {
  std::vector<string> vector = absl::StrSplit(absl::AsciiStrToLower(list), ',');
  *set = std::unordered_set<string>(vector.begin(), vector.end());
  return true;
}
string ZoneToRegion(string* zone) {
  return zone->substr(0, zone->find_last_of('-'));
}
}  
GcsFileSystem::GcsFileSystem(bool make_default_cache) {
  uint64 value;
  block_size_ = kDefaultBlockSize;
  size_t max_bytes = kDefaultMaxCacheSize;
  uint64 max_staleness = kDefaultMaxStaleness;
  http_request_factory_ = std::make_shared<CurlHttpRequest::Factory>();
  compute_engine_metadata_client_ =
      std::make_shared<ComputeEngineMetadataClient>(http_request_factory_);
  auth_provider_ = std::unique_ptr<AuthProvider>(
      new GoogleAuthProvider(compute_engine_metadata_client_));
  zone_provider_ = std::unique_ptr<ZoneProvider>(
      new ComputeEngineZoneProvider(compute_engine_metadata_client_));
  if (GetEnvVar(kReadaheadBufferSize, strings::safe_strtou64, &value)) {
    block_size_ = value;
  }
  if (GetEnvVar(kBlockSize, strings::safe_strtou64, &value)) {
    block_size_ = value * 1024 * 1024;
  }
  if (GetEnvVar(kMaxCacheSize, strings::safe_strtou64, &value)) {
    max_bytes = value * 1024 * 1024;
  }
  if (GetEnvVar(kMaxStaleness, strings::safe_strtou64, &value)) {
    max_staleness = value;
  }
  if (!make_default_cache) {
    max_bytes = 0;
  }
  VLOG(1) << "GCS cache max size = " << max_bytes << " ; "
          << "block size = " << block_size_ << " ; "
          << "max staleness = " << max_staleness;
  file_block_cache_ = MakeFileBlockCache(block_size_, max_bytes, max_staleness);
  uint64 stat_cache_max_age = kStatCacheDefaultMaxAge;
  size_t stat_cache_max_entries = kStatCacheDefaultMaxEntries;
  if (GetEnvVar(kStatCacheMaxAge, strings::safe_strtou64, &value)) {
    stat_cache_max_age = value;
  }
  if (GetEnvVar(kStatCacheMaxEntries, strings::safe_strtou64, &value)) {
    stat_cache_max_entries = value;
  }
  stat_cache_.reset(new ExpiringLRUCache<GcsFileStat>(stat_cache_max_age,
                                                      stat_cache_max_entries));
  uint64 matching_paths_cache_max_age = kMatchingPathsCacheDefaultMaxAge;
  size_t matching_paths_cache_max_entries =
      kMatchingPathsCacheDefaultMaxEntries;
  if (GetEnvVar(kMatchingPathsCacheMaxAge, strings::safe_strtou64, &value)) {
    matching_paths_cache_max_age = value;
  }
  if (GetEnvVar(kMatchingPathsCacheMaxEntries, strings::safe_strtou64,
                &value)) {
    matching_paths_cache_max_entries = value;
  }
  matching_paths_cache_.reset(new ExpiringLRUCache<std::vector<string>>(
      matching_paths_cache_max_age, matching_paths_cache_max_entries));
  bucket_location_cache_.reset(new ExpiringLRUCache<string>(
      kCacheNeverExpire, kBucketLocationCacheMaxEntries));
  int64_t resolve_frequency_secs;
  if (GetEnvVar(kResolveCacheSecs, strings::safe_strto64,
                &resolve_frequency_secs)) {
    dns_cache_.reset(new GcsDnsCache(resolve_frequency_secs));
    VLOG(1) << "GCS DNS cache is enabled.  " << kResolveCacheSecs << " = "
            << resolve_frequency_secs;
  } else {
    VLOG(1) << "GCS DNS cache is disabled, because " << kResolveCacheSecs
            << " = 0 (or is not set)";
  }
  absl::string_view add_header_contents;
  if (GetEnvVar(kAdditionalRequestHeader, StringPieceIdentity,
                &add_header_contents)) {
    size_t split = add_header_contents.find(':', 0);
    if (split != absl::string_view::npos) {
      absl::string_view header_name = add_header_contents.substr(0, split);
      absl::string_view header_value = add_header_contents.substr(split + 1);
      if (!header_name.empty() && !header_value.empty()) {
        additional_header_.reset(new std::pair<const string, const string>(
            string(header_name), string(header_value)));
        VLOG(1) << "GCS additional header ENABLED. "
                << "Name: " << additional_header_->first << ", "
                << "Value: " << additional_header_->second;
      } else {
        LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                   << add_header_contents;
      }
    } else {
      LOG(ERROR) << "GCS additional header DISABLED. Invalid contents: "
                 << add_header_contents;
    }
  } else {
    VLOG(1) << "GCS additional header DISABLED. No environment variable set.";
  }
  uint32 timeout_value;
  if (GetEnvVar(kRequestConnectionTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.connect = timeout_value;
  }
  if (GetEnvVar(kRequestIdleTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.idle = timeout_value;
  }
  if (GetEnvVar(kMetadataRequestTimeout, strings::safe_strtou32,
                &timeout_value)) {
    timeouts_.metadata = timeout_value;
  }
  if (GetEnvVar(kReadRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.read = timeout_value;
  }
  if (GetEnvVar(kWriteRequestTimeout, strings::safe_strtou32, &timeout_value)) {
    timeouts_.write = timeout_value;
  }
  int64_t token_value;
  if (GetEnvVar(kThrottleRate, strings::safe_strto64, &token_value)) {
    GcsThrottleConfig config;
    config.enabled = true;
    config.token_rate = token_value;
    if (GetEnvVar(kThrottleBucket, strings::safe_strto64, &token_value)) {
      config.bucket_size = token_value;
    }
    if (GetEnvVar(kTokensPerRequest, strings::safe_strto64, &token_value)) {
      config.tokens_per_request = token_value;
    }
    if (GetEnvVar(kInitialTokens, strings::safe_strto64, &token_value)) {
      config.initial_tokens = token_value;
    }
    throttle_.SetConfig(config);
  }
  GetEnvVar(kAllowedBucketLocations, SplitByCommaToLowercaseSet,
            &allowed_locations_);
  absl::string_view append_mode;
  GetEnvVar(kAppendMode, StringPieceIdentity, &append_mode);
  if (append_mode == kComposeAppend) {
    compose_append_ = true;
  } else {
    compose_append_ = false;
  }
  retry_config_ = GetGcsRetryConfig();
}
GcsFileSystem::GcsFileSystem(
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory,
    std::unique_ptr<ZoneProvider> zone_provider, size_t block_size,
    size_t max_bytes, uint64 max_staleness, uint64 stat_cache_max_age,
    size_t stat_cache_max_entries, uint64 matching_paths_cache_max_age,
    size_t matching_paths_cache_max_entries, RetryConfig retry_config,
    TimeoutConfig timeouts, const std::unordered_set<string>& allowed_locations,
    std::pair<const string, const string>* additional_header,
    bool compose_append)
    : timeouts_(timeouts),
      retry_config_(retry_config),
      auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)),
      zone_provider_(std::move(zone_provider)),
      block_size_(block_size),
      file_block_cache_(
          MakeFileBlockCache(block_size, max_bytes, max_staleness)),
      stat_cache_(new StatCache(stat_cache_max_age, stat_cache_max_entries)),
      matching_paths_cache_(new MatchingPathsCache(
          matching_paths_cache_max_age, matching_paths_cache_max_entries)),
      bucket_location_cache_(new BucketLocationCache(
          kCacheNeverExpire, kBucketLocationCacheMaxEntries)),
      allowed_locations_(allowed_locations),
      compose_append_(compose_append),
      additional_header_(additional_header) {}
absl::Status GcsFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  TF_RETURN_IF_ERROR(CheckBucketLocationConstraint(bucket));
  if (cache_enabled_) {
    result->reset(new GcsRandomAccessFile(fname, [this, bucket, object](
                                                     const string& fname,
                                                     uint64 offset, size_t n,
                                                     absl::string_view* result,
                                                     char* scratch) {
      tf_shared_lock l(block_cache_lock_);
      GcsFileStat stat;
      TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
          fname, &stat,
          [this, bucket, object](const string& fname, GcsFileStat* stat) {
            return UncachedStatForObject(fname, bucket, object, stat);
          }));
      if (!file_block_cache_->ValidateAndUpdateFileSignature(
              fname, stat.generation_number)) {
        VLOG(1)
            << "File signature has been changed. Refreshing the cache. Path: "
            << fname;
      }
      *result = absl::string_view();
      size_t bytes_transferred;
      TF_RETURN_IF_ERROR(file_block_cache_->Read(fname, offset, n, scratch,
                                                 &bytes_transferred));
      *result = absl::string_view(scratch, bytes_transferred);
      if (bytes_transferred < n) {
        return errors::OutOfRange("EOF reached, ", result->size(),
                                  " bytes were read out of ", n,
                                  " bytes requested.");
      }
      return absl::OkStatus();
    }));
  } else {
    result->reset(new BufferedGcsRandomAccessFile(
        fname, block_size_,
        [this, bucket, object](const string& fname, uint64 offset, size_t n,
                               absl::string_view* result, char* scratch) {
          *result = absl::string_view();
          size_t bytes_transferred;
          TF_RETURN_IF_ERROR(
              LoadBufferFromGCS(fname, offset, n, scratch, &bytes_transferred));
          *result = absl::string_view(scratch, bytes_transferred);
          if (bytes_transferred < n) {
            return errors::OutOfRange("EOF reached, ", result->size(),
                                      " bytes were read out of ", n,
                                      " bytes requested.");
          }
          return absl::OkStatus();
        }));
  }
  return absl::OkStatus();
}
void GcsFileSystem::ResetFileBlockCache(size_t block_size_bytes,
                                        size_t max_bytes,
                                        uint64 max_staleness_secs) {
  mutex_lock l(block_cache_lock_);
  file_block_cache_ =
      MakeFileBlockCache(block_size_bytes, max_bytes, max_staleness_secs);
  if (stats_ != nullptr) {
    stats_->Configure(this, &throttle_, file_block_cache_.get());
  }
}
std::unique_ptr<FileBlockCache> GcsFileSystem::MakeFileBlockCache(
    size_t block_size, size_t max_bytes, uint64 max_staleness) {
  std::unique_ptr<FileBlockCache> file_block_cache(new RamFileBlockCache(
      block_size, max_bytes, max_staleness,
      [this](const string& filename, size_t offset, size_t n, char* buffer,
             size_t* bytes_transferred) {
        return LoadBufferFromGCS(filename, offset, n, buffer,
                                 bytes_transferred);
      }));
  cache_enabled_ = file_block_cache->IsCacheEnabled();
  return file_block_cache;
}
absl::Status GcsFileSystem::LoadBufferFromGCS(const string& fname,
                                              size_t offset, size_t n,
                                              char* buffer,
                                              size_t* bytes_transferred) {
  *bytes_transferred = 0;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  profiler::TraceMe activity(
      [fname]() { return absl::StrCat("LoadBufferFromGCS ", fname); });
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  "when reading gs:
  request->SetUri(strings::StrCat("https:
                                  request->EscapeString(object)));
  request->SetRange(offset, offset + n - 1);
  request->SetResultBufferDirect(buffer, n);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.read);
  if (stats_ != nullptr) {
    stats_->RecordBlockLoadRequest(fname, offset);
  }
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading gs:
                                  bucket, "/", object);
  size_t bytes_read = request->GetResultBufferDirectBytesTransferred();
  *bytes_transferred = bytes_read;
  VLOG(1) << "Successful read of gs:
          << offset << " of size: " << bytes_read;
  activity.AppendMetadata([bytes_read]() {
    return profiler::TraceMeEncode({{"block_size", bytes_read}});
  });
  if (stats_ != nullptr) {
    stats_->RecordBlockRetrieved(fname, offset, bytes_read);
  }
  throttle_.RecordResponse(bytes_read);
  if (bytes_read < n) {
    GcsFileStat stat;
    if (stat_cache_->Lookup(fname, &stat)) {
      if (offset + bytes_read < stat.base.length) {
        return errors::Internal(strings::Printf(
            "File contents are inconsistent for file: %s @ %lu.", fname.c_str(),
            offset));
      }
      VLOG(2) << "Successful integrity check for: gs:
              << object << " @ " << offset;
    }
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::CreateNewUploadSession(
    uint64 start_offset, const std::string& object_to_upload,
    const std::string& bucket, uint64 file_size, const std::string& gcs_path,
    UploadSessionHandle* session_handle) {
  std::vector<char> output_buffer;
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  std::string uri = strings::StrCat(
      kGcsUploadUriBase, "b/", bucket,
      "/o?uploadType=resumable&name=", request->EscapeString(object_to_upload));
  request->SetUri(uri);
  request->AddHeader("X-Upload-Content-Length",
                     absl::StrCat(file_size - start_offset));
  request->SetPostEmptyBody();
  request->SetResultBuffer(&output_buffer);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(),
                                  " when initiating an upload to ", gcs_path);
  if (session_handle != nullptr) {
    session_handle->resumable = true;
    session_handle->session_uri = request->GetResponseHeader("Location");
    if (session_handle->session_uri.empty()) {
      return errors::Internal("Unexpected response from GCS when writing to ",
                              gcs_path, ": 'Location' header not returned.");
    }
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::UploadToSession(
    const std::string& session_uri, uint64 start_offset,
    uint64 already_uploaded, const std::string& tmp_content_filename,
    uint64 file_size, const std::string& file_path) {
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(session_uri);
  if (file_size > 0) {
    request->AddHeader("Content-Range",
                       strings::StrCat("bytes ", already_uploaded, "-",
                                       file_size - start_offset - 1, "/",
                                       file_size - start_offset));
  }
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.write);
  TF_RETURN_IF_ERROR(request->SetPutFromFile(tmp_content_filename,
                                             start_offset + already_uploaded));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when uploading ",
                                  file_path);
  return absl::OkStatus();
}
absl::Status GcsFileSystem::RequestUploadSessionStatus(
    const string& session_uri, uint64 file_size, const std::string& gcs_path,
    bool* completed, uint64* uploaded) {
  CHECK(completed != nullptr) << "RequestUploadSessionStatus() called with out "
                                 "param 'completed' == nullptr.";  
  CHECK(uploaded != nullptr) << "RequestUploadSessionStatus() called with out "
                                "param 'uploaded' == nullptr.";  
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(session_uri);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  request->AddHeader("Content-Range", strings::StrCat("bytes */", file_size));
  request->SetPutEmptyBody();
  absl::Status status = request->Send();
  if (status.ok()) {
    *completed = true;
    return absl::OkStatus();
  }
  *completed = false;
  if (request->GetResponseCode() != HTTP_CODE_RESUME_INCOMPLETE) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(status, " when resuming upload ", gcs_path);
  }
  const std::string received_range = request->GetResponseHeader("Range");
  if (received_range.empty()) {
    *uploaded = 0;
  } else {
    absl::string_view range_piece(received_range);
    absl::ConsumePrefix(&range_piece,
                        "bytes=");  
    auto return_error = [](const std::string& gcs_path,
                           const std::string& error_message) {
      return errors::Internal("Unexpected response from GCS when writing ",
                              gcs_path, ": ", error_message);
    };
    std::vector<string> range_strs = str_util::Split(range_piece, '-');
    if (range_strs.size() != 2) {
      return return_error(gcs_path, "Range header '" + received_range +
                                        "' could not be parsed.");
    }
    std::vector<int64_t> range_parts;
    for (const std::string& range_str : range_strs) {
      int64_t tmp;
      if (strings::safe_strto64(range_str, &tmp)) {
        range_parts.push_back(tmp);
      } else {
        return return_error(gcs_path, "Range header '" + received_range +
                                          "' could not be parsed.");
      }
    }
    if (range_parts[0] != 0) {
      return return_error(gcs_path, "The returned range '" + received_range +
                                        "' does not start at zero.");
    }
    *uploaded = range_parts[1] + 1;
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::ParseGcsPathForScheme(absl::string_view fname,
                                                  string scheme,
                                                  bool empty_object_ok,
                                                  string* bucket,
                                                  string* object) {
  absl::string_view parsed_scheme, bucketp, objectp;
  io::ParseURI(fname, &parsed_scheme, &bucketp, &objectp);
  if (parsed_scheme != scheme) {
    return errors::InvalidArgument("GCS path doesn't start with 'gs:
                                   fname);
  }
  *bucket = string(bucketp);
  if (bucket->empty() || *bucket == ".") {
    return errors::InvalidArgument("GCS path doesn't contain a bucket name: ",
                                   fname);
  }
  absl::ConsumePrefix(&objectp, "/");
  *object = string(objectp);
  if (!empty_object_ok && object->empty()) {
    return errors::InvalidArgument("GCS path doesn't contain an object name: ",
                                   fname);
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::ParseGcsPath(absl::string_view fname,
                                         bool empty_object_ok, string* bucket,
                                         string* object) {
  return ParseGcsPathForScheme(fname, "gs", empty_object_ok, bucket, object);
}
void GcsFileSystem::ClearFileCaches(const string& fname) {
  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->RemoveFile(fname);
  stat_cache_->Delete(fname);
}
absl::Status GcsFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  auto session_creator =
      [this](uint64 start_offset, const std::string& object_to_upload,
             const std::string& bucket, uint64 file_size,
             const std::string& gcs_path, UploadSessionHandle* session_handle) {
        return CreateNewUploadSession(start_offset, object_to_upload, bucket,
                                      file_size, gcs_path, session_handle);
      };
  auto object_uploader =
      [this](const std::string& session_uri, uint64 start_offset,
             uint64 already_uploaded, const std::string& tmp_content_filename,
             uint64 file_size, const std::string& file_path) {
        return UploadToSession(session_uri, start_offset, already_uploaded,
                               tmp_content_filename, file_size, file_path);
      };
  auto status_poller = [this](const string& session_uri, uint64 file_size,
                              const std::string& gcs_path, bool* completed,
                              uint64* uploaded) {
    return RequestUploadSessionStatus(session_uri, file_size, gcs_path,
                                      completed, uploaded);
  };
  auto generation_getter = [this](const string& fname, const string& bucket,
                                  const string& object, int64* generation) {
    GcsFileStat stat;
    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&fname, &bucket, &object, &stat, this]() {
          return UncachedStatForObject(fname, bucket, object, &stat);
        },
        retry_config_));
    *generation = stat.generation_number;
    return absl::OkStatus();
  };
  result->reset(new GcsWritableFile(
      bucket, object, this, &timeouts_,
      [this, fname]() { ClearFileCaches(fname); }, retry_config_,
      compose_append_, session_creator, object_uploader, status_poller,
      generation_getter));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<RandomAccessFile> reader;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, token, &reader));
  std::unique_ptr<char[]> buffer(new char[kReadAppendableFileBufferSize]);
  absl::Status status;
  uint64 offset = 0;
  absl::string_view read_chunk;
  string old_content_filename;
  TF_RETURN_IF_ERROR(GetTmpFilename(&old_content_filename));
  std::ofstream old_content(old_content_filename, std::ofstream::binary);
  while (true) {
    status = reader->Read(offset, kReadAppendableFileBufferSize, &read_chunk,
                          buffer.get());
    if (status.ok()) {
      old_content << read_chunk;
      offset += kReadAppendableFileBufferSize;
    } else if (status.code() == absl::StatusCode::kNotFound) {
      break;
    } else if (status.code() == absl::StatusCode::kOutOfRange) {
      old_content << read_chunk;
      break;
    } else {
      return status;
    }
  }
  old_content.close();
  auto session_creator =
      [this](uint64 start_offset, const std::string& object_to_upload,
             const std::string& bucket, uint64 file_size,
             const std::string& gcs_path, UploadSessionHandle* session_handle) {
        return CreateNewUploadSession(start_offset, object_to_upload, bucket,
                                      file_size, gcs_path, session_handle);
      };
  auto object_uploader =
      [this](const std::string& session_uri, uint64 start_offset,
             uint64 already_uploaded, const std::string& tmp_content_filename,
             uint64 file_size, const std::string& file_path) {
        return UploadToSession(session_uri, start_offset, already_uploaded,
                               tmp_content_filename, file_size, file_path);
      };
  auto status_poller = [this](const string& session_uri, uint64 file_size,
                              const std::string& gcs_path, bool* completed,
                              uint64* uploaded) {
    return RequestUploadSessionStatus(session_uri, file_size, gcs_path,
                                      completed, uploaded);
  };
  auto generation_getter = [this](const string& fname, const string& bucket,
                                  const string& object, int64* generation) {
    GcsFileStat stat;
    TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
        [&fname, &bucket, &object, &stat, this]() {
          return UncachedStatForObject(fname, bucket, object, &stat);
        },
        retry_config_));
    *generation = stat.generation_number;
    return absl::OkStatus();
  };
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  result->reset(new GcsWritableFile(
      bucket, object, this, old_content_filename, &timeouts_,
      [this, fname]() { ClearFileCaches(fname); }, retry_config_,
      compose_append_, session_creator, object_uploader, status_poller,
      generation_getter));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  uint64 size;
  TF_RETURN_IF_ERROR(GetFileSize(fname, token, &size));
  std::unique_ptr<char[]> data(new char[size]);
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(NewRandomAccessFile(fname, token, &file));
  absl::string_view piece;
  TF_RETURN_IF_ERROR(file->Read(0, size, &piece, data.get()));
  result->reset(new GcsReadOnlyMemoryRegion(std::move(data), size));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::FileExists(const string& fname,
                                       TransactionToken* token) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool result;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &result));
    if (result) {
      return absl::OkStatus();
    } else {
      return absl::NotFoundError(
          absl::StrCat("The specified bucket ", fname, " was not found."));
    }
  }
  GcsFileStat stat;
  const absl::Status status = StatForObject(fname, bucket, object, &stat);
  if (!absl::IsNotFound(status)) {
    return status;
  }
  bool result;
  TF_RETURN_IF_ERROR(FolderExists(fname, &result));
  if (result) {
    return absl::OkStatus();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}
absl::Status GcsFileSystem::ObjectExists(const string& fname,
                                         const string& bucket,
                                         const string& object, bool* result) {
  GcsFileStat stat;
  const absl::Status status = StatForObject(fname, bucket, object, &stat);
  switch (static_cast<int>(status.code())) {
    case static_cast<int>(error::Code::OK):
      *result = !stat.base.is_directory;
      return absl::OkStatus();
    case static_cast<int>(error::Code::NOT_FOUND):
      *result = false;
      return absl::OkStatus();
    default:
      return status;
  }
}
absl::Status GcsFileSystem::UncachedStatForObject(const string& fname,
                                                  const string& bucket,
                                                  const string& object,
                                                  GcsFileStat* stat) {
  std::vector<char> output_buffer;
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(CreateHttpRequest(&request),
                                  " when reading metadata of gs:
                                  "/", object);
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object),
                                  "?fields=size%2Cgeneration%2Cupdated"));
  request->SetResultBuffer(&output_buffer);
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  if (stats_ != nullptr) {
    stats_->RecordStatObjectRequest();
  }
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      request->Send(), " when reading metadata of gs:
  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
  TF_RETURN_IF_ERROR(GetInt64Value(root, "size", &stat->base.length));
  TF_RETURN_IF_ERROR(
      GetInt64Value(root, "generation", &stat->generation_number));
  string updated;
  TF_RETURN_IF_ERROR(GetStringValue(root, "updated", &updated));
  TF_RETURN_IF_ERROR(ParseRfc3339Time(updated, &(stat->base.mtime_nsec)));
  VLOG(1) << "Stat of: gs:
          << " length: " << stat->base.length
          << " generation: " << stat->generation_number
          << "; mtime_nsec: " << stat->base.mtime_nsec
          << "; updated: " << updated;
  if (absl::EndsWith(fname, "/")) {
    stat->base.is_directory = true;
  } else {
    stat->base.is_directory = false;
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::StatForObject(const string& fname,
                                          const string& bucket,
                                          const string& object,
                                          GcsFileStat* stat) {
  if (object.empty()) {
    return errors::InvalidArgument(strings::Printf(
        "'object' must be a non-empty string. (File: %s)", fname.c_str()));
  }
  TF_RETURN_IF_ERROR(stat_cache_->LookupOrCompute(
      fname, stat,
      [this, &bucket, &object](const string& fname, GcsFileStat* stat) {
        return UncachedStatForObject(fname, bucket, object, stat);
      }));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::BucketExists(const string& bucket, bool* result) {
  const absl::Status status = GetBucketMetadata(bucket, nullptr);
  switch (static_cast<absl::StatusCode>(status.code())) {
    case absl::StatusCode::kOk:
      *result = true;
      return absl::OkStatus();
    case absl::StatusCode::kNotFound:
      *result = false;
      return absl::OkStatus();
    default:
      return status;
  }
}
absl::Status GcsFileSystem::CheckBucketLocationConstraint(
    const string& bucket) {
  if (allowed_locations_.empty()) {
    return absl::OkStatus();
  }
  if (allowed_locations_.erase(kDetectZoneSentinelValue) == 1) {
    string zone;
    TF_RETURN_IF_ERROR(zone_provider_->GetZone(&zone));
    allowed_locations_.insert(ZoneToRegion(&zone));
  }
  string location;
  TF_RETURN_IF_ERROR(GetBucketLocation(bucket, &location));
  if (allowed_locations_.find(location) != allowed_locations_.end()) {
    return absl::OkStatus();
  }
  return errors::FailedPrecondition(strings::Printf(
      "Bucket '%s' is in '%s' location, allowed locations are: (%s).",
      bucket.c_str(), location.c_str(),
      absl::StrJoin(allowed_locations_, ", ").c_str()));
}
absl::Status GcsFileSystem::GetBucketLocation(const string& bucket,
                                              string* location) {
  auto compute_func = [this](const string& bucket, string* location) {
    std::vector<char> result_buffer;
    absl::Status status = GetBucketMetadata(bucket, &result_buffer);
    Json::Value result;
    TF_RETURN_IF_ERROR(ParseJson(result_buffer, &result));
    string bucket_location;
    TF_RETURN_IF_ERROR(
        GetStringValue(result, kBucketMetadataLocationKey, &bucket_location));
    *location = absl::AsciiStrToLower(bucket_location);
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(
      bucket_location_cache_->LookupOrCompute(bucket, location, compute_func));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::GetBucketMetadata(
    const string& bucket, std::vector<char>* result_buffer) {
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket));
  if (result_buffer != nullptr) {
    request->SetResultBuffer(result_buffer);
  }
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  return request->Send();
}
absl::Status GcsFileSystem::FolderExists(const string& dirname, bool* result) {
  StatCache::ComputeFunc compute_func = [this](const string& dirname,
                                               GcsFileStat* stat) {
    std::vector<string> children;
    TF_RETURN_IF_ERROR(
        GetChildrenBounded(dirname, 1, &children, true ,
                           true ));
    if (!children.empty()) {
      stat->base = DIRECTORY_STAT;
      return absl::OkStatus();
    } else {
      return errors::InvalidArgument("Not a directory!");
    }
  };
  GcsFileStat stat;
  absl::Status s = stat_cache_->LookupOrCompute(MaybeAppendSlash(dirname),
                                                &stat, compute_func);
  if (s.ok()) {
    *result = stat.base.is_directory;
    return absl::OkStatus();
  }
  if (absl::IsInvalidArgument(s)) {
    *result = false;
    return absl::OkStatus();
  }
  return s;
}
absl::Status GcsFileSystem::GetChildren(const string& dirname,
                                        TransactionToken* token,
                                        std::vector<string>* result) {
  return GetChildrenBounded(dirname, UINT64_MAX, result,
                            false ,
                            false );
}
absl::Status GcsFileSystem::GetMatchingPaths(const string& pattern,
                                             TransactionToken* token,
                                             std::vector<string>* results) {
  MatchingPathsCache::ComputeFunc compute_func =
      [this](const string& pattern, std::vector<string>* results) {
        results->clear();
        const string& fixed_prefix =
            pattern.substr(0, pattern.find_first_of("*?[\\"));
        const string dir(this->Dirname(fixed_prefix));
        if (dir.empty()) {
          return errors::InvalidArgument(
              "A GCS pattern doesn't have a bucket name: ", pattern);
        }
        std::vector<string> all_files;
        TF_RETURN_IF_ERROR(GetChildrenBounded(
            dir, UINT64_MAX, &all_files, true ,
            false ));
        const auto& files_and_folders = AddAllSubpaths(all_files);
        const absl::string_view dir_no_slash = absl::StripSuffix(dir, "/");
        for (const auto& path : files_and_folders) {
          const string full_path = strings::StrCat(dir_no_slash, "/", path);
          if (this->Match(full_path, pattern)) {
            results->push_back(full_path);
          }
        }
        return absl::OkStatus();
      };
  TF_RETURN_IF_ERROR(
      matching_paths_cache_->LookupOrCompute(pattern, results, compute_func));
  return absl::OkStatus();
}
absl::Status GcsFileSystem::GetChildrenBounded(
    const string& dirname, uint64 max_results, std::vector<string>* result,
    bool recursive, bool include_self_directory_marker) {
  if (!result) {
    return errors::InvalidArgument("'result' cannot be null");
  }
  string bucket, object_prefix;
  TF_RETURN_IF_ERROR(
      ParseGcsPath(MaybeAppendSlash(dirname), true, &bucket, &object_prefix));
  string nextPageToken;
  uint64 retrieved_results = 0;
  while (true) {  
    std::vector<char> output_buffer;
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
    auto uri = strings::StrCat(kGcsUriBase, "b/", bucket, "/o");
    if (recursive) {
      uri = strings::StrCat(uri, "?fields=items%2Fname%2CnextPageToken");
    } else {
      uri = strings::StrCat(uri,
                            "?fields=items%2Fname%2Cprefixes%2CnextPageToken");
      uri = strings::StrCat(uri, "&delimiter=%2F");
    }
    if (!object_prefix.empty()) {
      uri = strings::StrCat(uri,
                            "&prefix=", request->EscapeString(object_prefix));
    }
    if (!nextPageToken.empty()) {
      uri = strings::StrCat(
          uri, "&pageToken=", request->EscapeString(nextPageToken));
    }
    if (max_results - retrieved_results < kGetChildrenDefaultPageSize) {
      uri =
          strings::StrCat(uri, "&maxResults=", max_results - retrieved_results);
    }
    request->SetUri(uri);
    request->SetResultBuffer(&output_buffer);
    request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading ", dirname);
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
    const auto items = root.get("items", Json::Value::null);
    if (!items.isNull()) {
      if (!items.isArray()) {
        return errors::Internal(
            "Expected an array 'items' in the GCS response.");
      }
      for (size_t i = 0; i < items.size(); i++) {
        const auto item = items.get(i, Json::Value::null);
        if (!item.isObject()) {
          return errors::Internal(
              "Unexpected JSON format: 'items' should be a list of objects.");
        }
        string name;
        TF_RETURN_IF_ERROR(GetStringValue(item, "name", &name));
        absl::string_view relative_path(name);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(strings::StrCat(
              "Unexpected response: the returned file name ", name,
              " doesn't match the prefix ", object_prefix));
        }
        if (!relative_path.empty() || include_self_directory_marker) {
          result->emplace_back(relative_path);
        }
        if (++retrieved_results >= max_results) {
          return absl::OkStatus();
        }
      }
    }
    const auto prefixes = root.get("prefixes", Json::Value::null);
    if (!prefixes.isNull()) {
      if (!prefixes.isArray()) {
        return errors::Internal(
            "'prefixes' was expected to be an array in the GCS response.");
      }
      for (size_t i = 0; i < prefixes.size(); i++) {
        const auto prefix = prefixes.get(i, Json::Value::null);
        if (prefix.isNull() || !prefix.isString()) {
          return errors::Internal(
              "'prefixes' was expected to be an array of strings in the GCS "
              "response.");
        }
        const string& prefix_str = prefix.asString();
        absl::string_view relative_path(prefix_str);
        if (!absl::ConsumePrefix(&relative_path, object_prefix)) {
          return errors::Internal(
              "Unexpected response: the returned folder name ", prefix_str,
              " doesn't match the prefix ", object_prefix);
        }
        result->emplace_back(relative_path);
        if (++retrieved_results >= max_results) {
          return absl::OkStatus();
        }
      }
    }
    const auto token = root.get("nextPageToken", Json::Value::null);
    if (token.isNull()) {
      return absl::OkStatus();
    }
    if (!token.isString()) {
      return errors::Internal(
          "Unexpected response: nextPageToken is not a string");
    }
    nextPageToken = token.asString();
  }
}
absl::Status GcsFileSystem::Stat(const string& fname, TransactionToken* token,
                                 FileStatistics* stat) {
  if (!stat) {
    return errors::Internal("'stat' cannot be nullptr.");
  }
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      *stat = DIRECTORY_STAT;
      return absl::OkStatus();
    }
    return errors::NotFound("The specified bucket ", fname, " was not found.");
  }
  GcsFileStat gcs_stat;
  const absl::Status status = StatForObject(fname, bucket, object, &gcs_stat);
  if (status.ok()) {
    *stat = gcs_stat.base;
    return absl::OkStatus();
  }
  if (!absl::IsNotFound(status)) {
    return status;
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    *stat = DIRECTORY_STAT;
    return absl::OkStatus();
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}
absl::Status GcsFileSystem::DeleteFile(const string& fname,
                                       TransactionToken* token) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", bucket, "/o/",
                                  request->EscapeString(object)));
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  request->SetDeleteRequest();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when deleting ", fname);
  ClearFileCaches(fname);
  return absl::OkStatus();
}
absl::Status GcsFileSystem::CreateDir(const string& dirname,
                                      TransactionToken* token) {
  string dirname_with_slash = MaybeAppendSlash(dirname);
  VLOG(3) << "CreateDir: creating directory with dirname: " << dirname
          << " and dirname_with_slash: " << dirname_with_slash;
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(dirname_with_slash, true,
                                  &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    return is_bucket ? absl::OkStatus()
                     : errors::NotFound("The specified bucket ",
                                        dirname_with_slash, " was not found.");
  }
  if (FileExists(dirname_with_slash, token).ok()) {
    VLOG(3) << "CreateDir: directory already exists, not uploading " << dirname;
    return errors::AlreadyExists(dirname);
  }
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(
      kGcsUploadUriBase, "b/", bucket,
      "/o?uploadType=media&name=", request->EscapeString(object),
      "&ifGenerationMatch=0"));
  request->SetPostEmptyBody();
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  const absl::Status& status = request->Send();
  if (status.ok()) {
    VLOG(3) << "CreateDir: finished uploading directory " << dirname;
    return absl::OkStatus();
  }
  if (request->GetResponseCode() != HTTP_CODE_PRECONDITION_FAILED) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(status, " when uploading ",
                                    dirname_with_slash);
  }
  VLOG(3) << "Ignoring directory already exists on object "
          << dirname_with_slash;
  return errors::AlreadyExists(dirname);
}
absl::Status GcsFileSystem::DeleteDir(const string& dirname,
                                      TransactionToken* token) {
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(dirname, 2, &children, true ,
                         true ));
  if (children.size() > 1 || (children.size() == 1 && !children[0].empty())) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }
  if (children.size() == 1 && children[0].empty()) {
    return DeleteFile(MaybeAppendSlash(dirname), token);
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::GetFileSize(const string& fname,
                                        TransactionToken* token,
                                        uint64* file_size) {
  if (!file_size) {
    return errors::Internal("'file_size' cannot be nullptr.");
  }
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, false, &bucket, &object));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, token, &stat));
  *file_size = stat.length;
  return absl::OkStatus();
}
absl::Status GcsFileSystem::RenameFile(const string& src, const string& target,
                                       TransactionToken* token) {
  if (!IsDirectory(src, token).ok()) {
    return RenameObject(src, target);
  }
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      GetChildrenBounded(src, UINT64_MAX, &children, true ,
                         true ));
  for (const string& subpath : children) {
    TF_RETURN_IF_ERROR(
        RenameObject(JoinGcsPath(src, subpath), JoinGcsPath(target, subpath)));
  }
  return absl::OkStatus();
}
absl::Status GcsFileSystem::RenameObject(const string& src,
                                         const string& target) {
  VLOG(3) << "RenameObject: started gs:
  string src_bucket, src_object, target_bucket, target_object;
  TF_RETURN_IF_ERROR(ParseGcsPath(src, false, &src_bucket, &src_object));
  TF_RETURN_IF_ERROR(
      ParseGcsPath(target, false, &target_bucket, &target_object));
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateHttpRequest(&request));
  request->SetUri(strings::StrCat(kGcsUriBase, "b/", src_bucket, "/o/",
                                  request->EscapeString(src_object),
                                  "/rewriteTo/b/", target_bucket, "/o/",
                                  request->EscapeString(target_object)));
  request->SetPostEmptyBody();
  request->SetTimeouts(timeouts_.connect, timeouts_.idle, timeouts_.metadata);
  std::vector<char> output_buffer;
  request->SetResultBuffer(&output_buffer);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when renaming ", src,
                                  " to ", target);
  ClearFileCaches(target);
  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(output_buffer, &root));
  bool done;
  TF_RETURN_IF_ERROR(GetBoolValue(root, "done", &done));
  if (!done) {
    return errors::Unimplemented(
        "Couldn't rename ", src, " to ", target,
        ": moving large files between buckets with different "
        "locations or storage classes is not supported.");
  }
  VLOG(3) << "RenameObject: finished from: gs:
  return RetryingUtils::DeleteWithRetries(
      [this, &src]() { return DeleteFile(src, nullptr); }, retry_config_);
}
absl::Status GcsFileSystem::IsDirectory(const string& fname,
                                        TransactionToken* token) {
  string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, true, &bucket, &object));
  if (object.empty()) {
    bool is_bucket;
    TF_RETURN_IF_ERROR(BucketExists(bucket, &is_bucket));
    if (is_bucket) {
      return absl::OkStatus();
    }
    return errors::NotFound("The specified bucket gs:
                            " was not found.");
  }
  bool is_folder;
  TF_RETURN_IF_ERROR(FolderExists(fname, &is_folder));
  if (is_folder) {
    return absl::OkStatus();
  }
  bool is_object;
  TF_RETURN_IF_ERROR(ObjectExists(fname, bucket, object, &is_object));
  if (is_object) {
    return errors::FailedPrecondition("The specified path ", fname,
                                      " is not a directory.");
  }
  return errors::NotFound("The specified path ", fname, " was not found.");
}
absl::Status GcsFileSystem::DeleteRecursively(const string& dirname,
                                              TransactionToken* token,
                                              int64_t* undeleted_files,
                                              int64_t* undeleted_dirs) {
  if (!undeleted_files || !undeleted_dirs) {
    return errors::Internal(
        "'undeleted_files' and 'undeleted_dirs' cannot be nullptr.");
  }
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  if (!IsDirectory(dirname, token).ok()) {
    *undeleted_dirs = 1;
    return absl::Status(
        absl::StatusCode::kNotFound,
        strings::StrCat(dirname, " doesn't exist or not a directory."));
  }
  std::vector<string> all_objects;
  TF_RETURN_IF_ERROR(GetChildrenBounded(
      dirname, UINT64_MAX, &all_objects, true ,
      true ));
  for (const string& object : all_objects) {
    const string& full_path = JoinGcsPath(dirname, object);
    const auto& delete_file_status = RetryingUtils::DeleteWithRetries(
        [this, &full_path, token]() { return DeleteFile(full_path, token); },
        retry_config_);
    if (!delete_file_status.ok()) {
      if (IsDirectory(full_path, token).ok()) {
        (*undeleted_dirs)++;
      } else {
        (*undeleted_files)++;
      }
    }
  }
  return absl::OkStatus();
}
void GcsFileSystem::FlushCaches(TransactionToken* token) {
  tf_shared_lock l(block_cache_lock_);
  file_block_cache_->Flush();
  stat_cache_->Clear();
  matching_paths_cache_->Clear();
  bucket_location_cache_->Clear();
}
void GcsFileSystem::SetStats(GcsStatsInterface* stats) {
  CHECK(stats_ == nullptr) << "SetStats() has already been called.";
  CHECK(stats != nullptr);
  mutex_lock l(block_cache_lock_);
  stats_ = stats;
  stats_->Configure(this, &throttle_, file_block_cache_.get());
}
void GcsFileSystem::SetCacheStats(FileBlockCacheStatsInterface* cache_stats) {
  tf_shared_lock l(block_cache_lock_);
  if (file_block_cache_ == nullptr) {
    LOG(ERROR) << "Tried to set cache stats of non-initialized file block "
                  "cache object. This may result in not exporting the intended "
                  "monitoring data";
    return;
  }
  file_block_cache_->SetStats(cache_stats);
}
void GcsFileSystem::SetAuthProvider(
    std::unique_ptr<AuthProvider> auth_provider) {
  mutex_lock l(mu_);
  auth_provider_ = std::move(auth_provider);
}
absl::Status GcsFileSystem::CreateHttpRequest(
    std::unique_ptr<HttpRequest>* request) {
  std::unique_ptr<HttpRequest> new_request{http_request_factory_->Create()};
  if (dns_cache_) {
    dns_cache_->AnnotateRequest(new_request.get());
  }
  string auth_token;
  {
    tf_shared_lock l(mu_);
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));
  }
  new_request->AddAuthBearerHeader(auth_token);
  if (additional_header_) {
    new_request->AddHeader(additional_header_->first,
                           additional_header_->second);
  }
  if (stats_ != nullptr) {
    new_request->SetRequestStats(stats_->HttpStats());
  }
  if (!throttle_.AdmitRequest()) {
    return errors::Unavailable("Request throttled");
  }
  *request = std::move(new_request);
  return absl::OkStatus();
}
RetryingGcsFileSystem::RetryingGcsFileSystem()
    : RetryingFileSystem(std::make_unique<GcsFileSystem>(),
                         RetryConfig(GetGcsRetryConfig())) {}
}  
REGISTER_LEGACY_FILE_SYSTEM("gs", ::tsl::RetryingGcsFileSystem);