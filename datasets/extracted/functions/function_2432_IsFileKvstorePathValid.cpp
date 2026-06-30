#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>  
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"  
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/file_io_concurrency_resource.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/unique_handle.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/file/file_resource.h"
#include "tensorstore/kvstore/file/util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/internal/os/file_lister.h"
#include "tensorstore/internal/os/file_util.h"
using ::tensorstore::internal::OsErrorCode;
using ::tensorstore::internal_file_util::IsKeyValid;
using ::tensorstore::internal_file_util::LongestDirectoryPrefix;
using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::FileInfo;
using ::tensorstore::internal_os::kLockSuffix;
using ::tensorstore::internal_os::UniqueFileDescriptor;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::ReadResult;
using ::tensorstore::kvstore::SupportedFeatures;
namespace tensorstore {
namespace internal_file_kvstore {
namespace {
namespace jb = tensorstore::internal_json_binding;
struct FileMetrics : public internal_kvstore::CommonMetrics {
  internal_metrics::Counter<int64_t>& open_read;
  internal_metrics::Counter<int64_t>& lock_contention;
};
auto file_metrics = []() -> FileMetrics {
  return {TENSORSTORE_KVSTORE_COMMON_METRICS(file),
          TENSORSTORE_KVSTORE_COUNTER_IMPL(
              file, open_read, "Number of times a file is opened for reading"),
          TENSORSTORE_KVSTORE_COUNTER_IMPL(file, lock_contention,
                                           " kvstore::Write lock contention")};
}();
ABSL_CONST_INIT internal_log::VerboseFlag file_logging("file");
bool IsFileKvstorePathValid(std::string_view path) {
  if (path.empty() || path == "/") return true;
  if (path.back() == '/' || path.back() == '\\') {
    path.remove_suffix(1);
  }
  return IsKeyValid(path, kLockSuffix);
}
struct FileKeyValueStoreSpecData {
  Context::Resource<internal::FileIoConcurrencyResource> file_io_concurrency;
  Context::Resource<FileIoSyncResource> file_io_sync;
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.file_io_concurrency, x.file_io_sync);
  };
  constexpr static auto default_json_binder = jb::Object(
      jb::Member(
          internal::FileIoConcurrencyResource::id,
          jb::Projection<&FileKeyValueStoreSpecData::file_io_concurrency>()),
      jb::Member(FileIoSyncResource::id,
                 jb::Projection<&FileKeyValueStoreSpecData::file_io_sync>())
  );
};
class FileKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<FileKeyValueStoreSpec,
                                                    FileKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "file";
  absl::Status NormalizeSpec(std::string& path) override {
    if (!IsFileKvstorePathValid(path)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid file path: ", QuoteString(path)));
    }
    path = internal::LexicalNormalizePath(path);
    return absl::OkStatus();
  }
  Future<kvstore::DriverPtr> DoOpen() const override;
  Result<std::string> ToUrl(std::string_view path) const override {
    return absl::StrCat(id, ":
  }
};
class FileKeyValueStore
    : public internal_kvstore::RegisteredDriver<FileKeyValueStore,
                                                FileKeyValueStoreSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;
  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;
  Future<const void> DeleteRange(KeyRange range) override;
  void ListImpl(ListOptions options, ListReceiver receiver) override;
  const Executor& executor() { return spec_.file_io_concurrency->executor; }
  std::string DescribeKey(std::string_view key) override {
    return absl::StrCat("local file ", QuoteString(key));
  }
  absl::Status GetBoundSpecData(FileKeyValueStoreSpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }
  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
           SupportedFeatures::kAtomicWriteWithoutOverwrite;
  }
  bool sync() const { return *spec_.file_io_sync; }
  SpecData spec_;
};
absl::Status ValidateKey(std::string_view key) {
  if (!IsKeyValid(key, kLockSuffix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid key: ", QuoteString(key)));
  }
  return absl::OkStatus();
}
absl::Status ValidateKeyRange(const KeyRange& range) {
  auto prefix = LongestDirectoryPrefix(range);
  if (prefix.empty()) return absl::OkStatus();
  return ValidateKey(prefix);
}
StorageGeneration GetFileGeneration(const FileInfo& info) {
  return StorageGeneration::FromValues(
      internal_os::GetDeviceId(info), internal_os::GetFileId(info),
      absl::ToUnixNanos(internal_os::GetMTime(info)));
}
Result<UniqueFileDescriptor> OpenParentDirectory(std::string path) {
  size_t end_pos = path.size();
  Result<UniqueFileDescriptor> fd;
  while (true) {
    size_t separator_pos = end_pos;
    while (separator_pos != 0 &&
           !internal_os::IsDirSeparator(path[separator_pos - 1])) {
      --separator_pos;
    }
    --separator_pos;
    const char* dir_path;
    if (separator_pos == std::string::npos) {
      dir_path = ".";
    } else if (separator_pos == 0) {
      dir_path = "/";
    } else {
      path[separator_pos] = '\0';
      dir_path = path.c_str();
      end_pos = separator_pos;
    }
    fd = internal_os::OpenDirectoryDescriptor(dir_path);
    if (!fd.ok()) {
      if (absl::IsNotFound(fd.status())) {
        assert(separator_pos != 0 && separator_pos != std::string::npos);
        end_pos = separator_pos - 1;
        continue;
      }
      return fd.status();
    }
    if (dir_path == path.c_str()) path[separator_pos] = '/';
    break;
  }
  while (true) {
    size_t separator_pos = path.find('\0', end_pos);
    if (separator_pos == std::string::npos) {
      return fd;
    }
    TENSORSTORE_RETURN_IF_ERROR(internal_os::MakeDirectory(path));
    fd = internal_os::OpenDirectoryDescriptor(path);
    TENSORSTORE_RETURN_IF_ERROR(fd.status());
    path[separator_pos] = '/';
    end_pos = separator_pos + 1;
  }
}
Result<UniqueFileDescriptor> OpenValueFile(const std::string& path,
                                           StorageGeneration* generation,
                                           int64_t* size = nullptr) {
  auto fd = internal_os::OpenExistingFileForReading(path);
  if (!fd.ok()) {
    if (absl::IsNotFound(fd.status())) {
      *generation = StorageGeneration::NoValue();
      return UniqueFileDescriptor{};
    }
    return fd;
  }
  FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd->get(), &info));
  if (!internal_os::IsRegularFile(info)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Not a regular file: ", QuoteString(path)));
  }
  if (size) *size = internal_os::GetSize(info);
  *generation = GetFileGeneration(info);
  return fd;
}
Result<absl::Cord> ReadFromFileDescriptor(FileDescriptor fd,
                                          ByteRange byte_range) {
  file_metrics.batch_read.Increment();
  absl::Time start_time = absl::Now();
  internal::FlatCordBuilder buffer(byte_range.size(), false);
  size_t offset = 0;
  while (offset < buffer.size()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto n, internal_os::ReadFromFile(fd, buffer.data() + offset,
                                          buffer.size() - offset,
                                          byte_range.inclusive_min + offset));
    if (n > 0) {
      file_metrics.bytes_read.IncrementBy(n);
      offset += n;
      buffer.set_inuse(offset);
      continue;
    }
    if (n == 0) {
      return absl::UnavailableError("Length changed while reading");
    }
  }
  file_metrics.read_latency_ms.Observe(
      absl::ToInt64Milliseconds(absl::Now() - start_time));
  return std::move(buffer).Build();
}
class BatchReadTask;
using BatchReadTaskBase = internal_kvstore_batch::BatchReadEntry<
    FileKeyValueStore,
    internal_kvstore_batch::ReadRequest<kvstore::ReadGenerationConditions>,
    std::string >;
class BatchReadTask final
    : public BatchReadTaskBase,
      public internal::AtomicReferenceCount<BatchReadTask> {
 private:
  TimestampedStorageGeneration stamp_;
  UniqueFileDescriptor fd_;
  int64_t size_;
 public:
  BatchReadTask(BatchEntryKey&& batch_entry_key_)
      : BatchReadTaskBase(std::move(batch_entry_key_)),
        internal::AtomicReferenceCount<BatchReadTask>(1) {
  }
  void Submit(Batch::View batch) final {
    if (request_batch.requests.empty()) return;
    driver().executor()(
        [self = internal::IntrusivePtr<BatchReadTask>(
             this, internal::adopt_object_ref)] { self->ProcessBatch(); });
  }
  Result<kvstore::ReadResult> DoByteRangeRead(ByteRange byte_range) {
    absl::Cord value;
    TENSORSTORE_ASSIGN_OR_RETURN(
        value, ReadFromFileDescriptor(fd_.get(), byte_range),
        tensorstore::MaybeAnnotateStatus(_, "Error reading from open file"));
    return kvstore::ReadResult::Value(std::move(value), stamp_);
  }
  void ProcessBatch() {
    ABSL_LOG_IF(INFO, file_logging)
        << "BatchReadTask " << std::get<std::string>(batch_entry_key);
    stamp_.time = absl::Now();
    file_metrics.open_read.Increment();
    auto& requests = request_batch.requests;
    TENSORSTORE_ASSIGN_OR_RETURN(
        fd_,
        OpenValueFile(std::get<std::string>(batch_entry_key),
                      &stamp_.generation, &size_),
        internal_kvstore_batch::SetCommonResult(requests, std::move(_)));
    if (!fd_.valid()) {
      internal_kvstore_batch::SetCommonResult(
          requests, kvstore::ReadResult::Missing(stamp_.time));
      return;
    }
    internal_kvstore_batch::ValidateGenerationsAndByteRanges(requests, stamp_,
                                                             size_);
    if (requests.empty()) return;
    if (requests.size() == 1) {
      auto& byte_range_request =
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(requests[0]);
      byte_range_request.promise.SetResult(
          DoByteRangeRead(byte_range_request.byte_range.AsByteRange()));
      return;
    }
    const auto& executor = driver().executor();
    internal_kvstore_batch::CoalescingOptions coalescing_options;
    coalescing_options.max_extra_read_bytes = 255;
    internal_kvstore_batch::ForEachCoalescedRequest<Request>(
        requests, coalescing_options,
        [&](ByteRange coalesced_byte_range,
            tensorstore::span<Request> coalesced_requests) {
          auto self = internal::IntrusivePtr<BatchReadTask>(this);
          executor([self = std::move(self), coalesced_byte_range,
                    coalesced_requests] {
            self->ProcessCoalescedRead(coalesced_byte_range,
                                       coalesced_requests);
          });
        });
  }
  void ProcessCoalescedRead(ByteRange coalesced_byte_range,
                            tensorstore::span<Request> coalesced_requests) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 DoByteRangeRead(coalesced_byte_range),
                                 internal_kvstore_batch::SetCommonResult(
                                     coalesced_requests, std::move(_)));
    internal_kvstore_batch::ResolveCoalescedRequests(
        coalesced_byte_range, coalesced_requests, std::move(read_result));
  }
};
Future<ReadResult> FileKeyValueStore::Read(Key key, ReadOptions options) {
  file_metrics.read.Increment();
  TENSORSTORE_RETURN_IF_ERROR(ValidateKey(key));
  auto [promise, future] = PromiseFuturePair<kvstore::ReadResult>::Make();
  BatchReadTask::MakeRequest<BatchReadTask>(
      *this, {std::move(key)}, options.batch, options.staleness_bound,
      BatchReadTask::Request{{std::move(promise), options.byte_range},
                             std::move(options.generation_conditions)});
  return std::move(future);
}
Result<StorageGeneration> WriteWithSyncAndRename(
    FileDescriptor fd, const std::string& fd_path, absl::Cord value, bool sync,
    const std::string& rename_path) {
  auto start_write = absl::Now();
  while (!value.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto n, internal_os::WriteCordToFile(fd, value),
        MaybeAnnotateStatus(
            _, absl::StrCat("Failed writing: ", QuoteString(fd_path))));
    file_metrics.bytes_written.IncrementBy(n);
    if (n == value.size()) break;
    value.RemovePrefix(n);
  }
  if (sync) {
    TENSORSTORE_RETURN_IF_ERROR(internal_os::FsyncFile(fd));
  }
  FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd, &info));
  TENSORSTORE_RETURN_IF_ERROR(
      internal_os::RenameOpenFile(fd, fd_path, rename_path));
#if 0  
  FileInfo debug_info;
  ABSL_CHECK_OK(internal_os::GetFileInfo(fd, &debug_info));
  ABSL_CHECK_EQ(GetFileGeneration(info), GetFileGeneration(debug_info));
#endif
  file_metrics.write_latency_ms.Observe(
      absl::ToInt64Milliseconds(absl::Now() - start_write));
  return GetFileGeneration(info);
}
Result<UniqueFileDescriptor> OpenLockFile(const std::string& path,
                                          FileInfo* info) {
  auto fd = internal_os::OpenFileForWriting(path);
  if (!fd.ok()) return fd;
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd->get(), info));
  if (!internal_os::IsRegularFile(*info)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Not a regular file: ", path));
  }
  return fd;
}
struct WriteLockHelper {
  std::string lock_path;  
  UniqueFileDescriptor lock_fd;
  std::optional<internal_os::UnlockFn> unlock_fn;
  WriteLockHelper(const std::string& path)
      : lock_path(absl::StrCat(path, kLockSuffix)) {}
  ~WriteLockHelper() { Unlock(); }
  absl::Status CreateAndAcquire() {
    FileInfo a, b;
    FileInfo* info = &a;
    TENSORSTORE_ASSIGN_OR_RETURN(lock_fd, OpenLockFile(lock_path, info));
    while (true) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          unlock_fn, internal_os::AcquireFdLock(lock_fd.get()),
          MaybeAnnotateStatus(_,
                              absl::StrCat("Failed to acquire lock on file: ",
                                           QuoteString(lock_path))));
      FileInfo* other_info = info == &a ? &b : &a;
      TENSORSTORE_ASSIGN_OR_RETURN(UniqueFileDescriptor other_fd,
                                   OpenLockFile(lock_path, other_info));
      if (internal_os::GetDeviceId(a) == internal_os::GetDeviceId(b) &&
          internal_os::GetFileId(a) == internal_os::GetFileId(b)) {
        return absl::OkStatus();
      }
      Unlock();
      info = other_info;
      lock_fd = std::move(other_fd);
      file_metrics.lock_contention.Increment();
    }
  }
  absl::Status Delete() {
    auto status = internal_os::DeleteOpenFile(lock_fd.get(), lock_path);
    if (status.ok() || absl::IsNotFound(status)) {
      return absl::OkStatus();
    }
    return MaybeAnnotateStatus(std::move(status), "Failed to clean lock file");
  }
  void Unlock() {
    if (unlock_fn) {
      std::move (*unlock_fn)(lock_fd.get());
      unlock_fn = std::nullopt;
    }
  }
};
struct WriteTask {
  std::string full_path;
  absl::Cord value;
  kvstore::WriteOptions options;
  bool sync;
  Result<TimestampedStorageGeneration> operator()() const {
    ABSL_LOG_IF(INFO, file_logging) << "WriteTask " << full_path;
    TimestampedStorageGeneration r;
    r.time = absl::Now();
    TENSORSTORE_ASSIGN_OR_RETURN(auto dir_fd, OpenParentDirectory(full_path));
    WriteLockHelper lock_helper(full_path);
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.CreateAndAcquire());
    bool delete_lock_file = true;
    FileDescriptor fd = lock_helper.lock_fd.get();
    const std::string& lock_path = lock_helper.lock_path;
    auto generation_result = [&]() -> Result<StorageGeneration> {
      if (!StorageGeneration::IsUnknown(
              options.generation_conditions.if_equal)) {
        StorageGeneration generation;
        TENSORSTORE_ASSIGN_OR_RETURN(UniqueFileDescriptor value_fd,
                                     OpenValueFile(full_path, &generation));
        if (generation != options.generation_conditions.if_equal) {
          return StorageGeneration::Unknown();
        }
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto generation,
          WriteWithSyncAndRename(fd, lock_path, value, sync, full_path));
      delete_lock_file = false;
      if (sync) {
        TENSORSTORE_RETURN_IF_ERROR(
            internal_os::FsyncDirectory(dir_fd.get()),
            MaybeAnnotateStatus(
                _, absl::StrCat("Error calling fsync on parent directory of: ",
                                full_path)));
      }
      lock_helper.Unlock();
      return generation;
    }();
    if (delete_lock_file) {
      lock_helper.Delete().IgnoreError();
    }
    if (!generation_result) {
      return std::move(generation_result).status();
    }
    r.generation = *std::move(generation_result);
    return r;
  }
};
struct DeleteTask {
  std::string full_path;
  kvstore::WriteOptions options;
  bool sync;
  Result<TimestampedStorageGeneration> operator()() const {
    ABSL_LOG_IF(INFO, file_logging) << "DeleteTask " << full_path;
    TimestampedStorageGeneration r;
    r.time = absl::Now();
    WriteLockHelper lock_helper(full_path);
    TENSORSTORE_ASSIGN_OR_RETURN(auto dir_fd, OpenParentDirectory(full_path));
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.CreateAndAcquire());
    bool fsync_directory = false;
    auto generation_result = [&]() -> Result<StorageGeneration> {
      if (!StorageGeneration::IsUnknown(
              options.generation_conditions.if_equal)) {
        StorageGeneration generation;
        TENSORSTORE_ASSIGN_OR_RETURN(UniqueFileDescriptor value_fd,
                                     OpenValueFile(full_path, &generation));
        if (generation != options.generation_conditions.if_equal) {
          return StorageGeneration::Unknown();
        }
      }
      auto status = internal_os::DeleteFile(full_path);
      if (!status.ok() && !absl::IsNotFound(status)) {
        return status;
      }
      fsync_directory = sync;
      return StorageGeneration::NoValue();
    }();
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.Delete());
    if (fsync_directory) {
      TENSORSTORE_RETURN_IF_ERROR(
          internal_os::FsyncDirectory(dir_fd.get()),
          MaybeAnnotateStatus(
              _, absl::StrCat("Error calling fsync on parent directory of: ",
                              QuoteString(full_path))));
    }
    if (!generation_result) {
      return std::move(generation_result).status();
    }
    r.generation = *std::move(generation_result);
    return r;
  }
};
Future<TimestampedStorageGeneration> FileKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  file_metrics.write.Increment();
  TENSORSTORE_RETURN_IF_ERROR(ValidateKey(key));
  if (value) {
    return MapFuture(executor(), WriteTask{std::move(key), *std::move(value),
                                           std::move(options), this->sync()});
  } else {
    return MapFuture(executor(), DeleteTask{std::move(key), std::move(options),
                                            this->sync()});
  }
}
struct DeleteRangeTask {
  KeyRange range;
  void operator()(Promise<void> promise) {
    ABSL_LOG_IF(INFO, file_logging) << "DeleteRangeTask " << range;
    std::string prefix(internal_file_util::LongestDirectoryPrefix(range));
    absl::Status delete_status;
    auto status = internal_os::RecursiveFileList(
        prefix,
        [&](std::string_view path) {
          return tensorstore::IntersectsPrefix(range, path);
        },
        [&](auto entry) -> absl::Status {
          if (!promise.result_needed()) return absl::CancelledError("");
          bool do_delete = false;
          if (entry.IsDirectory()) {
            do_delete = tensorstore::ContainsPrefix(range, entry.GetFullPath());
          } else {
            do_delete = tensorstore::Contains(range, entry.GetFullPath());
          }
          if (do_delete) {
            auto s = entry.Delete();
            if (!s.ok() && !absl::IsNotFound(s) &&  
                !absl::IsFailedPrecondition(s)) {   
              ABSL_LOG_IF(INFO, file_logging) << s;
              delete_status.Update(s);
            }
          }
          return absl::OkStatus();
        });
    if (!status.ok()) {
      promise.SetResult(MakeResult(std::move(status)));
    }
    promise.SetResult(MakeResult(std::move(delete_status)));
  }
};
Future<const void> FileKeyValueStore::DeleteRange(KeyRange range) {
  file_metrics.delete_range.Increment();
  if (range.empty()) return absl::OkStatus();  
  TENSORSTORE_RETURN_IF_ERROR(ValidateKeyRange(range));
  return PromiseFuturePair<void>::Link(
             WithExecutor(executor(), DeleteRangeTask{std::move(range)}))
      .future;
}
struct ListTask {
  kvstore::ListOptions options;
  ListReceiver receiver;
  void operator()() {
    ABSL_LOG_IF(INFO, file_logging) << "ListTask " << options.range;
    std::atomic<bool> cancelled = false;
    execution::set_starting(receiver, [&cancelled] {
      cancelled.store(true, std::memory_order_relaxed);
    });
    std::string prefix(
        internal_file_util::LongestDirectoryPrefix(options.range));
    auto status = internal_os::RecursiveFileList(
        prefix,
        [&](std::string_view path) {
          return tensorstore::IntersectsPrefix(options.range, path);
        },
        [&](auto entry) -> absl::Status {
          if (cancelled.load(std::memory_order_relaxed)) {
            return absl::CancelledError("");
          }
          if (entry.IsDirectory()) return absl::OkStatus();
          std::string_view path = entry.GetFullPath();
          if (tensorstore::Contains(options.range, path) &&
              !absl::EndsWith(path, kLockSuffix)) {
            path.remove_prefix(options.strip_prefix_length);
            execution::set_value(receiver,
                                 ListEntry{std::string(path), entry.GetSize()});
          }
          return absl::OkStatus();
        });
    if (!status.ok() && !cancelled.load(std::memory_order_relaxed)) {
      execution::set_error(receiver, std::move(status));
      execution::set_stopping(receiver);
      return;
    }
    execution::set_done(receiver);
    execution::set_stopping(receiver);
  }
};
void FileKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  file_metrics.list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }
  if (auto error = ValidateKeyRange(options.range); !error.ok()) {
    execution::set_starting(receiver, [] {});
    execution::set_error(receiver, std::move(error));
    execution::set_stopping(receiver);
    return;
  }
  executor()(ListTask{std::move(options), std::move(receiver)});
}
Future<kvstore::DriverPtr> FileKeyValueStoreSpec::DoOpen() const {
  auto driver_ptr = internal::MakeIntrusivePtr<FileKeyValueStore>();
  driver_ptr->spec_ = data_;
  return driver_ptr;
}
Result<kvstore::Spec> ParseFileUrl(std::string_view url) {
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == internal_file_kvstore::FileKeyValueStoreSpec::id);
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  std::string path = internal::PercentDecode(parsed.authority_and_path);
  auto driver_spec = internal::MakeIntrusivePtr<FileKeyValueStoreSpec>();
  driver_spec->data_.file_io_concurrency =
      Context::Resource<internal::FileIoConcurrencyResource>::DefaultSpec();
  driver_spec->data_.file_io_sync =
      Context::Resource<FileIoSyncResource>::DefaultSpec();
  return {std::in_place, std::move(driver_spec), std::move(path)};
}
}  
}  
}  
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_file_kvstore::FileKeyValueStore)
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_file_kvstore::FileKeyValueStoreSpec>
    registration;
const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{
        tensorstore::internal_file_kvstore::FileKeyValueStoreSpec::id,
        tensorstore::internal_file_kvstore::ParseFileUrl};
}  