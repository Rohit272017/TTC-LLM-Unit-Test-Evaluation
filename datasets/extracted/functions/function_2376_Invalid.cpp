#ifndef TENSORSTORE_INTERNAL_OS_FILE_UTIL_H_
#define TENSORSTORE_INTERNAL_OS_FILE_UTIL_H_
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <string_view>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/unique_handle.h"
#include "tensorstore/util/result.h"
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif
#include "tensorstore/internal/os/include_windows.h"
namespace tensorstore {
namespace internal_os {
#ifdef _WIN32
using FileDescriptor = HANDLE;  
struct FileDescriptorTraits {
  static FileDescriptor Invalid() { return ((FileDescriptor)-1); }
  static void Close(FileDescriptor fd);
};
using FileInfo = ::BY_HANDLE_FILE_INFORMATION;
constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }
#else
using FileDescriptor = int;
struct FileDescriptorTraits {
  static FileDescriptor Invalid() { return -1; }
  static void Close(FileDescriptor fd) { ::close(fd); }
};
typedef struct ::stat FileInfo;
constexpr inline bool IsDirSeparator(char c) { return c == '/'; }
#endif
inline constexpr std::string_view kLockSuffix = ".__lock";
using UniqueFileDescriptor = UniqueHandle<FileDescriptor, FileDescriptorTraits>;
Result<UniqueFileDescriptor> OpenExistingFileForReading(
    const std::string& path);
Result<UniqueFileDescriptor> OpenFileForWriting(const std::string& path);
Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset);
Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf, size_t count);
Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value);
absl::Status TruncateFile(FileDescriptor fd);
absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name);
absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path);
absl::Status DeleteFile(const std::string& path);
absl::Status FsyncFile(FileDescriptor fd);
using UnlockFn = void (*)(FileDescriptor fd);
Result<UnlockFn> AcquireFdLock(FileDescriptor fd);
absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info);
absl::Status GetFileInfo(const std::string& path, FileInfo* info);
inline bool IsRegularFile(const FileInfo& info) {
#ifdef _WIN32
  return !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  return S_ISREG(info.st_mode);
#endif
}
inline bool IsDirectory(const FileInfo& info) {
#ifdef _WIN32
  return (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  return S_ISDIR(info.st_mode);
#endif
}
inline uint64_t GetSize(const FileInfo& info) {
#ifdef _WIN32
  return (static_cast<int64_t>(info.nFileSizeHigh) << 32) +
         static_cast<int64_t>(info.nFileSizeLow);
#else
  return info.st_size;
#endif
}
inline auto GetDeviceId(const FileInfo& info) {
#ifdef _WIN32
  return info.dwVolumeSerialNumber;
#else
  return info.st_dev;
#endif
}
inline uint64_t GetFileId(const FileInfo& info) {
#ifdef _WIN32
  return (static_cast<uint64_t>(info.nFileIndexHigh) << 32) |
         static_cast<uint64_t>(info.nFileIndexLow);
#else
  return info.st_ino;
#endif
}
inline absl::Time GetMTime(const FileInfo& info) {
#ifdef _WIN32
  uint64_t windowsTicks =
      (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      static_cast<uint64_t>(info.ftLastWriteTime.dwLowDateTime);
  return absl::UnixEpoch() +
         absl::Seconds((windowsTicks / 10000000) - 11644473600ULL) +
         absl::Nanoseconds(windowsTicks % 10000000);
#else
#if defined(__APPLE__)
  const struct ::timespec t = info.st_mtimespec;
#else
  const struct ::timespec t = info.st_mtim;
#endif
  return absl::FromTimeT(t.tv_sec) + absl::Nanoseconds(t.tv_nsec);
#endif
}
Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path);
absl::Status MakeDirectory(const std::string& path);
absl::Status FsyncDirectory(FileDescriptor fd);
#ifdef _WIN32
Result<std::string> GetWindowsTempDir();
#endif
}  
}  
#endif  