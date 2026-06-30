#ifndef TENSORSTORE_INTERNAL_OS_FILE_LISTER_H_
#define TENSORSTORE_INTERNAL_OS_FILE_LISTER_H_
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <string_view>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
namespace tensorstore {
namespace internal_os {
class ListerEntry {
 public:
  struct Impl;
  ListerEntry(Impl* impl) : impl_(impl) {}
  bool IsDirectory();
  const std::string& GetFullPath();
  std::string_view GetPathComponent();
  int64_t GetSize();
  absl::Status Delete();
 private:
  Impl* impl_;
};
absl::Status RecursiveFileList(
    std::string root_directory,
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item);
}  
}  
#endif  