#include "tsl/platform/file_system.h"
#include <sys/stat.h>
#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "tsl/platform/status.h"
#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
    defined(PLATFORM_GOOGLE)
#include <fnmatch.h>
#else
#include "tsl/platform/regexp.h"
#endif  
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/scanner.h"
#include "tsl/platform/str_util.h"
#include "tsl/platform/strcat.h"
namespace tsl {
bool FileSystem::Match(const string& filename, const string& pattern) {
#if defined(PLATFORM_POSIX) || defined(IS_MOBILE_PLATFORM) || \
    defined(PLATFORM_GOOGLE)
  return fnmatch(pattern.c_str(), filename.c_str(), FNM_PATHNAME) == 0;
#else
  string regexp(pattern);
  regexp = str_util::StringReplace(regexp, "*", "[^/]*", true);
  regexp = str_util::StringReplace(regexp, "?", ".", true);
  regexp = str_util::StringReplace(regexp, "(", "\\(", true);
  regexp = str_util::StringReplace(regexp, ")", "\\)", true);
  return RE2::FullMatch(filename, regexp);
#endif  
}
string FileSystem::TranslateName(const string& name) const {
  if (name.empty()) return name;
  absl::string_view scheme, host, path;
  this->ParseURI(name, &scheme, &host, &path);
  if (path.empty()) return "/";
  return this->CleanPath(path);
}
absl::Status FileSystem::IsDirectory(const string& name,
                                     TransactionToken* token) {
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return absl::OkStatus();
  }
  return absl::Status(absl::StatusCode::kFailedPrecondition, "Not a directory");
}
absl::Status FileSystem::HasAtomicMove(const string& path,
                                       bool* has_atomic_move) {
  *has_atomic_move = true;
  return absl::OkStatus();
}
absl::Status FileSystem::CanCreateTempFile(const std::string& fname,
                                           bool* can_create_temp_file) {
  *can_create_temp_file = true;
  return absl::OkStatus();
}
void FileSystem::FlushCaches(TransactionToken* token) {}
bool FileSystem::FilesExist(const std::vector<string>& files,
                            TransactionToken* token,
                            std::vector<absl::Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    absl::Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      return false;
    }
  }
  return result;
}
absl::Status FileSystem::DeleteRecursively(const string& dirname,
                                           TransactionToken* token,
                                           int64_t* undeleted_files,
                                           int64_t* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);
  *undeleted_files = 0;
  *undeleted_dirs = 0;
  absl::Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  if (!IsDirectory(dirname).ok()) {
    absl::Status delete_root_status = DeleteFile(dirname);
    if (!delete_root_status.ok()) (*undeleted_files)++;
    return delete_root_status;
  }
  std::deque<string> dir_q;      
  std::vector<string> dir_list;  
  dir_q.push_back(dirname);
  absl::Status ret;  
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    absl::Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = this->JoinPath(dir, child);
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        absl::Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  std::reverse(dir_list.begin(), dir_list.end());
  for (const string& dir : dir_list) {
    absl::Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}
absl::Status FileSystem::RecursivelyCreateDir(const string& dirname,
                                              TransactionToken* token) {
  absl::string_view scheme, host, remaining_dir;
  this->ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<absl::string_view> sub_dirs;
  while (!remaining_dir.empty()) {
    std::string current_entry = this->CreateURI(scheme, host, remaining_dir);
    absl::Status exists_status = FileExists(current_entry);
    if (exists_status.ok()) {
      absl::Status directory_status = IsDirectory(current_entry);
      if (directory_status.ok()) {
        break;  
      } else if (directory_status.code() == absl::StatusCode::kUnimplemented) {
        return directory_status;
      } else {
        return errors::FailedPrecondition(remaining_dir, " is not a directory");
      }
    }
    if (exists_status.code() != error::Code::NOT_FOUND) {
      return exists_status;
    }
    if (!absl::EndsWith(remaining_dir, "/")) {
      sub_dirs.push_back(this->Basename(remaining_dir));
    }
    remaining_dir = this->Dirname(remaining_dir);
  }
  std::reverse(sub_dirs.begin(), sub_dirs.end());
  string built_path(remaining_dir);
  for (const absl::string_view sub_dir : sub_dirs) {
    built_path = this->JoinPath(built_path, sub_dir);
    absl::Status status = CreateDir(this->CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != absl::StatusCode::kAlreadyExists) {
      return status;
    }
  }
  return absl::OkStatus();
}
absl::Status FileSystem::CopyFile(const string& src, const string& target,
                                  TransactionToken* token) {
  return FileSystemCopyFile(this, src, this, target);
}
char FileSystem::Separator() const { return '/'; }
string FileSystem::JoinPathImpl(
    std::initializer_list<absl::string_view> paths) {
  string result;
  for (absl::string_view path : paths) {
    if (path.empty()) continue;
    if (result.empty()) {
      result = string(path);
      continue;
    }
    if (result[result.size() - 1] == '/') {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path.substr(1));
      } else {
        strings::StrAppend(&result, path);
      }
    } else {
      if (this->IsAbsolutePath(path)) {
        strings::StrAppend(&result, path);
      } else {
        strings::StrAppend(&result, "/", path);
      }
    }
  }
  return result;
}
std::pair<absl::string_view, absl::string_view> FileSystem::SplitPath(
    absl::string_view uri) const {
  absl::string_view scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);
  if (path.empty()) {
    return std::make_pair(absl::string_view(), absl::string_view());
  }
  size_t pos = path.rfind(this->Separator());
#ifdef PLATFORM_WINDOWS
  size_t pos2 = path.rfind('/');
  if (pos == string::npos) {
    pos = pos2;
  } else {
    if (pos2 != string::npos) {
      pos = pos > pos2 ? pos : pos2;
    }
  }
#endif
  if (pos == absl::string_view::npos) {
    if (host.empty()) {
      return std::make_pair(absl::string_view(), path);
    }
    return std::make_pair(
        absl::string_view(uri.data(), host.end() - uri.begin()), path);
  }
  if (pos == 0) {
    return std::make_pair(
        absl::string_view(uri.data(), path.begin() + 1 - uri.begin()),
        absl::string_view(path.data() + 1, path.size() - 1));
  }
  return std::make_pair(
      absl::string_view(uri.data(), path.begin() + pos - uri.begin()),
      absl::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}
bool FileSystem::IsAbsolutePath(absl::string_view path) const {
  return !path.empty() && path[0] == '/';
}
absl::string_view FileSystem::Dirname(absl::string_view path) const {
  return this->SplitPath(path).first;
}
absl::string_view FileSystem::Basename(absl::string_view path) const {
  return this->SplitPath(path).second;
}
absl::string_view FileSystem::Extension(absl::string_view path) const {
  absl::string_view basename = this->Basename(path);
  size_t pos = basename.rfind('.');
  if (pos == absl::string_view::npos) {
    return absl::string_view(path.data() + path.size(), 0);
  } else {
    return absl::string_view(path.data() + pos + 1, path.size() - (pos + 1));
  }
}
string FileSystem::CleanPath(absl::string_view unclean_path) const {
  string path(unclean_path);
  const char* src = path.c_str();
  string::iterator dst = path.begin();
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  string::const_iterator backtrack_limit = dst;
  while (*src) {
    bool parsed = false;
    if (src[0] == '.') {
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        src += 2;
        if (dst != backtrack_limit) {
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
          }
        } else if (!is_absolute_path) {
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }
    while (*src == '/') {
      ++src;
    }
  }
  string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    path.assign(1, '.');
  }
  return path;
}
void FileSystem::ParseURI(absl::string_view remaining,
                          absl::string_view* scheme, absl::string_view* host,
                          absl::string_view* path) const {
  if (!strings::Scanner(remaining)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral(":
           .GetResult(&remaining, scheme)) {
    *scheme = absl::string_view();
    *host = absl::string_view();
    *path = remaining;
    return;
  }
  if (!strings::Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
    *host = remaining;
    *path = absl::string_view();
    return;
  }
  *path = remaining;
}
string FileSystem::CreateURI(absl::string_view scheme, absl::string_view host,
                             absl::string_view path) const {
  if (scheme.empty()) {
    return string(path);
  }
  return strings::StrCat(scheme, ":
}
std::string FileSystem::DecodeTransaction(const TransactionToken* token) {
  if (token) {
    std::stringstream oss;
    oss << "Token= " << token->token << ", Owner=" << token->owner;
    return oss.str();
  }
  return "No Transaction";
}
}  