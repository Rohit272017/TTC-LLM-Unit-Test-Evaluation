#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#ifndef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
#include <dlfcn.h>
#endif  
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#endif
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>  
#include <vector>
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/constants.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#if defined(__ANDROID__) && !defined(TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS)
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#endif  
namespace tflite {
namespace acceleration {
namespace {
std::string ShellEscape(const std::string& src);
}  
MinibenchmarkStatus ProcessRunner::Init() {
  if (!function_pointer_) {
    return kMinibenchmarkPreconditionNotMet;
  }
#if !defined(__ANDROID__) || defined(TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS)
  return kMinibenchmarkSuccess;
#else  
  tflite::acceleration::AndroidInfo android_info;
  if (!tflite::acceleration::RequestAndroidInfo(&android_info).ok()) {
    return kMinibenchmarkRequestAndroidInfoFailed;
  }
  if (android_info.android_sdk_version.length() < 2 ||
      android_info.android_sdk_version < "23") {
    return kMinibenchmarkUnsupportedPlatform;
  }
  std::string soname;
  Dl_info dl_info;
  int status = dladdr(function_pointer_, &dl_info);
  if (status != 0) {
    if (dl_info.dli_fname) {
      soname = dl_info.dli_fname;
    } else {
      return kMinibenchmarkDliFnameWasNull;
    }
  } else {
    return kMinibenchmarkDladdrReturnedZero;
  }
  if (soname.size() >= 4 && soname.substr(soname.size() - 4) == ".apk") {
    return kMinibenchmarkDliFnameHasApkNameOnly;
  }
  std::string runner_path;
  runner_path = temporary_path_ + "/runner";
  (void)unlink(runner_path.c_str());
  std::string runner_contents(
      reinterpret_cast<const char*>(g_tflite_acceleration_embedded_runner),
      g_tflite_acceleration_embedded_runner_len);
  std::ofstream f(runner_path, std::ios::binary);
  if (!f.is_open()) {
    return kMinibenchmarkCouldntOpenTemporaryFileForBinary;
  }
  f << runner_contents;
  f.close();
  if (chmod(runner_path.c_str(), 0500) != 0) {
    return kMinibenchmarkCouldntChmodTemporaryFile;
  }
  runner_path = ShellEscape(runner_path);
  if (android_info.android_sdk_version >= "29") {
#if defined(__arm__) || defined(__i386__)
    std::string linker_path = "/system/bin/linker";
#else
    std::string linker_path = "/system/bin/linker64";
#endif
    runner_path = linker_path + " " + runner_path;
  }
  runner_path_ = runner_path;
  soname_ = soname;
  return kMinibenchmarkSuccess;
#endif  
}
#ifndef _WIN32
bool ProcessRunner::KillProcessWhenTimedOut(FILE* fstream) {
  const int array_length = 1 + kPidBufferLength;
  char buffer[array_length];
  memset(buffer, '\0', array_length);
  ssize_t length = fread(buffer, 1, kPidBufferLength, fstream);
  int pid;
  if (length != kPidBufferLength || !absl::SimpleAtoi(buffer, &pid)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed to get Validator subprocess id: %s", buffer);
    return false;
  }
  struct pollfd pfd[1];
  pfd[0].fd = fileno(fstream);
  pfd[0].events = POLLHUP;
  int poll_ret = poll(pfd, 1, timeout_millisec_);
  if (poll_ret == 0) {
    kill(pid, SIGKILL);
    return true;
  } else if (poll_ret < 0) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Validator timer failed: %s",
                         strerror(errno));
  }
  return false;
}
#endif  
MinibenchmarkStatus ProcessRunner::Run(const Allocation* model_allocation,
                                       const std::vector<std::string>& args,
                                       std::string* output, int* exitcode,
                                       int* signal) {
#ifdef _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#else  
  if (!output || !exitcode) {
    return kMinibenchmarkPreconditionNotMet;
  }
  int benchmark_process_status = 0;
  MinibenchmarkStatus status = kMinibenchmarkCommandFailed;
#ifdef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
  if (function_pointer_) {
    benchmark_process_status = RunInprocess(model_allocation, args);
  } else {
    return kMinibenchmarkPreconditionNotMet;
  }
#else   
  if (runner_path_.empty()) {
    return kMinibenchmarkPreconditionNotMet;
  }
  std::string cmd = runner_path_ + " " + ShellEscape(soname_) + " " +
                    ShellEscape(function_name_);
  int pipe_fds[2];
  if (model_allocation != nullptr) {
    if (pipe(pipe_fds) < 0) {
      *exitcode = errno;
      return kMinibenchmarkPipeFailed;
    }
    std::string pipe_model_path = absl::StrCat(
        "pipe:", pipe_fds[0], ":", pipe_fds[1], ":", model_allocation->bytes());
    cmd = cmd + " " + ShellEscape(pipe_model_path);
  }
  for (const auto& arg : args) {
    cmd = cmd + " " + ShellEscape(arg);
  }
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) {
    *exitcode = errno;
    return kMinibenchmarkPopenFailed;
  }
  if (model_allocation != nullptr) {
    close(pipe_fds[0]);
    int written_bytes = 0;
    int remaining_bytes = model_allocation->bytes();
    const uint8_t* current =
        static_cast<const uint8_t*>(model_allocation->base());
    while (remaining_bytes > 0 &&
           (written_bytes = write(pipe_fds[1], current, remaining_bytes)) > 0) {
      remaining_bytes -= written_bytes;
      current += written_bytes;
    }
    close(pipe_fds[1]);
    if (written_bytes <= 0 || remaining_bytes > 0) {
      *exitcode = errno;
      return kMinibenchmarkPipeFailed;
    }
  }
  if (timeout_millisec_ > 0 && KillProcessWhenTimedOut(f)) {
    status = kMinibenchmarkCommandTimedOut;
    TFLITE_LOG_PROD(
        TFLITE_LOG_INFO,
        "Validator did not finish after %dms. Tried to kill the test.",
        timeout_millisec_);
  }
  std::vector<char> buffer(4 * 1024, 0);
  ssize_t length;
  std::string ret;
  do {
    length = fread(buffer.data(), 1, buffer.size(), f);
    ret = ret + std::string(buffer.data(), length);
  } while (length == buffer.size());
  *output = ret;
  benchmark_process_status = pclose(f);
#endif  
  if (WIFEXITED(benchmark_process_status)) {
    *exitcode = WEXITSTATUS(benchmark_process_status);
    *signal = 0;
    if (*exitcode == kMinibenchmarkSuccess) {
      status = kMinibenchmarkSuccess;
    }
  } else if (WIFSIGNALED(benchmark_process_status)) {
    *exitcode = 0;
    *signal = WTERMSIG(benchmark_process_status);
  }
  return status;
#endif  
}
#ifdef TFLITE_ACCELERATION_BENCHMARK_IN_PROCESS
#ifndef __W_EXITCODE  
#define __W_EXITCODE(ret, sig) ((ret) << 8 | (sig))
#endif
int ProcessRunner::RunInprocess(const Allocation* model_allocation,
                                const std::vector<std::string>& user_args) {
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Running Validator in-process.");
  std::vector<std::string> args_string;
  args_string.push_back("inprocess");
  args_string.push_back("inprocess");
  args_string.push_back(function_name_);
  std::thread write_thread;
  if (model_allocation != nullptr) {
    int pipe_fds[2];
    if (pipe(pipe_fds) < 0) {
      return __W_EXITCODE(kMinibenchmarkPipeFailed, 0);
    }
    args_string.push_back(
        absl::StrCat("pipe:", pipe_fds[0], ":-1:", model_allocation->bytes()));
    write_thread = std::thread([pipe_fds, model_allocation,
                                error_reporter = error_reporter_]() {
      int written_bytes = 0;
      int remaining_bytes = model_allocation->bytes();
      const uint8_t* current =
          static_cast<const uint8_t*>(model_allocation->base());
      while (remaining_bytes > 0 &&
             (written_bytes = write(pipe_fds[1], current, remaining_bytes)) >
                 0) {
        remaining_bytes -= written_bytes;
        current += written_bytes;
      }
      close(pipe_fds[1]);
      if (written_bytes < 0 || remaining_bytes > 0) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Failed to write Model to pipe: %s. Expect to write %d "
            "bytes, %d bytes written.",
            strerror(errno), remaining_bytes, written_bytes);
      }
    });
  }
  for (int i = 0; i < user_args.size(); i++) {
    args_string.push_back(user_args[i]);
  }
  std::vector<std::vector<char>> args_char(args_string.size());
  std::vector<char*> argv(args_string.size());
  for (int i = 0; i < args_string.size(); i++) {
    args_char[i] = {args_string[i].begin(), args_string[i].end()};
    args_char[i].push_back('\0');
    argv[i] = args_char[i].data();
  }
  int (*function_pointer)(int, char**) =
      reinterpret_cast<int (*)(int, char**)>(function_pointer_);
  int exit_code = __W_EXITCODE(function_pointer(argv.size(), argv.data()), 0);
  if (write_thread.joinable()) {
    write_thread.join();
  }
  return exit_code;
}
#endif  
namespace {
static const char kDontNeedShellEscapeChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+-_.=/:,@";
std::string ShellEscape(const std::string& src) {
  if (!src.empty() &&  
      src.find_first_not_of(kDontNeedShellEscapeChars) == std::string::npos) {
    return src;
  } else if (src.find('\'') == std::string::npos) {  
    return "'" + src + "'";
  } else {
    std::string result = "\"";
    for (const char c : src) {
      switch (c) {
        case '\\':
        case '$':
        case '"':
        case '`':
          result.push_back('\\');
      }
      result.push_back(c);
    }
    result.push_back('"');
    return result;
  }
}
}  
}  
}  