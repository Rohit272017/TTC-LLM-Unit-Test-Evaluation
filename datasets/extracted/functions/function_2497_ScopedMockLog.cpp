#ifndef GLOG_SRC_MOCK_LOG_H_
#define GLOG_SRC_MOCK_LOG_H_
#include <gmock/gmock.h>
#include <string>
#include "glog/logging.h"
#include "utilities.h"
namespace google {
namespace glog_testing {
class ScopedMockLog : public google::LogSink {
 public:
  ScopedMockLog() { AddLogSink(this); }
  ~ScopedMockLog() override { RemoveLogSink(this); }
  MOCK_METHOD3(Log,
               void(google::LogSeverity severity, const std::string& file_path,
                    const std::string& message));
 private:
  void send(google::LogSeverity severity, const char* full_filename,
            const char* , int ,
            const LogMessageTime& , const char* message,
            size_t message_len) override {
    message_info_.severity = severity;
    message_info_.file_path = full_filename;
    message_info_.message = std::string(message, message_len);
  }
  void WaitTillSent() override {
    MessageInfo message_info = message_info_;
    Log(message_info.severity, message_info.file_path, message_info.message);
  }
  struct MessageInfo {
    google::LogSeverity severity;
    std::string file_path;
    std::string message;
  };
  MessageInfo message_info_;
};
}  
}  
#endif  