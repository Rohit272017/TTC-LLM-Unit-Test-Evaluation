#include "tensorstore/proto/proto_util.h"
#include <stddef.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
namespace tensorstore {
namespace {
class ErrorCollector : public google::protobuf::io::ErrorCollector {
 public:
  ErrorCollector() = default;
  ~ErrorCollector() override = default;
  void RecordError(int line, google::protobuf::io::ColumnNumber column,
                   absl::string_view message) override {
    errors.emplace_back(absl::StrCat("Line: ", std::max(1, line + 1),
                                     ", col: ", column + 1, ": ", message));
  }
  void RecordWarning(int line, google::protobuf::io::ColumnNumber column,
                     absl::string_view message) override {
    errors.emplace_back(absl::StrCat("Line: ", std::max(1, line + 1),
                                     ", col: ", column + 1, ": ", message));
  }
  std::vector<std::string> errors;
};
class ConcisePrinter : public google::protobuf::TextFormat::FastFieldValuePrinter {
 public:
  void PrintString(
      const std::string& val,
      google::protobuf::TextFormat::BaseTextGenerator* generator) const override {
    if (val.size() <= 80) {
      FastFieldValuePrinter::PrintString(val, generator);
      return;
    }
    std::string output = absl::StrFormat("<%d bytes: ", val.size());
    for (size_t i = 0; i < 8; i++) {
      absl::StrAppendFormat(&output, "\\x%02x", val[i]);
    }
    absl::StrAppend(&output, "...>");
    generator->PrintString(output);
  }
};
}  
bool TryParseTextProto(absl::string_view asciipb, google::protobuf::Message* msg,
                       std::vector<std::string>* errors,
                       bool allow_partial_messages,
                       bool allow_unknown_extensions) {
  google::protobuf::TextFormat::Parser parser;
  parser.AllowPartialMessage(allow_partial_messages);
  parser.AllowUnknownExtension(allow_unknown_extensions);
  ErrorCollector error_collector;
  parser.RecordErrorsTo(&error_collector);
  google::protobuf::io::ArrayInputStream asciipb_istream(asciipb.data(), asciipb.size());
  if (parser.Parse(&asciipb_istream, msg)) {
    return true;
  }
  msg->Clear();  
  if (errors) {
    *errors = std::move(error_collector.errors);
  }
  return false;
}
std::string ConciseDebugString(const google::protobuf::Message& message) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetDefaultFieldValuePrinter(new ConcisePrinter());
  printer.SetSingleLineMode(true);
  printer.SetExpandAny(true);
  std::string debugstring;
  printer.PrintToString(message, &debugstring);
  if (!debugstring.empty() && debugstring.back() == ' ') {
    debugstring.pop_back();
  }
  return debugstring;
}
}  