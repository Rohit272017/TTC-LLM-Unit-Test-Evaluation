#include "checker/type_check_issue.h"
#include <string>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "common/source.h"
namespace cel {
namespace {
absl::string_view SeverityString(TypeCheckIssue::Severity severity) {
  switch (severity) {
    case TypeCheckIssue::Severity::kInformation:
      return "INFORMATION";
    case TypeCheckIssue::Severity::kWarning:
      return "WARNING";
    case TypeCheckIssue::Severity::kError:
      return "ERROR";
    case TypeCheckIssue::Severity::kDeprecated:
      return "DEPRECATED";
    default:
      return "SEVERITY_UNSPECIFIED";
  }
}
}  
std::string TypeCheckIssue::ToDisplayString(const Source& source) const {
  return absl::StrCat(
      absl::StrFormat("%s: %s:%d:%d: %s", SeverityString(severity_),
                      source.description(), location_.line, location_.column,
                      message_),
      source.DisplayErrorLocation(location_));
}
}  