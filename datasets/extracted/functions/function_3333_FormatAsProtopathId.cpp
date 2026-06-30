#include "arolla/naming/protopath_id.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "arolla/naming/table.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::naming {
namespace {
std::string FormatAsProtopathId(const std::vector<PathSegment>& segments) {
  return absl::StrJoin(segments, "",
                       [](std::string* ret, const PathSegment& segment) {
                         absl::StrAppend(ret, "/", segment.FieldName(),
                                         segment.IsIndex() ? kIndexMarker : "");
                       });
}
PathSegment ParsePathSegment(absl::string_view segment_name) {
  bool is_index = absl::ConsumeSuffix(&segment_name, kIndexMarker);
  return PathSegment(segment_name, is_index);
}
absl::StatusOr<std::vector<PathSegment>> ParseProtopathId(
    absl::string_view protopath_id) {
  std::vector<PathSegment> parsed;
  if (protopath_id.empty()) {
    return parsed;  
  }
  if (protopath_id[0] != '/') {
    return absl::InvalidArgumentError(
        absl::StrFormat("ProtopathId (%s) formatted incorrectly. "
                        "Must start with a slash (/).",
                        protopath_id));
  }
  protopath_id.remove_prefix(1);
  while (!protopath_id.empty()) {
    const size_t segment_len = protopath_id.find_first_of('/');
    const absl::string_view segment = protopath_id.substr(0, segment_len);
    parsed.push_back(ParsePathSegment(segment));
    if (segment_len == std::string::npos) {
      break;
    }
    protopath_id.remove_prefix(segment_len + 1);
  }
  return parsed;
}
}  
std::string TablePathToProtopathId(const TablePath& table_path) {
  return FormatAsProtopathId(table_path.PathSegments());
}
std::string ColumnPathToProtopathId(const ColumnPath& column_path) {
  return FormatAsProtopathId(column_path.PathSegments());
}
absl::StatusOr<TablePath> TablePathFromProtopathId(
    absl::string_view protopath_id) {
  ASSIGN_OR_RETURN(auto segments, ParseProtopathId(protopath_id));
  return TablePath(std::move(segments));
}
absl::StatusOr<ColumnPath> ColumnPathFromProtopathId(
    absl::string_view protopath_id) {
  ASSIGN_OR_RETURN(auto segments, ParseProtopathId(protopath_id));
  return ColumnPath(std::move(segments));
}
}  