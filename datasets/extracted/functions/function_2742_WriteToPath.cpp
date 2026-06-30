#include "xla/text_literal_writer.h"
#include <memory>
#include <string>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/file_system.h"
namespace xla {
 absl::Status TextLiteralWriter::WriteToPath(
    const Literal& literal, absl::string_view path) {
  std::unique_ptr<tsl::WritableFile> f;
  auto s = tsl::Env::Default()->NewWritableFile(std::string(path), &f);
  if (!s.ok()) {
    return s;
  }
  s = f->Append(ShapeUtil::HumanString(literal.shape()) + "\n");
  if (!s.ok()) {
    return s;
  }
  absl::Status status;
  tsl::WritableFile* f_ptr = f.get();
  literal.EachCellAsString([f_ptr, &status](absl::Span<const int64_t> indices,
                                            const std::string& value) {
    if (!status.ok()) {
      return;
    }
    std::string coordinates =
        absl::StrCat("(", absl::StrJoin(indices, ", "), ")");
    status = f_ptr->Append(absl::StrCat(coordinates, ": ", value, "\n"));
  });
  auto ignored = f->Close();
  return status;
}
}  