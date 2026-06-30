#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"
namespace tensorflow {
namespace generator {
namespace cpp {
Renderer::Renderer(RendererContext context) : context_(context) {}
Renderer& Renderer::BlankLine() {
  context_.code.AddLineWithoutIndent("");
  return *this;
}
Renderer& Renderer::CodeLine(const string& text) {
  context_.code.AddLineWithoutIndent(text);
  return *this;
}
Renderer& Renderer::CodeLines(const string& text) {
  StringPiece trimmed_text(text);
  str_util::RemoveWhitespaceContext(&trimmed_text);
  for (const string& line : str_util::Split(trimmed_text, '\n')) {
    context_.code.AddLineWithoutIndent(line);
  }
  return *this;
}
Renderer& Renderer::Statement(const string& text) {
  if (absl::EndsWith(text, ";")) {
    LOG(WARNING) << "Superfluous terminating ';' in '" << text << "'";
    context_.code.AddLineWithIndent(text);
  } else {
    context_.code.AddLineWithIndent(absl::StrCat(text, ";"));
  }
  return *this;
}
Renderer& Renderer::TFStatement(const string& text) {
  return Statement(absl::Substitute("TF_RETURN_IF_ERROR($0)", text));
}
Renderer& Renderer::CommentLine(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat("
  return *this;
}
Renderer& Renderer::BlockOpen(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat(text, " {"));
  context_.code.IncreaseIndent();
  return *this;
}
Renderer& Renderer::BlockClose(const string& text) {
  context_.code.DecreaseIndent();
  context_.code.AddLineWithIndent(absl::StrCat("}", text));
  return *this;
}
}  
}  
}  