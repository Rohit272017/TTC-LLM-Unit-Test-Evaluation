#ifndef PYTHON_TENSORSTORE_KEYWORD_ARGUMENTS_H_
#define PYTHON_TENSORSTORE_KEYWORD_ARGUMENTS_H_
#include <pybind11/pybind11.h>
#include <string_view>
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/type_name_override.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_python {
template <typename T>
struct KeywordArgumentPlaceholder {
  pybind11::object value;
  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("Optional[") +
      pybind11::detail::make_caster<T>::name + pybind11::detail::_("]");
};
template <typename ParamDef>
using KeywordArgument = KeywordArgumentPlaceholder<typename ParamDef::type>;
template <typename ParamDef>
void AppendKeywordArgumentDoc(std::string& doc) {
  tensorstore::StrAppend(&doc, "  ", ParamDef::name, ": ");
  std::string_view delim = "";
  for (std::string_view line :
       absl::StrSplit(absl::StripAsciiWhitespace(ParamDef::doc), '\n')) {
    tensorstore::StrAppend(&doc, delim, line, "\n");
    delim = "    ";
  }
}
template <typename... ParamDef>
void AppendKeywordArgumentDocs(std::string& doc, ParamDef... params) {
  (AppendKeywordArgumentDoc<ParamDef>(doc), ...);
}
template <typename ParamDef>
auto MakeKeywordArgumentPyArg(ParamDef param_def) {
  return (pybind11::arg(decltype(param_def)::name) = pybind11::none());
}
template <typename ParamDef, typename Target>
void SetKeywordArgumentOrThrow(Target& target, KeywordArgument<ParamDef>& arg) {
  if (arg.value.is_none()) return;
  pybind11::detail::make_caster<typename ParamDef::type> caster;
  if (!caster.load(arg.value, true)) {
    throw pybind11::type_error(tensorstore::StrCat("Invalid ", ParamDef::name));
  }
  auto status = ParamDef::Apply(
      target,
      pybind11::detail::cast_op<typename ParamDef::type&&>(std::move(caster)));
  if (!status.ok()) {
    ThrowStatusException(MaybeAnnotateStatus(
        status, tensorstore::StrCat("Invalid ", ParamDef::name)));
  }
}
template <typename... ParamDef, typename Target>
void ApplyKeywordArguments(Target& target, KeywordArgument<ParamDef>&... arg) {
  (SetKeywordArgumentOrThrow<ParamDef>(target, arg), ...);
}
}  
}  
#endif  