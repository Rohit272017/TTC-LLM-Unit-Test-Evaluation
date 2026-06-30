#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "arolla/codegen/expr/types.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/dict/dict_types.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/typed_ref.h"
#include "arolla/util/bytes.h"
#include "arolla/util/text.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::codegen {
namespace {
template <class T>
absl::StatusOr<std::string> CppDictLiteralRepr(TypedRef dict_ref) {
  ASSIGN_OR_RETURN(const KeyToRowDict<T>& dict, dict_ref.As<KeyToRowDict<T>>());
  std::vector<std::pair<T, int64_t>> sorted_dict(dict.map().begin(),
                                                 dict.map().end());
  std::sort(sorted_dict.begin(), sorted_dict.end());
  std::ostringstream oss;
  ASSIGN_OR_RETURN(std::string type_name, CppTypeName(::arolla::GetQType<T>()));
  oss << "::arolla::KeyToRowDict<" << type_name << ">{";
  for (const auto& [k, v] : sorted_dict) {
    ASSIGN_OR_RETURN(std::string key_repr,
                     CppLiteralRepr(TypedRef::FromValue(k)));
    ASSIGN_OR_RETURN(std::string value_repr,
                     CppLiteralRepr(TypedRef::FromValue(v)));
    oss << "{" << key_repr << "," << value_repr << "},";
  }
  oss << "}";
  return oss.str();
}
#define REGISTER_CPP_TYPE(NAME, CTYPE)                                         \
  {                                                                            \
    auto status = []() -> absl::Status {                                       \
      ASSIGN_OR_RETURN(std::string type_name, CppTypeName(GetQType<CTYPE>())); \
      return RegisterCppType(                                                  \
          GetKeyToRowDictQType<CTYPE>(),                                       \
          absl::StrFormat("::arolla::KeyToRowDict<%s>", type_name),            \
          CppDictLiteralRepr<CTYPE>);                                          \
    }();                                                                       \
    if (!status.ok()) {                                                        \
      LOG(FATAL) << status.message();                                         \
    }                                                                          \
  }
int Register() {
  REGISTER_CPP_TYPE(INT32, int32_t);
  REGISTER_CPP_TYPE(INT64, int64_t);
  REGISTER_CPP_TYPE(UINT64, uint64_t);
  REGISTER_CPP_TYPE(BOOLEAN, bool);
  REGISTER_CPP_TYPE(BYTES, ::arolla::Bytes);
  REGISTER_CPP_TYPE(TEXT, ::arolla::Text);
  return 0;
}
int registered = Register();
}  
}  