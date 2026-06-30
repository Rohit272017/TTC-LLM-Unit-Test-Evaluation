#ifndef THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_STRUCTS_TRIVIAL_LEGACY_TYPE_INFO_H_
#define THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_STRUCTS_TRIVIAL_LEGACY_TYPE_INFO_H_
#include <string>
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "eval/public/message_wrapper.h"
#include "eval/public/structs/legacy_type_info_apis.h"
namespace google::api::expr::runtime {
class TrivialTypeInfo : public LegacyTypeInfoApis {
 public:
  absl::string_view GetTypename(const MessageWrapper& wrapper) const override {
    return "opaque";
  }
  std::string DebugString(const MessageWrapper& wrapper) const override {
    return "opaque";
  }
  const LegacyTypeAccessApis* GetAccessApis(
      const MessageWrapper& wrapper) const override {
    return nullptr;
  }
  static const TrivialTypeInfo* GetInstance() {
    static absl::NoDestructor<TrivialTypeInfo> kInstance;
    return &*kInstance;
  }
};
}  
#endif  