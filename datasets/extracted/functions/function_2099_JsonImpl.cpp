#include "json.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <rapidjson/document.h>
#include <rapidjson/reader.h>
namespace i18n {
namespace addressinput {
using rapidjson::Document;
using rapidjson::kParseValidateEncodingFlag;
using rapidjson::Value;
class Json::JsonImpl {
 public:
  JsonImpl(const JsonImpl&) = delete;
  JsonImpl& operator=(const JsonImpl&) = delete;
  explicit JsonImpl(const std::string& json)
      : document_(new Document),
        value_(document_.get()),
        dictionaries_(),
        valid_(false) {
    document_->Parse<kParseValidateEncodingFlag>(json.c_str());
    valid_ = !document_->HasParseError() && document_->IsObject();
  }
  ~JsonImpl() {
    for (auto ptr : dictionaries_) {
      delete ptr;
    }
  }
  bool valid() const { return valid_; }
  const std::vector<const Json*>& GetSubDictionaries() {
    if (dictionaries_.empty()) {
      for (Value::ConstMemberIterator member = value_->MemberBegin();
           member != value_->MemberEnd(); ++member) {
        if (member->value.IsObject()) {
          dictionaries_.push_back(new Json(new JsonImpl(&member->value)));
        }
      }
    }
    return dictionaries_;
  }
  bool GetStringValueForKey(const std::string& key, std::string* value) const {
    assert(value != nullptr);
    Value::ConstMemberIterator member = value_->FindMember(key.c_str());
    if (member == value_->MemberEnd() || !member->value.IsString()) {
      return false;
    }
    value->assign(member->value.GetString(), member->value.GetStringLength());
    return true;
  }
 private:
  explicit JsonImpl(const Value* value)
      : document_(),
        value_(value),
        dictionaries_(),
        valid_(true) {
    assert(value_ != nullptr);
    assert(value_->IsObject());
  }
  const std::unique_ptr<Document> document_;
  const Value* const value_;
  std::vector<const Json*> dictionaries_;
  bool valid_;
};
Json::Json() : impl_() {}
Json::~Json() = default;
bool Json::ParseObject(const std::string& json) {
  assert(impl_ == nullptr);
  impl_.reset(new JsonImpl(json));
  if (!impl_->valid()) {
    impl_.reset();
  }
  return impl_ != nullptr;
}
const std::vector<const Json*>& Json::GetSubDictionaries() const {
  assert(impl_ != nullptr);
  return impl_->GetSubDictionaries();
}
bool Json::GetStringValueForKey(const std::string& key,
                                std::string* value) const {
  assert(impl_ != nullptr);
  return impl_->GetStringValueForKey(key, value);
}
Json::Json(JsonImpl* impl) : impl_(impl) {}
}  
}  