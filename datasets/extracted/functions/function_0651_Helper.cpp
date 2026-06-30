#include "rule_retriever.h"
#include <libaddressinput/callback.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include "retriever.h"
#include "rule.h"
namespace i18n {
namespace addressinput {
namespace {
class Helper {
 public:
  Helper(const Helper&) = delete;
  Helper& operator=(const Helper&) = delete;
  Helper(const std::string& key,
         const RuleRetriever::Callback& rule_ready,
         const Retriever& data_retriever)
      : rule_ready_(rule_ready),
        data_retrieved_(BuildCallback(this, &Helper::OnDataRetrieved)) {
    data_retriever.Retrieve(key, *data_retrieved_);
  }
 private:
  ~Helper() = default;
  void OnDataRetrieved(bool success,
                       const std::string& key,
                       const std::string& data) {
    Rule rule;
    if (!success) {
      rule_ready_(false, key, rule);
    } else {
      success = rule.ParseSerializedRule(data);
      rule_ready_(success, key, rule);
    }
    delete this;
  }
  const RuleRetriever::Callback& rule_ready_;
  const std::unique_ptr<const Retriever::Callback> data_retrieved_;
};
}  
RuleRetriever::RuleRetriever(const Retriever* retriever)
    : data_retriever_(retriever) {
  assert(data_retriever_ != nullptr);
}
RuleRetriever::~RuleRetriever() = default;
void RuleRetriever::RetrieveRule(const std::string& key,
                                 const Callback& rule_ready) const {
  new Helper(key, rule_ready, *data_retriever_);
}
}  
}  