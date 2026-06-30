#ifndef I18N_ADDRESSINPUT_SUPPLIER_H_
#define I18N_ADDRESSINPUT_SUPPLIER_H_
#include <libaddressinput/callback.h>
#include <string>
namespace i18n {
namespace addressinput {
class LookupKey;
class Rule;
class Supplier {
 public:
  struct RuleHierarchy;
  using Callback =
      i18n::addressinput::Callback<const LookupKey&, const RuleHierarchy&>;
  virtual ~Supplier() = default;
  virtual void Supply(const LookupKey& lookup_key,
                      const Callback& supplied) = 0;
  virtual void SupplyGlobally(const LookupKey& lookup_key,
                              const Callback& supplied) = 0;
  virtual size_t GetLoadedRuleDepth(const std::string& region_code) const {
    return 0;
  }
  struct RuleHierarchy {
    RuleHierarchy() : rule() {}
    const Rule* rule[4];  
  };
};
}  
}  
#endif  