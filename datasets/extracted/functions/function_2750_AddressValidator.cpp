#include <libaddressinput/address_validator.h>
#include <cassert>
#include <cstddef>
#include "validation_task.h"
namespace i18n {
namespace addressinput {
AddressValidator::AddressValidator(Supplier* supplier) : supplier_(supplier) {
  assert(supplier_ != nullptr);
}
void AddressValidator::Validate(const AddressData& address,
                                bool allow_postal,
                                bool require_name,
                                const FieldProblemMap* filter,
                                FieldProblemMap* problems,
                                const Callback& validated) const {
  (new ValidationTask(
       address,
       allow_postal,
       require_name,
       filter,
       problems,
       validated))->Run(supplier_);
}
}  
}  