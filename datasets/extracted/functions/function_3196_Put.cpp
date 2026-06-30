#include <libaddressinput/null_storage.h>
#include <cassert>
#include <cstddef>
#include <string>
namespace i18n {
namespace addressinput {
NullStorage::NullStorage() = default;
NullStorage::~NullStorage() = default;
void NullStorage::Put(const std::string& key, std::string* data) {
  assert(data != nullptr);  
  delete data;
}
void NullStorage::Get(const std::string& key,
                      const Callback& data_ready) const {
  data_ready(false, key, nullptr);
}
}  
}  