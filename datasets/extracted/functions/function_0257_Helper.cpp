#include "validating_storage.h"
#include <libaddressinput/callback.h>
#include <libaddressinput/storage.h>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <memory>
#include <string>
#include "validating_util.h"
namespace i18n {
namespace addressinput {
namespace {
class Helper {
 public:
  Helper(const Helper&) = delete;
  Helper& operator=(const Helper&) = delete;
  Helper(const std::string& key,
         const ValidatingStorage::Callback& data_ready,
         const Storage& wrapped_storage)
      : data_ready_(data_ready),
        wrapped_data_ready_(BuildCallback(this, &Helper::OnWrappedDataReady)) {
    wrapped_storage.Get(key, *wrapped_data_ready_);
  }
 private:
  ~Helper() = default;
  void OnWrappedDataReady(bool success,
                          const std::string& key,
                          std::string* data) {
    if (success) {
      assert(data != nullptr);
      bool is_stale =
          !ValidatingUtil::UnwrapTimestamp(data, std::time(nullptr));
      bool is_corrupted = !ValidatingUtil::UnwrapChecksum(data);
      success = !is_corrupted && !is_stale;
      if (is_corrupted) {
        delete data;
        data = nullptr;
      }
    } else {
      delete data;
      data = nullptr;
    }
    data_ready_(success, key, data);
    delete this;
  }
  const Storage::Callback& data_ready_;
  const std::unique_ptr<const Storage::Callback> wrapped_data_ready_;
};
}  
ValidatingStorage::ValidatingStorage(Storage* storage)
    : wrapped_storage_(storage) {
  assert(wrapped_storage_ != nullptr);
}
ValidatingStorage::~ValidatingStorage() = default;
void ValidatingStorage::Put(const std::string& key, std::string* data) {
  assert(data != nullptr);
  ValidatingUtil::Wrap(std::time(nullptr), data);
  wrapped_storage_->Put(key, data);
}
void ValidatingStorage::Get(const std::string& key,
                            const Callback& data_ready) const {
  new Helper(key, data_ready, *wrapped_storage_);
}
}  
}  