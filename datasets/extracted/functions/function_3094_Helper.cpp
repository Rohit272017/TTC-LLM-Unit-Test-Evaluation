#include "retriever.h"
#include <libaddressinput/callback.h>
#include <libaddressinput/source.h>
#include <libaddressinput/storage.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include "validating_storage.h"
namespace i18n {
namespace addressinput {
namespace {
class Helper {
 public:
  Helper(const Helper&) = delete;
  Helper& operator=(const Helper&) = delete;
  Helper(const std::string& key,
         const Retriever::Callback& retrieved,
         const Source& source,
         ValidatingStorage* storage)
      : retrieved_(retrieved),
        source_(source),
        storage_(storage),
        fresh_data_ready_(BuildCallback(this, &Helper::OnFreshDataReady)),
        validated_data_ready_(
            BuildCallback(this, &Helper::OnValidatedDataReady)),
        stale_data_() {
    assert(storage_ != nullptr);
    storage_->Get(key, *validated_data_ready_);
  }
 private:
  ~Helper() = default;
  void OnValidatedDataReady(bool success,
                            const std::string& key,
                            std::string* data) {
    if (success) {
      assert(data != nullptr);
      retrieved_(success, key, *data);
      delete this;
    } else {
      if (data != nullptr && !data->empty()) {
        stale_data_ = *data;
      }
      source_.Get(key, *fresh_data_ready_);
    }
    delete data;
  }
  void OnFreshDataReady(bool success,
                        const std::string& key,
                        std::string* data) {
    if (success) {
      assert(data != nullptr);
      retrieved_(true, key, *data);
      storage_->Put(key, data);
      data = nullptr;  
    } else if (!stale_data_.empty()) {
      retrieved_(true, key, stale_data_);
    } else {
      retrieved_(false, key, std::string());
    }
    delete data;
    delete this;
  }
  const Retriever::Callback& retrieved_;
  const Source& source_;
  ValidatingStorage* storage_;
  const std::unique_ptr<const Source::Callback> fresh_data_ready_;
  const std::unique_ptr<const Storage::Callback> validated_data_ready_;
  std::string stale_data_;
};
}  
Retriever::Retriever(const Source* source, Storage* storage)
    : source_(source), storage_(new ValidatingStorage(storage)) {
  assert(source_ != nullptr);
  assert(storage_ != nullptr);
}
Retriever::~Retriever() = default;
void Retriever::Retrieve(const std::string& key,
                         const Callback& retrieved) const {
  new Helper(key, retrieved, *source_, storage_.get());
}
}  
}  