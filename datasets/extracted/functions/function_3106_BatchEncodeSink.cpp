#include "tensorstore/serialization/batch.h"
#include "absl/status/status.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace serialization {
BatchEncodeSink::BatchEncodeSink(riegeli::Writer& writer)
    : EncodeSink(writer) {}
BatchEncodeSink::~BatchEncodeSink() = default;
bool BatchEncodeSink::DoIndirect(const std::type_info& type,
                                 ErasedEncodeWrapperFunction encode,
                                 std::shared_ptr<void> object) {
  auto [it, inserted] = indirect_map_.emplace(object, indirect_map_.size());
  return serialization::WriteSize(writer(), it->second) &&
         (!inserted || encode(*this, object));
}
BatchDecodeSource::BatchDecodeSource(riegeli::Reader& reader)
    : DecodeSource(reader) {}
BatchDecodeSource::~BatchDecodeSource() = default;
bool BatchDecodeSource::DoIndirect(const std::type_info& type,
                                   ErasedDecodeWrapperFunction decode,
                                   std::shared_ptr<void>& value) {
  size_t id;
  if (!serialization::ReadSize(reader(), id)) return false;
  if (id > indirect_objects_.size()) {
    Fail(DecodeError(tensorstore::StrCat("Indirect object index ", id,
                                         " out of range [0, ",
                                         indirect_objects_.size(), ")")));
    return false;
  }
  if (id < indirect_objects_.size()) {
    auto& entry = indirect_objects_[id];
    if (*entry.type != type) {
      Fail(absl::InvalidArgumentError(tensorstore::StrCat(
          "Type mismatch for indirect object, received ", entry.type->name(),
          " but expected ", type.name())));
      return false;
    }
    value = entry.value;
    return true;
  }
  indirect_objects_.emplace_back();
  if (!decode(*this, value)) return false;
  auto& entry = indirect_objects_[id];
  entry.type = &type;
  entry.value = value;
  return true;
}
}  
}  