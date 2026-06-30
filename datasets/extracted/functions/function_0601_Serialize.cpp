#include "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h"
#include <string>
#include <utility>
#include <vector>
namespace {
constexpr int kMod = (1 << 7);
void Serialize(std::string* out, uint32_t value) {
  for (; value >= kMod; value /= kMod) {
    out->push_back(value % kMod + kMod);
  }
  out->push_back(value);
}
bool Parse(const char** data, size_t* size, uint32_t* out) {
  *out = 0;
  uint32_t mul = 1;
  for (bool done = false; !done;
       mul *= kMod, done = !(**data & kMod), ++*data, --*size) {
    if (*size == 0) {
      return false;
    }
    *out += static_cast<unsigned char>(**data) % kMod * mul;
  }
  return true;
}
void Serialize(std::string* out, int32_t value) {
  Serialize(out, static_cast<uint32_t>(
                     value < 0 ? static_cast<uint32_t>(-(value + 1)) * 2 + 1
                               : static_cast<uint32_t>(value) * 2));
}
bool Parse(const char** data, size_t* size, int32_t* out) {
  uint32_t value = 0;
  if (!Parse(data, size, &value)) {
    return false;
  }
  const int32_t magnitude = value / 2;
  *out = (value % 2) ? (-magnitude - 1) : magnitude;
  return true;
}
template <class First, class Second>
void Serialize(std::string* out, const std::pair<First, Second>& in) {
  Serialize(out, in.first);
  Serialize(out, in.second);
}
template <class First, class Second>
bool Parse(const char** data, size_t* size, std::pair<First, Second>* out) {
  return Parse(data, size, &(out->first)) && Parse(data, size, &(out->second));
}
template <class Value>
void Serialize(std::string* out, const std::vector<Value>& in) {
  Serialize(out, static_cast<uint32_t>(in.size()));
  for (const auto& val : in) {
    Serialize(out, val);
  }
}
template <class T>
bool Parse(const char** data, size_t* size, std::vector<T>* out) {
  uint32_t num_elems = 0;
  if (!Parse(data, size, &num_elems)) {
    return false;
  }
  out->assign(num_elems, T{});
  for (auto& elem : *out) {
    if (!Parse(data, size, &elem)) {
      return false;
    }
  }
  return true;
}
}  
namespace tflite {
std::string SerializeModelControlDependencies(
    const ModelControlDependencies& in) {
  std::string out;
  Serialize(&out, kModelControlDependenciesMetadataVersion);
  Serialize(&out, in);
  return out;
}
bool ParseModelControlDependencies(const char* data, size_t size,
                                   ModelControlDependencies* out) {
  out->clear();
  uint32_t version = 0;
  return Parse(&data, &size, &version) &&
         (version == kModelControlDependenciesMetadataVersion) &&
         Parse(&data, &size, out) && (size == 0);
}
}  