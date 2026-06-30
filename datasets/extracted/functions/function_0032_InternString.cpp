#include "internal/string_pool.h"
#include <cstring>  
#include <string>   
#include "absl/base/optimization.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
namespace cel::internal {
absl::string_view StringPool::InternString(absl::string_view string) {
  if (string.empty()) {
    return "";
  }
  return *strings_.lazy_emplace(string, [&](const auto& ctor) {
    ABSL_ASSUME(arena_ != nullptr);
    char* data = google::protobuf::Arena::CreateArray<char>(arena_, string.size());
    std::memcpy(data, string.data(), string.size());
    ctor(absl::string_view(data, string.size()));
  });
}
}  