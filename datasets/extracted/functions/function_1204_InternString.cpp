#ifndef THIRD_PARTY_CEL_CPP_COMMON_ARENA_STRING_POOL_H_
#define THIRD_PARTY_CEL_CPP_COMMON_ARENA_STRING_POOL_H_
#include <memory>
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "common/arena_string.h"
#include "internal/string_pool.h"
#include "google/protobuf/arena.h"
namespace cel {
class ArenaStringPool;
absl::Nonnull<std::unique_ptr<ArenaStringPool>> NewArenaStringPool(
    absl::Nonnull<google::protobuf::Arena*> arena ABSL_ATTRIBUTE_LIFETIME_BOUND);
class ArenaStringPool final {
 public:
  ArenaStringPool(const ArenaStringPool&) = delete;
  ArenaStringPool(ArenaStringPool&&) = delete;
  ArenaStringPool& operator=(const ArenaStringPool&) = delete;
  ArenaStringPool& operator=(ArenaStringPool&&) = delete;
  ArenaString InternString(absl::string_view string) {
    return ArenaString(strings_.InternString(string));
  }
  ArenaString InternString(ArenaString) = delete;
 private:
  friend absl::Nonnull<std::unique_ptr<ArenaStringPool>> NewArenaStringPool(
      absl::Nonnull<google::protobuf::Arena*>);
  explicit ArenaStringPool(absl::Nonnull<google::protobuf::Arena*> arena)
      : strings_(arena) {}
  internal::StringPool strings_;
};
inline absl::Nonnull<std::unique_ptr<ArenaStringPool>> NewArenaStringPool(
    absl::Nonnull<google::protobuf::Arena*> arena ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  return std::unique_ptr<ArenaStringPool>(new ArenaStringPool(arena));
}
}  
#endif  