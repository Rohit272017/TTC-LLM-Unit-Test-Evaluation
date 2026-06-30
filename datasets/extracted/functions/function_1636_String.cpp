#ifndef LANGSVR_JSON_BUILDER_
#define LANGSVR_JSON_BUILDER_
#include <memory>
#include <type_traits>
#include "langsvr/json/value.h"
#include "langsvr/span.h"
#include "langsvr/traits.h"
namespace langsvr::json {
class Value;
}
namespace langsvr::json {
class Builder {
  public:
    virtual ~Builder();
    static std::unique_ptr<Builder> Create();
    virtual Result<const Value*> Parse(std::string_view json) = 0;
    virtual const Value* Null() = 0;
    virtual const Value* Bool(json::Bool value) = 0;
    virtual const Value* I64(json::I64 value) = 0;
    virtual const Value* U64(json::U64 value) = 0;
    virtual const Value* F64(json::F64 value) = 0;
    virtual const Value* String(json::String value) = 0;
    const Value* String(std::string_view value) { return String(json::String(value)); }
    const Value* String(const char* value) { return String(json::String(value)); }
    virtual const Value* Array(Span<const Value*> elements) = 0;
    struct Member {
        json::String name;
        const Value* value;
    };
    virtual const Value* Object(Span<Member> members) = 0;
    template <typename T>
    auto Create(T&& value) {
        static constexpr bool is_bool = std::is_same_v<T, json::Bool>;
        static constexpr bool is_i64 = std::is_integral_v<T> && std::is_signed_v<T>;
        static constexpr bool is_u64 = std::is_integral_v<T> && std::is_unsigned_v<T>;
        static constexpr bool is_f64 = std::is_floating_point_v<T>;
        static constexpr bool is_string = IsStringLike<T>;
        static_assert(is_bool || is_i64 || is_u64 || is_f64 || is_string);
        if constexpr (is_bool) {
            return Bool(std::forward<T>(value));
        } else if constexpr (is_i64) {
            return I64(static_cast<json::I64>(std::forward<T>(value)));
        } else if constexpr (is_u64) {
            return U64(static_cast<json::U64>(std::forward<T>(value)));
        } else if constexpr (is_f64) {
            return F64(static_cast<json::F64>(std::forward<T>(value)));
        } else if constexpr (is_string) {
            return String(std::forward<T>(value));
        }
    }
};
}  
#endif  