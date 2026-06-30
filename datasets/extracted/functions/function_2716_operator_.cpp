#ifndef LANGSVR_UTILS_RESULT_RESULT_H_
#define LANGSVR_UTILS_RESULT_RESULT_H_
#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include "langsvr/traits.h"
namespace langsvr {
struct SuccessType {};
static constexpr const SuccessType Success;
struct Failure {
    std::string reason;
};
static inline std::ostream& operator<<(std::ostream& out, const Failure& failure) {
    return out << failure.reason;
}
template <typename SUCCESS_TYPE, typename FAILURE_TYPE = Failure>
struct [[nodiscard]] Result {
    using ResultSuccess = SUCCESS_TYPE;
    using ResultFailure = FAILURE_TYPE;
    static_assert(!std::is_same_v<SUCCESS_TYPE, FAILURE_TYPE>,
                  "Result must not have the same type for SUCCESS_TYPE and FAILURE_TYPE");
    Result() : value(std::monostate{}) {}
    Result(const SUCCESS_TYPE& success) : value{success} {}
    Result(SUCCESS_TYPE&& success) : value(std::move(SUCCESS_TYPE(std::move(success)))) {}
    Result(const FAILURE_TYPE& failure) : value{failure} {}
    Result(FAILURE_TYPE&& failure) : value{std::move(failure)} {}
    template <typename S,
              typename F,
              typename = std::void_t<decltype(SUCCESS_TYPE{std::declval<S>()}),
                                     decltype(FAILURE_TYPE{std::declval<F>()})>>
    Result(const Result<S, F>& other) {
        if (other == Success) {
            value = SUCCESS_TYPE{other.Get()};
        } else {
            value = FAILURE_TYPE{other.Failure()};
        }
    }
    const SUCCESS_TYPE* operator->() const {
        Validate();
        return &(Get());
    }
    SUCCESS_TYPE* operator->() {
        Validate();
        return &(Get());
    }
    const SUCCESS_TYPE& Get() const {
        Validate();
        return std::get<SUCCESS_TYPE>(value);
    }
    SUCCESS_TYPE& Get() {
        Validate();
        return std::get<SUCCESS_TYPE>(value);
    }
    SUCCESS_TYPE&& Move() {
        Validate();
        return std::get<SUCCESS_TYPE>(std::move(value));
    }
    const FAILURE_TYPE& Failure() const {
        Validate();
        return std::get<FAILURE_TYPE>(value);
    }
    template <typename T>
    bool operator==(const Result& other) const {
        return value == other.value;
    }
    template <typename T>
    bool operator==(const T& val) const {
        Validate();
        using D = std::decay_t<T>;
        static constexpr bool is_success = std::is_same_v<D, SuccessType>;  
        static constexpr bool is_success_ty =
            std::is_same_v<D, SUCCESS_TYPE> ||
            (IsStringLike<SUCCESS_TYPE> && IsStringLike<D>);  
        static constexpr bool is_failure_ty =
            std::is_same_v<D, FAILURE_TYPE> ||
            (IsStringLike<FAILURE_TYPE> && IsStringLike<D>);  
        static_assert(is_success || is_success_ty || is_failure_ty,
                      "unsupported type for Result equality operator");
        static_assert(!(is_success_ty && is_failure_ty),
                      "ambiguous success / failure type for Result equality operator");
        if constexpr (is_success) {
            return std::holds_alternative<SUCCESS_TYPE>(value);
        } else if constexpr (is_success_ty) {
            if (auto* v = std::get_if<SUCCESS_TYPE>(&value)) {
                return *v == val;
            }
            return false;
        } else if constexpr (is_failure_ty) {
            if (auto* v = std::get_if<FAILURE_TYPE>(&value)) {
                return *v == val;
            }
            return false;
        }
    }
    template <typename T>
    bool operator!=(const T& val) const {
        return !(*this == val);
    }
  private:
    void Validate() const { assert(!std::holds_alternative<std::monostate>(value)); }
    std::variant<std::monostate, SUCCESS_TYPE, FAILURE_TYPE> value;
};
template <typename SUCCESS, typename FAILURE>
static inline std::ostream& operator<<(std::ostream& out, const Result<SUCCESS, FAILURE>& res) {
    if (res == Success) {
        if constexpr (HasOperatorShiftLeft<std::ostream&, SUCCESS>) {
            return out << "success: " << res.Get();
        } else {
            return out << "success";
        }
    } else {
        if constexpr (HasOperatorShiftLeft<std::ostream&, FAILURE>) {
            return out << "failure: " << res.Failure();
        } else {
            return out << "failure";
        }
    }
}
}  
#endif  