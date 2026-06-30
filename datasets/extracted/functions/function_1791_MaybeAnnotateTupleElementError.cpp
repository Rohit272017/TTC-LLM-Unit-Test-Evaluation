#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STD_TUPLE_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STD_TUPLE_H_
#include <stddef.h>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_json_binding {
inline absl::Status MaybeAnnotateTupleElementError(absl::Status status,
                                                   size_t i, bool is_loading) {
  return status.ok()
             ? status
             : MaybeAnnotateStatus(
                   status, tensorstore::StrCat(
                               "Error ", is_loading ? "parsing" : "converting",
                               " value at position ", i));
}
template <bool IsLoading>
Result<::nlohmann::json::array_t*> EnsureJsonTupleRepresentationImpl(
    std::integral_constant<bool, IsLoading> is_loading, ::nlohmann::json* j,
    size_t n) {
  if constexpr (is_loading) {
    auto* array_ptr = j->get_ptr<::nlohmann::json::array_t*>();
    if (!array_ptr) return internal_json::ExpectedError(*j, "array");
    TENSORSTORE_RETURN_IF_ERROR(
        internal_json::JsonValidateArrayLength(array_ptr->size(), n));
    return array_ptr;
  } else {
    *j = ::nlohmann::json::array_t(n);
    return j->get_ptr<::nlohmann::json::array_t*>();
  }
}
template <size_t... Is, typename... ElementBinder>
constexpr auto TupleJsonBinderImpl(std::index_sequence<Is...>,
                                   ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    if (absl::Status status;
        (((status = element_binder(is_loading, options, &std::get<Is>(*obj),
                                   &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}
template <size_t... Is>
constexpr auto TupleDefaultJsonBinderImpl(std::index_sequence<Is...>) {
  return [](auto is_loading, const auto& options, auto* obj,
            ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    using std::get;
    if (absl::Status status;
        (((status = DefaultBinder<>(is_loading, options, &get<Is>(*obj),
                                    &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}
template <size_t... Is, typename... ElementBinder>
constexpr auto HeterogeneousArrayJsonBinderImpl(
    std::index_sequence<Is...>, ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ::nlohmann::json::array_t * array_ptr,
        EnsureJsonTupleRepresentationImpl(is_loading, j, sizeof...(Is)));
    if (absl::Status status;
        (((status = element_binder(is_loading, options, obj, &(*array_ptr)[Is]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, Is, is_loading)),
           false)) &&
         ...)) {
      return status;
    }
    return absl::OkStatus();
  };
}
template <typename... ElementBinder>
constexpr auto Tuple(ElementBinder... element_binder) {
  return TupleJsonBinderImpl(std::index_sequence_for<ElementBinder...>{},
                             std::move(element_binder)...);
}
constexpr auto Tuple() {
  return [](auto is_loading, const auto& options, auto* obj, auto* j) {
    constexpr size_t N =
        std::tuple_size_v<absl::remove_cvref_t<decltype(*obj)>>;
    return TupleDefaultJsonBinderImpl(std::make_index_sequence<N>{})(
        is_loading, options, obj, j);
  };
}
template <typename... ElementBinder>
constexpr auto HeterogeneousArray(ElementBinder... element_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) {
    TENSORSTORE_ASSIGN_OR_RETURN(::nlohmann::json::array_t * array_ptr,
                                 EnsureJsonTupleRepresentationImpl(
                                     is_loading, j, sizeof...(ElementBinder)));
    absl::Status status;
    size_t i = 0;
    [[maybe_unused]] bool ok =
        (((status =
               element_binder(is_loading, options, obj, &(*array_ptr)[i++]))
              .ok() ||
          ((status = MaybeAnnotateTupleElementError(status, i - 1, is_loading)),
           false)) &&
         ...);
    return status;
  };
}
template <typename... T>
constexpr inline auto DefaultBinder<std::tuple<T...>> = Tuple();
template <typename T, typename U>
constexpr inline auto DefaultBinder<std::pair<T, U>> = Tuple();
}  
}  
#endif  