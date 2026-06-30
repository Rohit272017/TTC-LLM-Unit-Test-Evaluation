#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_JSON_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_JSON_H_
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_json_binding {
namespace empty_binder {
constexpr inline auto EmptyBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) -> absl::Status {
  return absl::OkStatus();
};
}
using empty_binder::EmptyBinder;
namespace loose_value_as_binder {
constexpr inline auto LooseValueAsBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(*j, obj, false);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  
using loose_value_as_binder::LooseValueAsBinder;
namespace value_as_binder {
constexpr inline auto ValueAsBinder = [](auto is_loading, const auto& options,
                                         auto* obj,
                                         ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(*j, obj, true);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  
using value_as_binder::ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<bool> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::int64_t> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::string> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<uint64_t> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<double> = ValueAsBinder;
template <>
constexpr inline auto DefaultBinder<std::nullptr_t> = ValueAsBinder;
namespace loose_float_binder {
constexpr inline auto LooseFloatBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    double x;
    auto status = internal_json::JsonRequireValueAs(*j, &x, false);
    if (status.ok()) *obj = x;
    return status;
  } else {
    *j = static_cast<double>(*obj);
    return absl::OkStatus();
  }
};
}  
using loose_float_binder::LooseFloatBinder;
namespace float_binder {
constexpr inline auto FloatBinder = [](auto is_loading, const auto& options,
                                       auto* obj,
                                       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    double x;
    auto status = internal_json::JsonRequireValueAs(*j, &x, true);
    if (status.ok()) *obj = x;
    return status;
  } else {
    *j = static_cast<double>(*obj);
    return absl::OkStatus();
  }
};
}  
using float_binder::FloatBinder;
template <typename T>
constexpr inline auto
    DefaultBinder<T, std::enable_if_t<std::is_floating_point_v<T>>> =
        FloatBinder;
template <typename T>
constexpr auto LooseInteger(T min = std::numeric_limits<T>::min(),
                            T max = std::numeric_limits<T>::max()) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      return internal_json::JsonRequireInteger(*j, obj, false, min,
                                               max);
    } else {
      *j = *obj;
      return absl::OkStatus();
    }
  };
}
template <typename T>
constexpr auto Integer(T min = std::numeric_limits<T>::min(),
                       T max = std::numeric_limits<T>::max()) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      return internal_json::JsonRequireInteger(*j, obj, true, min,
                                               max);
    } else {
      *j = *obj;
      return absl::OkStatus();
    }
  };
}
template <typename T>
constexpr inline auto
    DefaultBinder<T, std::enable_if_t<std::numeric_limits<T>::is_integer>> =
        Integer<T>();
namespace non_empty_string_binder {
constexpr inline auto NonEmptyStringBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    return internal_json::JsonRequireValueAs(
        *j, obj, [](const std::string& value) { return !value.empty(); },
        true);
  } else {
    *j = *obj;
    return absl::OkStatus();
  }
};
}  
using non_empty_string_binder::NonEmptyStringBinder;
namespace copy_binder {
constexpr inline auto CopyJsonBinder = [](auto is_loading, const auto& options,
                                          auto* obj,
                                          ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    *obj = std::move(*j);
  } else {
    *j = *obj;
  }
  return absl::OkStatus();
};
}  
using copy_binder::CopyJsonBinder;
template <>
constexpr inline auto DefaultBinder<::nlohmann::json> = CopyJsonBinder;
namespace object_binder {
constexpr inline auto CopyJsonObjectBinder = [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) -> absl::Status {
  if constexpr (is_loading) {
    if constexpr (std::is_same_v<decltype(j), ::nlohmann::json::object_t*>) {
      *obj = std::move(*j);
    } else {
      if (auto* j_obj = j->template get_ptr<::nlohmann::json::object_t*>()) {
        *obj = std::move(*j_obj);
      } else {
        return internal_json::ExpectedError(*j, "object");
      }
    }
  } else {
    *j = *obj;
  }
  return absl::OkStatus();
};
}  
using object_binder::CopyJsonObjectBinder;
template <>
constexpr inline auto DefaultBinder<::nlohmann::json::object_t> =
    CopyJsonObjectBinder;
template <typename GetValue>
constexpr auto Constant(GetValue get_value) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      const auto& value = get_value();
      if (!internal_json::JsonSame(*j, value)) {
        return internal_json::ExpectedError(*j, ::nlohmann::json(value).dump());
      }
    } else {
      *j = get_value();
    }
    return absl::OkStatus();
  };
}
template <typename Validator, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Validate(Validator validator, Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
      return internal::InvokeForStatus(validator, options, obj);
    } else {
      return binder(is_loading, options, obj, j);
    }
  };
}
template <typename Initializer>
constexpr auto Initialize(Initializer initializer) {
  return [=](auto is_loading, const auto& options, [[maybe_unused]] auto* obj,
             auto*) -> absl::Status {
    if constexpr (is_loading) {
      return internal::InvokeForStatus(initializer, obj);
    } else {
      return absl::OkStatus();
    }
  };
}
template <auto Proj, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Projection(Binder binder = DefaultBinder<>) {
  return [binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    auto&& projected = std::invoke(Proj, *obj);
    return binder(is_loading, options, &projected, j);
  };
}
template <typename Proj, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Projection(Proj projection, Binder binder = DefaultBinder<>) {
  return [projection = std::move(projection), binder = std::move(binder)](
             auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    auto&& projected = std::invoke(projection, *obj);
    return binder(is_loading, options, &projected, j);
  };
}
template <typename T = void, typename Get, typename Set,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto GetterSetter(Get get, Set set, Binder binder = DefaultBinder<>) {
  return [get = std::move(get), set = std::move(set),
          binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    if constexpr (is_loading) {
      using Projected = std::conditional_t<
          std::is_void_v<T>,
          absl::remove_cvref_t<std::invoke_result_t<Get, decltype(*obj)>>, T>;
      Projected projected;
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, &projected, j));
      return internal::InvokeForStatus(set, *obj, std::move(projected));
    } else {
      auto&& projected = std::invoke(get, *obj);
      return binder(is_loading, options, &projected, j);
    }
  };
}
template <typename LoadBinder = decltype(EmptyBinder),
          typename SaveBinder = decltype(EmptyBinder)>
constexpr auto LoadSave(LoadBinder load_binder = EmptyBinder,
                        SaveBinder save_binder = EmptyBinder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      return load_binder(is_loading, options, obj, j);
    } else {
      return save_binder(is_loading, options, obj, j);
    }
  };
}
enum IncludeDefaultsPolicy {
  kMaybeIncludeDefaults,
  kNeverIncludeDefaults,
  kAlwaysIncludeDefaults,
};
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename GetDefault, typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultValue(GetDefault get_default,
                            Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(obj)>>;
    if constexpr (is_loading) {
      if (j->is_discarded()) {
        return internal::InvokeForStatus(get_default, obj);
      }
      return binder(is_loading, options, obj, j);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(binder(is_loading, options, obj, j));
      if constexpr (Policy == kAlwaysIncludeDefaults) {
        return absl::OkStatus();
      }
      if constexpr (Policy == kMaybeIncludeDefaults) {
        IncludeDefaults include_defaults(options);
        if (include_defaults.include_defaults()) {
          return absl::OkStatus();
        }
      }
      T default_obj;
      ::nlohmann::json default_j;
      if (internal::InvokeForStatus(get_default, &default_obj).ok() &&
          binder(is_loading, options, &default_obj, &default_j).ok() &&
          internal_json::JsonSame(default_j, *j)) {
        *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
      }
      return absl::OkStatus();
    }
  };
}
template <IncludeDefaultsPolicy DefaultsPolicy = kMaybeIncludeDefaults,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultInitializedValue(Binder binder = DefaultBinder<>) {
  return internal_json_binding::DefaultValue<DefaultsPolicy>(
      [](auto* obj) { *obj = absl::remove_cvref_t<decltype(*obj)>{}; },
      std::move(binder));
}
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename GetDefault, typename IsDefault,
          typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultPredicate(GetDefault get_default, IsDefault is_default,
                                Binder binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      if (j->is_discarded()) {
        return internal::InvokeForStatus(get_default, obj);
      }
      return binder(is_loading, options, obj, j);
    } else {
      bool include_defaults_value = Policy == kAlwaysIncludeDefaults;
      if constexpr (Policy == kMaybeIncludeDefaults) {
        IncludeDefaults include_defaults(options);
        include_defaults_value = include_defaults.include_defaults();
      }
      if (!include_defaults_value && is_default(obj)) {
        *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
        return absl::OkStatus();
      }
      return binder(is_loading, options, obj, j);
    }
  };
}
template <IncludeDefaultsPolicy Policy = kMaybeIncludeDefaults,
          typename IsDefault, typename Binder = decltype(DefaultBinder<>)>
constexpr auto DefaultInitializedPredicate(IsDefault is_default,
                                           Binder binder = DefaultBinder<>) {
  return internal_json_binding::DefaultPredicate<Policy>(
      [](auto* obj) { *obj = absl::remove_cvref_t<decltype(*obj)>{}; },
      std::move(is_default), std::move(binder));
}
template <typename T, typename TransformedValueBinder,
          typename OriginalValueBinder = decltype(DefaultBinder<>)>
constexpr auto Compose(
    TransformedValueBinder transformed_value_binder,
    OriginalValueBinder original_value_binder = DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    T value;
    if constexpr (is_loading) {
      TENSORSTORE_RETURN_IF_ERROR(
          original_value_binder(is_loading, options, &value, j));
      return transformed_value_binder(is_loading, options, obj, &value);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(
          transformed_value_binder(is_loading, options, obj, &value));
      return original_value_binder(is_loading, options, &value, j);
    }
  };
}
template <typename GetBinder>
constexpr auto Dependent(GetBinder get_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto*... j) -> absl::Status {
    return get_binder(is_loading, options, obj, j...)(is_loading, options, obj,
                                                      j...);
  };
}
namespace sequence_impl {
template <typename Loading, typename Options, typename Obj, typename J,
          typename... Binder>
inline absl::Status invoke_reverse(Loading is_loading, Options& options,
                                   Obj* obj, J* j, Binder... binder) {
  absl::Status s;
  std::true_type right_to_left;
  right_to_left =
      (((s.ok() ? (void)(s = binder(is_loading, options, obj, j)) : (void)0),
        right_to_left) = ... = right_to_left);
  return s;
}
template <typename Loading, typename Options, typename Obj, typename J,
          typename... Binder>
inline absl::Status invoke_forward(Loading is_loading, Options& options,
                                   Obj* obj, J* j, Binder... binder) {
  absl::Status s;
  [[maybe_unused]] bool ok =
      (((s = binder(is_loading, options, obj, j)).ok()) && ...);
  return s;
}
}  
template <typename... Binder>
constexpr auto Sequence(Binder... binder) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    if constexpr (is_loading) {
      return sequence_impl::invoke_forward(is_loading, options, obj, j,
                                           binder...);
    } else {
      return sequence_impl::invoke_reverse(is_loading, options, obj, j,
                                           binder...);
    }
  };
}
template <typename... MemberBinder>
constexpr auto Object(MemberBinder... member_binder) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    ::nlohmann::json::object_t* j_obj;
    if constexpr (is_loading) {
      if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
        j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
        if (!j_obj) {
          return internal_json::ExpectedError(*j, "object");
        }
      } else {
        j_obj = j;
      }
      TENSORSTORE_RETURN_IF_ERROR(sequence_impl::invoke_forward(
          is_loading, options, obj, j_obj, member_binder...));
      if (!j_obj->empty()) {
        return internal_json::JsonExtraMembersError(*j_obj);
      }
      return absl::OkStatus();
    } else {
      if constexpr (std::is_same_v<::nlohmann::json*, decltype(j)>) {
        *j = ::nlohmann::json::object_t();
        j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
      } else {
        j_obj = j;
        j_obj->clear();
      }
      return sequence_impl::invoke_reverse(is_loading, options, obj, j_obj,
                                           member_binder...);
    }
  };
}
template <bool kDropDiscarded, typename MemberName, typename Binder>
struct MemberBinderImpl {
  MemberName name;
  Binder binder;
  template <typename Options, typename Obj>
  absl::Status operator()(std::true_type is_loading, const Options& options,
                          Obj* obj, ::nlohmann::json::object_t* j_obj) const {
    ::nlohmann::json j_member = internal_json::JsonExtractMember(j_obj, name);
    if constexpr (kDropDiscarded) {
      if (j_member.is_discarded()) return absl::OkStatus();
    }
    auto status = binder(is_loading, options, obj, &j_member);
    return status.ok()
               ? status
               : MaybeAnnotateStatus(
                     status, tensorstore::StrCat("Error parsing object member ",
                                                 QuoteString(name)));
  }
  template <typename Options, typename Obj>
  absl::Status operator()(std::false_type is_loading, const Options& options,
                          Obj* obj, ::nlohmann::json::object_t* j_obj) const {
    ::nlohmann::json j_member(::nlohmann::json::value_t::discarded);
    TENSORSTORE_RETURN_IF_ERROR(
        binder(is_loading, options, obj, &j_member),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Error converting object member ",
                                   QuoteString(name))));
    if (!j_member.is_discarded()) {
      j_obj->emplace(name, std::move(j_member));
    }
    return absl::OkStatus();
  }
};
template <typename MemberName, typename Binder = decltype(DefaultBinder<>)>
constexpr auto Member(MemberName name, Binder binder = DefaultBinder<>) {
  return MemberBinderImpl<false, MemberName, Binder>{std::move(name),
                                                     std::move(binder)};
}
template <typename MemberName, typename Binder = decltype(DefaultBinder<>)>
constexpr auto OptionalMember(MemberName name,
                              Binder binder = DefaultBinder<>) {
  return MemberBinderImpl<true, MemberName, Binder>{std::move(name),
                                                    std::move(binder)};
}
template <typename... MemberName>
constexpr auto AtMostOne(MemberName... names) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    if constexpr (is_loading) {
      const auto has_member = [&](auto name) {
        return j->find(name) == j->end() ? 0 : 1;
      };
      if ((has_member(names) + ...) > 1) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "At most one of ",
            absl::StrJoin({QuoteString(std::string_view(names))...}, ", "),
            " members is allowed"));
      }
    }
    return absl::OkStatus();
  };
}
template <typename... MemberName>
constexpr auto AtLeastOne(MemberName... names) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    if constexpr (is_loading) {
      const auto has_member = [&](auto name) {
        return j->find(name) == j->end() ? 0 : 1;
      };
      if ((has_member(names) + ...) == 0) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "At least one of ",
            absl::StrJoin(
                std::make_tuple(QuoteString(std::string_view(names))...), ", "),
            " members must be specified"));
      }
    }
    return absl::OkStatus();
  };
}
namespace discard_extra_members_binder {
constexpr inline auto DiscardExtraMembers =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json::object_t* j_obj) -> absl::Status {
  if constexpr (is_loading) {
    j_obj->clear();
  }
  return absl::OkStatus();
};
}  
using discard_extra_members_binder::DiscardExtraMembers;
}  
}  
#endif  