#ifndef AROLLA_QEXPR_LIFTING_H_
#define AROLLA_QEXPR_LIFTING_H_
#include <cstdint>
#include <tuple>
#include <type_traits>
#include "absl/base/attributes.h"
#include "arolla/util/meta.h"
namespace arolla {
template <class T>
struct DoNotLiftTag {
  using type = T;
};
template <class T>
using DecayDoNotLiftTag = meta::strip_template_t<DoNotLiftTag, T>;
template <template <class> class Lifted, class T>
using LiftedType = std::conditional_t<meta::is_wrapped_with_v<DoNotLiftTag, T>,
                                      DecayDoNotLiftTag<T>, Lifted<T>>;
namespace lifting_internal {
template <class ArgTypeList>
struct CallOnLiftedArgsImpl;
template <>
struct CallOnLiftedArgsImpl<meta::type_list<>> {
  template <class Fn, class... Ts>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(Fn&& fn, Ts&&... args) const {
    return std::forward<Fn>(fn)(std::forward<Ts>(args)...);
  }
};
template <class LeftArg, class... LeftArgs>
struct CallOnLiftedArgsImpl<meta::type_list<LeftArg, LeftArgs...>> {
  template <class Fn, class T, class... Ts>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(Fn&& fn, T&& arg,
                                               Ts&&... args) const {
    if constexpr (meta::is_wrapped_with_v<DoNotLiftTag, LeftArg>) {
      return CallOnLiftedArgsImpl<meta::type_list<LeftArgs...>>{}(
          std::forward<Fn>(fn), std::forward<Ts>(args)...);
    } else {
      return CallOnLiftedArgsImpl<meta::type_list<LeftArgs...>>{}(
          std::forward<Fn>(fn), std::forward<Ts>(args)...,
          std::forward<T>(arg));
    }
  }
};
template <uint64_t kDontLiftMask, class ScalarArgsList, class LiftedArgsList,
          class MergedArgsList>
struct CallShuffledArgsFn;
template <class... LiftedArgs, class... MergedArgs>
struct CallShuffledArgsFn<0, meta::type_list<>, meta::type_list<LiftedArgs...>,
                          meta::type_list<MergedArgs...>> {
  template <class SctrictFn>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(
      const SctrictFn& fn, const LiftedArgs&... lifted_args,
      const MergedArgs&... merged_args) const {
    return fn(merged_args..., lifted_args...);
  }
};
template <uint64_t kDontLiftMask, class... ScalarArgs, class... MergedArgs>
struct CallShuffledArgsFn<kDontLiftMask, meta::type_list<ScalarArgs...>,
                          meta::type_list<>, meta::type_list<MergedArgs...>> {
  static_assert(kDontLiftMask == (1ull << sizeof...(ScalarArgs)) - 1);
  template <class SctrictFn>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(
      const SctrictFn& fn, const ScalarArgs&... scalar_args,
      const MergedArgs&... merged_args) const {
    return fn(merged_args..., scalar_args...);
  }
};
template <class... MergedArgs>
struct CallShuffledArgsFn<0, meta::type_list<>, meta::type_list<>,
                          meta::type_list<MergedArgs...>> {
  template <class SctrictFn>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(
      const SctrictFn& fn, const MergedArgs&... merged_args) const {
    return fn(merged_args...);
  }
};
template <uint64_t kDontLiftMask, class ScalarArg, class... ScalarArgs,
          class LiftedArg, class... LiftedArgs, class... MergedArgs>
struct CallShuffledArgsFn<
    kDontLiftMask, meta::type_list<ScalarArg, ScalarArgs...>,
    meta::type_list<LiftedArg, LiftedArgs...>, meta::type_list<MergedArgs...>> {
  template <class SctrictFn>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(
      const SctrictFn& fn, const ScalarArg& scalar_arg,
      const ScalarArgs&... scalar_args, const LiftedArg& lifted_arg,
      const LiftedArgs&... lifted_args, MergedArgs... merged_args) const {
    if constexpr (kDontLiftMask % 2 == 1) {
      return CallShuffledArgsFn<kDontLiftMask / 2,
                                meta::type_list<ScalarArgs...>,
                                meta::type_list<LiftedArg, LiftedArgs...>,
                                meta::type_list<MergedArgs..., ScalarArg>>()(
          fn, scalar_args..., lifted_arg, lifted_args..., merged_args...,
          scalar_arg);
    } else {
      return CallShuffledArgsFn<kDontLiftMask / 2,
                                meta::type_list<ScalarArg, ScalarArgs...>,
                                meta::type_list<LiftedArgs...>,
                                meta::type_list<MergedArgs..., LiftedArg>>()(
          fn, scalar_arg, scalar_args..., lifted_args..., merged_args...,
          lifted_arg);
    }
  }
};
template <template <typename> class LiftedViewType, class ArgsToProcessList,
          class LiftedArgList, uint64_t kDontLiftMask>
struct CaptureDontLift;
template <template <typename> class LiftedViewType, uint64_t kDontLiftMask,
          class LeftArg, class... LeftArgs, class... LiftedArgs>
struct CaptureDontLift<LiftedViewType, meta::type_list<LeftArg, LeftArgs...>,
                       meta::type_list<LiftedArgs...>, kDontLiftMask> {
  template <class Fn, class T, class... Ts>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(const Fn& fn, const T& arg,
                                               const Ts&... args) const {
    if constexpr (meta::is_wrapped_with_v<DoNotLiftTag, LeftArg>) {
      constexpr uint64_t total_arg_count =
          1 + sizeof...(Ts) + sizeof...(LiftedArgs);
      constexpr uint64_t arg_id = total_arg_count - (sizeof...(LeftArgs) + 1);
      return CaptureDontLift<LiftedViewType, meta::type_list<LeftArgs...>,
                             meta::type_list<LiftedArgs...>,
                             kDontLiftMask + (1ull << arg_id)>{}(fn, args...,
                                                                 arg);
    } else {
      return CaptureDontLift<LiftedViewType, meta::type_list<LeftArgs...>,
                             meta::type_list<LiftedArgs..., LeftArg>,
                             kDontLiftMask>{}(fn, args...);
    }
  }
};
template <template <typename> class LiftedViewType, uint64_t kDontLiftMask,
          class... LiftedArgs>
struct CaptureDontLift<LiftedViewType, meta::type_list<>,
                       meta::type_list<LiftedArgs...>, kDontLiftMask> {
  template <class Fn, class... Ts>
  ABSL_ATTRIBUTE_ALWAYS_INLINE auto operator()(const Fn& fn,
                                               const Ts&... args) const {
    return [fn, &args...](LiftedViewType<LiftedArgs>... view_args)
               ABSL_ATTRIBUTE_ALWAYS_INLINE {
                 return CallShuffledArgsFn<
                     kDontLiftMask, meta::type_list<Ts...>,
                     meta::type_list<LiftedViewType<LiftedArgs>...>,
                     meta::type_list<>>()(fn, args..., view_args...);
               };
  }
};
template <class ArgList>
struct LiftableArgs;
template <>
struct LiftableArgs<meta::type_list<>> {
  using type = meta::type_list<>;
};
template <class T, class... Ts>
struct LiftableArgs<meta::type_list<T, Ts...>> {
  using type =
      meta::concat_t<meta::type_list<T>,
                     typename LiftableArgs<meta::type_list<Ts...>>::type>;
};
template <class T, class... Ts>
struct LiftableArgs<meta::type_list<DoNotLiftTag<T>, Ts...>> {
  using type = typename LiftableArgs<meta::type_list<Ts...>>::type;
};
}  
template <class... Args>
class LiftingTools {
  static_assert(sizeof...(Args) <= 64, "Arg count limit is 64");
 public:
  using LiftableArgs =
      typename lifting_internal::LiftableArgs<meta::type_list<Args...>>::type;
  static constexpr bool kAllLiftable =
      std::tuple_size_v<typename LiftableArgs::tuple> == sizeof...(Args);
  template <template <typename> class LiftedViewType, class Fn, class... Ts>
  static auto CreateFnWithDontLiftCaptured(const Fn& fn, const Ts&... args) {
    static_assert(sizeof...(Args) == sizeof...(Ts));
    if constexpr (kAllLiftable) {
      return [fn](LiftedViewType<Args>... largs) { return fn(largs...); };
    } else {
      return lifting_internal::CaptureDontLift<
          LiftedViewType, meta::type_list<Args...>, meta::type_list<>, 0>{}(
          fn, args...);
    }
  }
  template <class Fn, class... Ts>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static auto CallOnLiftedArgs(Fn&& fn,
                                                            Ts&&... args) {
    static_assert(sizeof...(Args) == sizeof...(Ts));
    if constexpr (kAllLiftable) {
      return std::forward<Fn>(fn)(std::forward<Ts>(args)...);
    } else {
      return lifting_internal::CallOnLiftedArgsImpl<meta::type_list<Args...>>{}(
          std::forward<Fn>(fn), std::forward<Ts>(args)...);
    }
  }
};
}  
#endif  