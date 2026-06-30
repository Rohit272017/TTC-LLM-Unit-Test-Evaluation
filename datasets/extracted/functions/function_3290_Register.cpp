#ifndef TENSORSTORE_INTERNAL_JSON_REGISTRY_H_
#define TENSORSTORE_INTERNAL_JSON_REGISTRY_H_
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <utility>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_registry_fwd.h"
#include "tensorstore/internal/json_registry_impl.h"
#include "tensorstore/json_serialization_options.h"
namespace tensorstore {
namespace internal {
template <typename Base, typename LoadOptions, typename SaveOptions,
          typename BasePtr>
class JsonRegistry {
  static_assert(std::has_virtual_destructor_v<Base>);
 public:
  auto KeyBinder() const { return KeyBinderImpl{impl_}; }
  constexpr auto RegisteredObjectBinder() const {
    return RegisteredObjectBinderImpl{impl_};
  }
  template <typename MemberName>
  auto MemberBinder(MemberName member_name) const {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Sequence(jb::Member(member_name, this->KeyBinder()),
                        RegisteredObjectBinder());
  }
  template <typename T, typename Binder>
  void Register(std::string_view id, Binder binder) {
    static_assert(std::is_base_of_v<Base, T>);
    auto entry =
        std::make_unique<internal_json_registry::JsonRegistryImpl::Entry>();
    entry->id = std::string(id);
    entry->type = &typeid(T);
    entry->allocate =
        +[](void* obj) { static_cast<BasePtr*>(obj)->reset(new T); };
    entry->binder = [binder](
                        auto is_loading, const void* options, const void* obj,
                        ::nlohmann::json::object_t* j_obj) -> absl::Status {
      using Options = std::conditional_t<decltype(is_loading)::value,
                                         LoadOptions, SaveOptions>;
      using Obj = std::conditional_t<decltype(is_loading)::value, T, const T>;
      return binder(is_loading, *static_cast<const Options*>(options),
                    const_cast<Obj*>(
                        static_cast<const Obj*>(static_cast<const Base*>(obj))),
                    j_obj);
    };
    impl_.Register(std::move(entry));
  }
 private:
  struct KeyBinderImpl {
    const internal_json_registry::JsonRegistryImpl& impl;
    template <typename Options>
    absl::Status operator()(std::true_type is_loading, const Options& options,
                            BasePtr* obj, ::nlohmann::json* j) const {
      return impl.LoadKey(obj, j);
    }
    template <typename Ptr, typename Options>
    absl::Status operator()(std::false_type is_loading, const Options& options,
                            const Ptr* obj, ::nlohmann::json* j) const {
      static_assert(std::is_convertible_v<decltype(&**obj), const Base*>);
      return impl.SaveKey(typeid(**obj), j);
    }
  };
  struct RegisteredObjectBinderImpl {
    const internal_json_registry::JsonRegistryImpl& impl;
    absl::Status operator()(std::true_type is_loading,
                            const LoadOptions& options, BasePtr* obj,
                            ::nlohmann::json::object_t* j_obj) const {
      if (!*obj) return absl::OkStatus();
      return impl.LoadRegisteredObject(typeid(*obj->get()), &options,
                                       static_cast<const Base*>(&**obj), j_obj);
    }
    template <typename Ptr>
    absl::Status operator()(std::false_type is_loading,
                            const SaveOptions& options, const Ptr* obj,
                            ::nlohmann::json::object_t* j_obj) const {
      static_assert(std::is_convertible_v<decltype(&**obj), const Base*>);
      if (!*obj) return absl::OkStatus();
      return impl.SaveRegisteredObject(typeid(**obj), &options,
                                       static_cast<const Base*>(&**obj), j_obj);
    }
  };
  internal_json_registry::JsonRegistryImpl impl_;
};
}  
}  
#endif  