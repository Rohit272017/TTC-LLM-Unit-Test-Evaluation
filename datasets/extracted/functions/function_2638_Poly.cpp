#ifndef TENSORSTORE_INTERNAL_POLY_POLY_H_
#define TENSORSTORE_INTERNAL_POLY_POLY_H_
#include <cstddef>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "absl/meta/type_traits.h"
#include "tensorstore/internal/poly/poly_impl.h"  
namespace tensorstore {
namespace poly {
template <typename T, typename... Signature>
using SupportsPolySignatures =
    std::conjunction<typename internal_poly::SignatureTraits<
        Signature>::template IsSupportedBy<T>...>;
template <size_t InlineSize, bool Copyable, typename... Signature>
class Poly;
template <typename T>
struct IsPoly : public std::false_type {};
template <size_t InlineSize, bool Copyable, typename... Signature>
struct IsPoly<Poly<InlineSize, Copyable, Signature...>>
    : public std::true_type {};
template <typename T, bool Copyable, typename... Signature>
struct IsCompatibleWithPoly : public SupportsPolySignatures<T, Signature...> {};
template <typename T, typename... Signature>
struct IsCompatibleWithPoly<T, true, Signature...>
    : public std::integral_constant<
          bool, (std::is_copy_constructible<T>::value &&
                 SupportsPolySignatures<T, Signature...>::value)> {};
template <size_t InlineSize_, bool Copyable, typename... Signature>
class Poly
    : private internal_poly::PolyImpl<Poly<InlineSize_, Copyable, Signature...>,
                                      Signature...> {
  template <typename, typename...>
  friend class internal_poly::PolyImpl;
  template <size_t, bool, typename...>
  friend class Poly;
  static constexpr size_t InlineSize =
      internal_poly_storage::ActualInlineSize(InlineSize_);
  using Storage = internal_poly_storage::Storage<InlineSize, Copyable>;
  using Base = internal_poly::PolyImpl<Poly, Signature...>;
  using VTable = internal_poly::VTableType<Signature...>;
  template <typename Self>
  using VTInstance =
      internal_poly::VTableInstance<typename Storage::template Ops<Self>,
                                    Copyable, Signature...>;
  template <typename... S>
  using HasConvertibleVTable =
      std::is_convertible<internal_poly::VTableType<S...>, VTable>;
 public:
  template <typename T>
  using IsCompatible =
      std::disjunction<std::is_same<Poly, T>,
                       IsCompatibleWithPoly<T, Copyable, Signature...>>;
  template <typename T>
  using IsCompatibleAndConstructible =
      std::disjunction<
          std::is_same<Poly, absl::remove_cvref_t<T>>,
          std::conjunction<
              IsCompatibleWithPoly<absl::remove_cvref_t<T>, Copyable,
                                   Signature...>,
              std::is_constructible<absl::remove_cvref_t<T>, T&&>>>;
  Poly() = default;
  Poly(std::nullptr_t) noexcept {}
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  Poly(T&& obj) {
    Construct(std::in_place_type_t<absl::remove_cvref_t<T>>{},
              std::forward<T>(obj));
  }
  template <typename T, typename... U,
            std::enable_if_t<(IsCompatible<T>::value &&
                              std::is_constructible_v<T, U&&...>)>* = nullptr>
  Poly(std::in_place_type_t<T> in_place, U&&... arg) {
    Construct(in_place, std::forward<U>(arg)...);
  }
  Poly(const Poly&) = default;
  Poly(Poly&&) = default;
  Poly& operator=(const Poly&) = default;
  Poly& operator=(Poly&&) noexcept = default;
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  Poly& operator=(T&& obj) {
    emplace(std::forward<T>(obj));
    return *this;
  }
  Poly& operator=(std::nullptr_t) noexcept {
    storage_.Destroy();
    return *this;
  }
  template <typename T, typename... U,
            std::enable_if_t<(IsCompatible<T>::value &&
                              std::is_constructible_v<T, U&&...>)>* = nullptr>
  void emplace(U&&... arg) {
    storage_.Destroy();
    Construct(std::in_place_type_t<T>{}, std::forward<U>(arg)...);
  }
  template <typename T,
            std::enable_if_t<IsCompatibleAndConstructible<T>::value>* = nullptr>
  void emplace(T&& obj) {
    storage_.Destroy();
    Construct(std::in_place_type_t<absl::remove_cvref_t<T>>{},
              std::forward<T>(obj));
  }
  using Base::operator();
  explicit operator bool() const { return !storage_.null(); }
  template <typename T>
  T* target() {
    return storage_.template get_if<T>();
  }
  template <typename T>
  const T* target() const {
    return storage_.template get_if<T>();
  }
  friend bool operator==(std::nullptr_t, const Poly& poly) {
    return static_cast<bool>(poly) == false;
  }
  friend bool operator!=(std::nullptr_t, const Poly& poly) {
    return static_cast<bool>(poly) == true;
  }
  friend bool operator==(const Poly& poly, std::nullptr_t) {
    return static_cast<bool>(poly) == false;
  }
  friend bool operator!=(const Poly& poly, std::nullptr_t) {
    return static_cast<bool>(poly) == true;
  }
 private:
  template <typename T, typename... U>
  std::enable_if_t<!IsPoly<T>::value> Construct(std::in_place_type_t<T>,
                                                U&&... arg) {
    return storage_.template ConstructT<T>(&VTInstance<T>::vtable,
                                           static_cast<U&&>(arg)...);
  }
  template <size_t ISize, bool C, typename... S, typename T>
  void Construct(std::in_place_type_t<Poly<ISize, C, S...>>, T&& poly) {
    if constexpr (internal_poly_storage::ActualInlineSize(ISize) <=
                      InlineSize &&
                  HasConvertibleVTable<S...>::value) {
      if constexpr (std::is_lvalue_reference_v<decltype(poly)>) {
        storage_.CopyConstruct(std::forward<T>(poly).storage_);
      } else {
        storage_.Construct(std::forward<T>(poly).storage_);
      }
    } else {
      storage_.template ConstructT<Poly<ISize, C, S...>>(
          &VTInstance<Poly<ISize, C, S...>>::vtable, std::forward<T>(poly));
    }
  }
  Storage storage_;
};
}  
}  
#endif  