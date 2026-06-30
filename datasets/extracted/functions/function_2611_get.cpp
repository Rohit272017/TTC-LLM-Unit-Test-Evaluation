#ifndef TENSORFLOW_CORE_LIB_GTL_MANUAL_CONSTRUCTOR_H_
#define TENSORFLOW_CORE_LIB_GTL_MANUAL_CONSTRUCTOR_H_
#include <stddef.h>
#include <new>
#include <utility>
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
namespace tensorflow {
namespace gtl {
namespace internal {
#ifndef SWIG
template <int alignment, int size>
struct AlignType {};
template <int size>
struct AlignType<0, size> {
  typedef char result[size];
};
#if defined(_MSC_VER)
#define TF_LIB_GTL_ALIGN_ATTRIBUTE(X) __declspec(align(X))
#define TF_LIB_GTL_ALIGN_OF(T) __alignof(T)
#else
#define TF_LIB_GTL_ALIGN_ATTRIBUTE(X) __attribute__((aligned(X)))
#define TF_LIB_GTL_ALIGN_OF(T) __alignof__(T)
#endif
#if defined(TF_LIB_GTL_ALIGN_ATTRIBUTE)
#define TF_LIB_GTL_ALIGNTYPE_TEMPLATE(X)                     \
  template <int size>                                        \
  struct AlignType<X, size> {                                \
    typedef TF_LIB_GTL_ALIGN_ATTRIBUTE(X) char result[size]; \
  }
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(1);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(2);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(4);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(8);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(16);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(32);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(64);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(128);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(256);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(512);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(1024);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(2048);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(4096);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(8192);
#define TF_LIB_GTL_ALIGNED_CHAR_ARRAY(T, Size)                          \
  typename tensorflow::gtl::internal::AlignType<TF_LIB_GTL_ALIGN_OF(T), \
                                                sizeof(T) * Size>::result
#undef TF_LIB_GTL_ALIGNTYPE_TEMPLATE
#undef TF_LIB_GTL_ALIGN_ATTRIBUTE
#else  
#error "You must define TF_LIB_GTL_ALIGNED_CHAR_ARRAY for your compiler."
#endif  
#else  
template <typename Size>
struct AlignType {
  typedef char result[Size];
};
#define TF_LIB_GTL_ALIGNED_CHAR_ARRAY(T, Size) \
  tensorflow::gtl::internal::AlignType<Size * sizeof(T)>::result
#define TF_LIB_GTL_ALIGN_OF(Type) 16
#endif  
}  
}  
template <typename Type>
class ManualConstructor {
 public:
  static void* operator new[](size_t size) {
    return port::AlignedMalloc(size, TF_LIB_GTL_ALIGN_OF(Type));
  }
  static void operator delete[](void* mem) { port::AlignedFree(mem); }
  inline Type* get() { return reinterpret_cast<Type*>(space_); }
  inline const Type* get() const {
    return reinterpret_cast<const Type*>(space_);
  }
  inline Type* operator->() { return get(); }
  inline const Type* operator->() const { return get(); }
  inline Type& operator*() { return *get(); }
  inline const Type& operator*() const { return *get(); }
  inline void Init() { new (space_) Type; }
#ifdef LANG_CXX11
  template <typename... Ts>
  inline void Init(Ts&&... args) {                 
    new (space_) Type(std::forward<Ts>(args)...);  
  }
#else   
  template <typename T1>
  inline void Init(const T1& p1) {
    new (space_) Type(p1);
  }
  template <typename T1, typename T2>
  inline void Init(const T1& p1, const T2& p2) {
    new (space_) Type(p1, p2);
  }
  template <typename T1, typename T2, typename T3>
  inline void Init(const T1& p1, const T2& p2, const T3& p3) {
    new (space_) Type(p1, p2, p3);
  }
  template <typename T1, typename T2, typename T3, typename T4>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4) {
    new (space_) Type(p1, p2, p3, p4);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5) {
    new (space_) Type(p1, p2, p3, p4, p5);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6) {
    new (space_) Type(p1, p2, p3, p4, p5, p6);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9, typename T10>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9, const T10& p10) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
  }
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9, typename T10,
            typename T11>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9, const T10& p10, const T11& p11) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
  }
#endif  
  inline void Destroy() { get()->~Type(); }
 private:
  TF_LIB_GTL_ALIGNED_CHAR_ARRAY(Type, 1) space_;
};
#undef TF_LIB_GTL_ALIGNED_CHAR_ARRAY
#undef TF_LIB_GTL_ALIGN_OF
}  
#endif  