#ifndef TENSORSTORE_INTERNAL_UNIQUE_WITH_INTRUSIVE_ALLOCATOR_H_
#define TENSORSTORE_INTERNAL_UNIQUE_WITH_INTRUSIVE_ALLOCATOR_H_
#include <memory>
#include <new>
#include <utility>
namespace tensorstore {
namespace internal {
template <typename T>
struct IntrusiveAllocatorDeleter {
  void operator()(T* p) {
    auto allocator = p->get_allocator();
    typename std::allocator_traits<decltype(
        allocator)>::template rebind_alloc<T>
        rebound_allocator(std::move(allocator));
    std::allocator_traits<decltype(rebound_allocator)>::destroy(
        rebound_allocator, p);
    std::allocator_traits<decltype(rebound_allocator)>::deallocate(
        rebound_allocator, p, 1);
  }
};
template <typename T, typename Allocator, typename... Arg>
std::unique_ptr<T, IntrusiveAllocatorDeleter<T>>
MakeUniqueWithIntrusiveAllocator(Allocator allocator, Arg&&... arg) {
  using ReboundAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
  ReboundAllocator rebound_allocator(std::move(allocator));
  auto temp_deleter = [&rebound_allocator](T* p) {
    std::allocator_traits<ReboundAllocator>::deallocate(rebound_allocator, p,
                                                        1);
  };
  std::unique_ptr<T, decltype(temp_deleter)> temp_ptr(
      std::allocator_traits<ReboundAllocator>::allocate(rebound_allocator, 1),
      temp_deleter);
  new (temp_ptr.get())
      T(std::forward<Arg>(arg)..., std::move(rebound_allocator));
  return std::unique_ptr<T, IntrusiveAllocatorDeleter<T>>(temp_ptr.release());
}
struct VirtualDestroyDeleter {
  template <typename T>
  void operator()(T* p) const {
    p->Destroy();
  }
};
template <typename Derived, typename IntrusiveBase>
class IntrusiveAllocatorBase : public IntrusiveBase {
 public:
  using IntrusiveBase::IntrusiveBase;
  void Destroy() override {
    IntrusiveAllocatorDeleter<Derived>()(static_cast<Derived*>(this));
  }
};
template <typename T, typename Allocator, typename... Arg>
std::unique_ptr<T, VirtualDestroyDeleter>
MakeUniqueWithVirtualIntrusiveAllocator(Allocator allocator, Arg&&... arg) {
  return std::unique_ptr<T, VirtualDestroyDeleter>(
      MakeUniqueWithIntrusiveAllocator<T>(std::move(allocator),
                                          std::forward<Arg>(arg)...)
          .release());
}
}  
}  
#endif  