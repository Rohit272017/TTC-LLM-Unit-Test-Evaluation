#ifndef TENSORSTORE_INTERNAL_STRING_TO_SHARED_H_
#define TENSORSTORE_INTERNAL_STRING_TO_SHARED_H_
#include <stddef.h>
#include <memory>
#include <utility>
namespace tensorstore {
namespace internal {
template <typename Container>
inline std::shared_ptr<typename Container::value_type>
ContainerToSharedDataPointerWithOffset(Container&& container,
                                       size_t offset = 0) {
  auto ptr = std::make_shared<Container>(std::forward<Container>(container));
  return std::shared_ptr<typename Container::value_type>(std::move(ptr),
                                                         ptr->data() + offset);
}
}  
}  
#endif  