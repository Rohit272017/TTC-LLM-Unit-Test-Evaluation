#ifndef TENSORSTORE_INTERNAL_CONTAINER_INTRUSIVE_LINKED_LIST_H_
#define TENSORSTORE_INTERNAL_CONTAINER_INTRUSIVE_LINKED_LIST_H_
namespace tensorstore {
namespace internal {
namespace intrusive_linked_list {
template <typename T, T* T::*PrevMember = &T::prev,
          T* T::*NextMember = &T::next>
struct MemberAccessor {
  using Node = T*;
  static void SetPrev(T* node, T* prev) { node->*PrevMember = prev; }
  static void SetNext(T* node, T* next) { node->*NextMember = next; }
  static T* GetPrev(T* node) { return node->*PrevMember; }
  static T* GetNext(T* node) { return node->*NextMember; }
};
template <typename Accessor>
void Initialize(Accessor accessor, typename Accessor::Node node) {
  accessor.SetPrev(node, node);
  accessor.SetNext(node, node);
}
template <typename Accessor>
void InsertBefore(Accessor accessor, typename Accessor::Node existing_node,
                  typename Accessor::Node new_node) {
  accessor.SetPrev(new_node, accessor.GetPrev(existing_node));
  accessor.SetNext(new_node, existing_node);
  accessor.SetNext(accessor.GetPrev(existing_node), new_node);
  accessor.SetPrev(existing_node, new_node);
}
template <typename Accessor>
void Remove(Accessor accessor, typename Accessor::Node node) {
  accessor.SetPrev(accessor.GetNext(node), accessor.GetPrev(node));
  accessor.SetNext(accessor.GetPrev(node), accessor.GetNext(node));
}
template <typename Accessor>
bool OnlyContainsNode(Accessor accessor, typename Accessor::Node node) {
  return accessor.GetNext(node) == node;
}
}  
}  
}  
#endif  