#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include <stddef.h>
#include <array>
#include <cassert>
#include <utility>
namespace tensorstore {
namespace internal {
namespace intrusive_red_black_tree {
namespace ops {
inline void SetParent(NodeData* node, NodeData* parent) {
  node->rbtree_parent_ = {parent, node->rbtree_parent_.tag()};
}
inline void SetColor(NodeData* node, Color color) {
  node->rbtree_parent_.set_tag(color);
}
inline Direction ChildDir(NodeData* node) {
  return static_cast<Direction>(node != ops::Child(ops::Parent(node), kLeft));
}
inline NodeData* Grandparent(NodeData* node) {
  return ops::Parent(ops::Parent(node));
}
void Rotate(NodeData*& root, NodeData* x, Direction dir) {
  auto* y = ops::Child(x, !dir);
  ops::Child(x, !dir) = ops::Child(y, dir);
  if (ops::Child(y, dir)) {
    ops::SetParent(ops::Child(y, dir), x);
  }
  ops::SetParent(y, ops::Parent(x));
  if (!ops::Parent(x)) {
    root = y;
  } else {
    ops::Child(ops::Parent(x), ops::ChildDir(x)) = y;
  }
  ops::Child(y, dir) = x;
  ops::SetParent(x, y);
}
bool InsertFixup(NodeData*& root, NodeData* z) {
  assert(ops::IsRed(z));
  while (ops::IsRed(ops::Parent(z))) {
    Direction dir = ops::ChildDir(ops::Parent(z));
    if (NodeData* y = ops::Child(ops::Grandparent(z), !dir); ops::IsRed(y)) {
      ops::SetColor(ops::Parent(z), kBlack);
      ops::SetColor(y, kBlack);
      ops::SetColor(ops::Grandparent(z), kRed);
      z = ops::Grandparent(z);
    } else {
      if (ops::ChildDir(z) == !dir) {
        z = ops::Parent(z);
        ops::Rotate(root, z, dir);
      }
      ops::SetColor(ops::Parent(z), kBlack);
      ops::SetColor(ops::Grandparent(z), kRed);
      ops::Rotate(root, ops::Grandparent(z), !dir);
      assert(!ops::IsRed(ops::Parent(z)));
      break;
    }
  }
  const Color existing_color = ops::GetColor(root);
  ops::SetColor(root, kBlack);
  return existing_color == kRed;
}
struct TreeWithBlackHeight {
  NodeData* root = nullptr;
  size_t black_height = 0;
};
size_t BlackHeight(NodeData* node) {
  size_t black_height = 0;
  while (node) {
    if (ops::GetColor(node) == kBlack) ++black_height;
    node = ops::Child(node, kLeft);
  }
  return black_height;
}
TreeWithBlackHeight Join(TreeWithBlackHeight a_tree, NodeData* center,
                         TreeWithBlackHeight b_tree, Direction a_dir) {
  assert(a_tree.black_height == ops::BlackHeight(a_tree.root));
  assert(b_tree.black_height == ops::BlackHeight(b_tree.root));
  if (a_tree.black_height < b_tree.black_height) {
    a_dir = !a_dir;
    std::swap(a_tree, b_tree);
  }
  size_t difference = a_tree.black_height - b_tree.black_height;
  NodeData* a_graft = a_tree.root;
  NodeData* a_graft_parent = nullptr;
  while (true) {
    if (!ops::IsRed(a_graft)) {
      if (difference == 0) break;
      --difference;
    }
    a_graft_parent = a_graft;
    a_graft = ops::Child(a_graft, !a_dir);
  }
  assert(!ops::IsRed(a_graft));
  ops::SetColor(center, kRed);
  ops::SetParent(center, a_graft_parent);
  if (a_graft_parent) {
    ops::Child(a_graft_parent, !a_dir) = center;
  } else {
    a_tree.root = center;
  }
  ops::Child(center, a_dir) = a_graft;
  if (a_graft) {
    ops::SetParent(a_graft, center);
  }
  ops::Child(center, !a_dir) = b_tree.root;
  if (b_tree.root) {
    ops::SetParent(b_tree.root, center);
  }
  a_tree.black_height += ops::InsertFixup(a_tree.root, center);
  return a_tree;
}
TreeWithBlackHeight ExtractSubtreeWithBlackHeight(NodeData* child,
                                                  size_t black_height) {
  TreeWithBlackHeight tree{child, black_height};
  if (child) {
    ops::SetParent(child, nullptr);
    if (ops::GetColor(child) == kRed) {
      ++tree.black_height;
      ops::SetColor(child, kBlack);
    }
  }
  return tree;
}
NodeData* ExtremeNode(NodeData* x, Direction dir) {
  assert(x);
  while (auto* child = ops::Child(x, dir)) x = child;
  return x;
}
NodeData* TreeExtremeNode(NodeData* root, Direction dir) {
  if (!root) return nullptr;
  return ops::ExtremeNode(root, dir);
}
NodeData* Traverse(NodeData* x, Direction dir) {
  if (auto* child = ops::Child(x, dir)) {
    return ops::ExtremeNode(child, !dir);
  }
  auto* y = ops::Parent(x);
  while (y && x == ops::Child(y, dir)) {
    x = y;
    y = ops::Parent(y);
  }
  return y;
}
void Insert(NodeData*& root, NodeData* parent, Direction direction,
            NodeData* new_node) {
  if (!parent) {
    assert(!root);
    root = new_node;
  } else {
    if (ops::Child(parent, direction)) {
      parent = ops::Traverse(parent, direction);
      direction = !direction;
    }
    ops::Child(parent, direction) = new_node;
  }
  ops::SetParent(new_node, parent);
  ops::Child(new_node, kLeft) = nullptr;
  ops::Child(new_node, kRight) = nullptr;
  ops::SetColor(new_node, kRed);
  ops::InsertFixup(root, new_node);
}
NodeData* Join(NodeData* a_tree, NodeData* center, NodeData* b_tree,
               Direction a_dir) {
  return ops::Join({a_tree, ops::BlackHeight(a_tree)}, center,
                   {b_tree, ops::BlackHeight(b_tree)}, a_dir)
      .root;
}
NodeData* Join(NodeData* a_tree, NodeData* b_tree, Direction a_dir) {
  if (!a_tree) return b_tree;
  if (!b_tree) return a_tree;
  auto* center = ops::ExtremeNode(a_tree, !a_dir);
  ops::Remove(a_tree, center);
  return ops::Join(a_tree, center, b_tree, a_dir);
}
std::array<NodeData*, 2> Split(NodeData* root, NodeData* center) {
  std::array<TreeWithBlackHeight, 2> split_trees;
  size_t center_black_height = ops::BlackHeight(center);
  size_t child_black_height =
      center_black_height - (ops::GetColor(center) == kBlack);
  for (int dir = 0; dir < 2; ++dir) {
    split_trees[dir] = ops::ExtractSubtreeWithBlackHeight(
        ops::Child(center, static_cast<Direction>(dir)), child_black_height);
  }
  NodeData* parent = ops::Parent(center);
  while (parent) {
    Direction dir =
        static_cast<Direction>(ops::Child(parent, kRight) == center);
    NodeData* grandparent = ops::Parent(parent);
    auto parent_color = ops::GetColor(parent);
    split_trees[!dir] =
        ops::Join(split_trees[!dir], parent,
                  ops::ExtractSubtreeWithBlackHeight(ops::Child(parent, !dir),
                                                     center_black_height),
                  dir);
    center = parent;
    parent = grandparent;
    center_black_height += (parent_color == kBlack);
  }
  assert(center == root);
  return {{split_trees[0].root, split_trees[1].root}};
}
std::array<NodeData*, 2> Split(NodeData* root, NodeData*& center, Direction dir,
                               bool found) {
  if (!center) return {{nullptr, nullptr}};
  auto split_trees = ops::Split(root, center);
  if (!found) {
    ops::InsertExtreme(split_trees[!dir], dir, center);
    center = nullptr;
  }
  return split_trees;
}
void InsertExtreme(NodeData*& root, Direction dir, NodeData* new_node) {
  ops::Insert(root, ops::TreeExtremeNode(root, dir), dir, new_node);
}
void Remove(NodeData*& root, NodeData* z) {
  NodeData* y;
  if (!ops::Child(z, kLeft) || !ops::Child(z, kRight)) {
    y = z;
  } else {
    y = ops::Traverse(z, kRight);
  }
  NodeData* x =
      ops::Child(y, static_cast<Direction>(ops::Child(y, kLeft) == nullptr));
  NodeData* px = ops::Parent(y);
  if (x) {
    ops::SetParent(x, px);
  }
  if (!px) {
    root = x;
  } else {
    ops::Child(px, ops::ChildDir(y)) = x;
  }
  const Color color_removed = ops::GetColor(y);
  if (y != z) {
    if (px == z) px = y;
    Replace(root, z, y);
  } else {
    z->rbtree_parent_ = ops::DisconnectedParentValue();
  }
  if (color_removed == kRed) {
    return;
  }
  while (px && !ops::IsRed(x)) {
    const Direction dir = static_cast<Direction>(x == ops::Child(px, kRight));
    NodeData* w = ops::Child(px, !dir);
    assert(w != nullptr);
    if (ops::GetColor(w) == kRed) {
      ops::SetColor(w, kBlack);
      ops::SetColor(px, kRed);
      ops::Rotate(root, px, dir);
      w = ops::Child(px, !dir);
    }
    assert(ops::GetColor(w) == kBlack);
    if (!ops::IsRed(ops::Child(w, kLeft)) &&
        !ops::IsRed(ops::Child(w, kRight))) {
      ops::SetColor(w, kRed);
      x = px;
      px = ops::Parent(x);
    } else {
      if (!ops::IsRed(ops::Child(w, !dir))) {
        ops::SetColor(ops::Child(w, dir), kBlack);
        ops::SetColor(w, kRed);
        ops::Rotate(root, w, !dir);
        w = ops::Child(px, !dir);
      }
      ops::SetColor(w, ops::GetColor(px));
      ops::SetColor(px, kBlack);
      ops::SetColor(ops::Child(w, !dir), kBlack);
      ops::Rotate(root, px, dir);
      x = root;
      px = nullptr;
    }
  }
  if (x) ops::SetColor(x, kBlack);
}
void Replace(NodeData*& root, NodeData* existing, NodeData* replacement) {
  *replacement = *existing;
  for (int dir = 0; dir < 2; ++dir) {
    if (ops::Child(replacement, static_cast<Direction>(dir))) {
      ops::SetParent(ops::Child(replacement, static_cast<Direction>(dir)),
                     replacement);
    }
  }
  if (!ops::Parent(existing)) {
    root = replacement;
  } else {
    ops::Child(ops::Parent(existing), ops::ChildDir(existing)) = replacement;
  }
  existing->rbtree_parent_ = ops::DisconnectedParentValue();
}
}  
}  
}  
}  