#include "utils/cycle_detector.h"
#include <algorithm>
#include <optional>
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
namespace mlir {
namespace {
using NodeSet = llvm::DenseSet<int32_t>;
using OrderedNodeSet = OrderedSet<int32_t>;
template <typename T>
struct VecStruct {
  using type = llvm::SmallVector<T, 4>;
};
template <typename T>
using Vec = typename VecStruct<T>::type;
struct Node {
  int32_t rank;
  bool visited;
  void* data;
  OrderedNodeSet in;
  OrderedNodeSet out;
};
}  
struct GraphCycles::Rep {
  Vec<Node*> nodes;
  Vec<int32_t> freeNodes;
  Vec<int32_t> deltaf;
  Vec<int32_t> deltab;
  Vec<int32_t> list;
  Vec<int32_t> merged;
  Vec<int32_t> stack;
};
GraphCycles::GraphCycles(int32_t numNodes) : rep_(new Rep) {
  rep_->nodes.reserve(numNodes);
  for (int32_t i = 0; i < numNodes; ++i) {
    Node* n = new Node;
    n->visited = false;
    n->data = nullptr;
    n->rank = rep_->nodes.size();
    rep_->nodes.push_back(n);
  }
}
GraphCycles::~GraphCycles() {
  for (Vec<Node*>::size_type i = 0, e = rep_->nodes.size(); i < e; ++i) {
    delete rep_->nodes[i];
  }
  delete rep_;
}
bool GraphCycles::HasEdge(int32_t x, int32_t y) const {
  return rep_->nodes[x]->out.Contains(y);
}
void GraphCycles::RemoveEdge(int32_t x, int32_t y) {
  rep_->nodes[x]->out.Erase(y);
  rep_->nodes[y]->in.Erase(x);
}
static bool forwardDfs(GraphCycles::Rep* r, int32_t n, int32_t upperBound);
static void backwardDfs(GraphCycles::Rep* r, int32_t n, int32_t lowerBound);
static void reorder(GraphCycles::Rep* r);
static void sort(const Vec<Node*>&, Vec<int32_t>* delta);
static void moveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst);
static void clearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes);
bool GraphCycles::InsertEdge(int32_t x, int32_t y) {
  if (x == y) return false;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  if (!nx->out.Insert(y)) {
    return true;
  }
  Node* ny = r->nodes[y];
  ny->in.Insert(x);
  if (nx->rank <= ny->rank) {
    return true;
  }
  if (forwardDfs(r, y, nx->rank)) {
    nx->out.Erase(y);
    ny->in.Erase(x);
    clearVisitedBits(r, r->deltaf);
    return false;
  }
  backwardDfs(r, x, ny->rank);
  reorder(r);
  return true;
}
static bool forwardDfs(GraphCycles::Rep* r, int32_t n, int32_t upperBound) {
  r->deltaf.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;
    nn->visited = true;
    r->deltaf.push_back(n);
    for (auto w : nn->out.GetSequence()) {
      Node* nw = r->nodes[w];
      if (nw->rank == upperBound) {
        return true;
      }
      if (!nw->visited && nw->rank < upperBound) {
        r->stack.push_back(w);
      }
    }
  }
  return false;
}
static void backwardDfs(GraphCycles::Rep* r, int32_t n, int32_t lowerBound) {
  r->deltab.clear();
  r->stack.clear();
  r->stack.push_back(n);
  while (!r->stack.empty()) {
    n = r->stack.back();
    r->stack.pop_back();
    Node* nn = r->nodes[n];
    if (nn->visited) continue;
    nn->visited = true;
    r->deltab.push_back(n);
    for (auto w : nn->in.GetSequence()) {
      Node* nw = r->nodes[w];
      if (!nw->visited && lowerBound < nw->rank) {
        r->stack.push_back(w);
      }
    }
  }
}
static void reorder(GraphCycles::Rep* r) {
  sort(r->nodes, &r->deltab);
  sort(r->nodes, &r->deltaf);
  r->list.clear();
  moveToList(r, &r->deltab, &r->list);
  moveToList(r, &r->deltaf, &r->list);
  r->merged.resize(r->deltab.size() + r->deltaf.size());
  std::merge(r->deltab.begin(), r->deltab.end(), r->deltaf.begin(),
             r->deltaf.end(), r->merged.begin());
  for (Vec<int32_t>::size_type i = 0, e = r->list.size(); i < e; ++i) {
    r->nodes[r->list[i]]->rank = r->merged[i];
  }
}
static void sort(const Vec<Node*>& nodes, Vec<int32_t>* delta) {
  struct ByRank {
    const Vec<Node*>* nodes;
    bool operator()(int32_t a, int32_t b) const {
      return (*nodes)[a]->rank < (*nodes)[b]->rank;
    }
  };
  ByRank cmp;
  cmp.nodes = &nodes;
  std::sort(delta->begin(), delta->end(), cmp);
}
static void moveToList(GraphCycles::Rep* r, Vec<int32_t>* src,
                       Vec<int32_t>* dst) {
  for (Vec<int32_t>::size_type i = 0, e = src->size(); i < e; i++) {
    int32_t w = (*src)[i];
    (*src)[i] = r->nodes[w]->rank;
    r->nodes[w]->visited = false;
    dst->push_back(w);
  }
}
static void clearVisitedBits(GraphCycles::Rep* r, const Vec<int32_t>& nodes) {
  for (Vec<int32_t>::size_type i = 0, e = nodes.size(); i < e; i++) {
    r->nodes[nodes[i]]->visited = false;
  }
}
bool GraphCycles::IsReachable(int32_t x, int32_t y) {
  if (x == y) return true;
  Rep* r = rep_;
  Node* nx = r->nodes[x];
  Node* ny = r->nodes[y];
  if (nx->rank >= ny->rank) {
    return false;
  }
  bool reachable = forwardDfs(r, x, ny->rank);
  clearVisitedBits(r, r->deltaf);
  return reachable;
}
std::optional<int32_t> GraphCycles::ContractEdge(int32_t a, int32_t b) {
  assert(HasEdge(a, b));
  RemoveEdge(a, b);
  if (IsReachable(a, b)) {
    InsertEdge(a, b);
    return {};
  }
  if (rep_->nodes[b]->in.Size() + rep_->nodes[b]->out.Size() >
      rep_->nodes[a]->in.Size() + rep_->nodes[a]->out.Size()) {
    std::swap(a, b);
  }
  Node* nb = rep_->nodes[b];
  OrderedNodeSet out = std::move(nb->out);
  OrderedNodeSet in = std::move(nb->in);
  for (int32_t y : out.GetSequence()) {
    rep_->nodes[y]->in.Erase(b);
  }
  for (int32_t y : in.GetSequence()) {
    rep_->nodes[y]->out.Erase(b);
  }
  rep_->freeNodes.push_back(b);
  rep_->nodes[a]->out.Reserve(rep_->nodes[a]->out.Size() + out.Size());
  for (int32_t y : out.GetSequence()) {
    InsertEdge(a, y);
  }
  rep_->nodes[a]->in.Reserve(rep_->nodes[a]->in.Size() + in.Size());
  for (int32_t y : in.GetSequence()) {
    InsertEdge(y, a);
  }
  return a;
}
std::vector<int32_t> GraphCycles::SuccessorsCopy(int32_t node) const {
  return rep_->nodes[node]->out.GetSequence();
}
namespace {
void sortInPostOrder(const Vec<Node*>& nodes, std::vector<int32_t>* toSort) {
  std::sort(toSort->begin(), toSort->end(), [&](int32_t a, int32_t b) {
    return nodes[a]->rank > nodes[b]->rank;
  });
}
}  
std::vector<int32_t> GraphCycles::AllNodesInPostOrder() const {
  llvm::DenseSet<int32_t> freeNodesSet;
  for (int32_t n : rep_->freeNodes) freeNodesSet.insert(n);
  std::vector<int32_t> allNodes;
  allNodes.reserve(rep_->nodes.size() - freeNodesSet.size());
  for (size_t i = 0, e = rep_->nodes.size(); i < e; i++) {
    if (!freeNodesSet.count(i)) {
      allNodes.push_back(i);
    }
  }
  sortInPostOrder(rep_->nodes, &allNodes);
  return allNodes;
}
}  