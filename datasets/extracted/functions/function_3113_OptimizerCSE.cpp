#include "tensorflow/core/graph/optimizer_cse.h"
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
namespace tensorflow {
class OptimizerCSE {
 public:
  explicit OptimizerCSE(Graph* g) : g_(g) {}
  bool Optimize(const std::function<bool(const Node*)>& consider_fn);
 private:
  static size_t NodeHash(const Node* n);
  static bool Equivalent(const Node* a, const Node* b,
                         AttrSlice::Scratch* scratch);
  Graph* g_;
};
static void FillInputs(
    const Node* n, absl::InlinedVector<const Node*, 4UL>* control_edges,
    absl::InlinedVector<std::pair<const Node*, int>, 4UL>* in) {
  DCHECK_EQ(in->size(), n->num_inputs());
  control_edges->clear();
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
  if (n->op_def().is_commutative()) {
    std::sort(in->begin(), in->end());
  }
}
static size_t kIllegalNodeHash = 0;
class Hasher {
 public:
  uint64 hash() { return h_ == kIllegalNodeHash ? kIllegalNodeHash + 1 : h_; }
  void MixString(const string& s) { h_ = Hash64(s.data(), s.size(), h_); }
  void MixInteger(size_t z) { h_ = Hash64Combine(h_, z); }
  void MixProto(const protobuf::MessageLite& msg) {
    msg.ByteSizeLong();  
    HashingOutputStream hasher;
    {
      protobuf::io::CodedOutputStream stream(&hasher);
      stream.EnableAliasing(true);
      stream.SetSerializationDeterministic(true);
      msg.SerializeWithCachedSizes(&stream);
    }
    h_ = Hash64Combine(h_, hasher.hash());
  }
 private:
  class HashingOutputStream : public protobuf::io::ZeroCopyOutputStream {
   public:
    static constexpr size_t kBufSize = 228;
    static constexpr uint64 kDefaultSeed = 2570847921467975139ULL;
    bool Next(void** data, int* size) override {
      if (i_ == kBufSize) {
        Mix(buf_, kBufSize);
        *data = buf_;
        *size = kBufSize;
      } else {
        *data = buf_ + i_;
        *size = kBufSize - i_;
      }
      i_ = kBufSize;
      return true;
    }
    void BackUp(int count) override { i_ -= count; }
    int64_t ByteCount() const override { return byte_count_; }
    bool WriteAliasedRaw(const void* void_data, int size) override {
      const char* data = static_cast<const char*>(void_data);
      const auto remaining = kBufSize - i_;
      if (remaining > 0) {
        if (size < remaining) {
          memcpy(buf_ + i_, data, size);
          i_ += size;
          return true;
        }
        memcpy(buf_ + i_, data, remaining);
        i_ = kBufSize;
        data += remaining;
        size -= remaining;
      }
      if (i_ == kBufSize) {
        Mix(buf_, kBufSize);
        i_ = 0;
      }
      while (size >= kBufSize) {
        Mix(data, kBufSize);
        data += kBufSize;
        size -= kBufSize;
      }
      memcpy(buf_, data, size);
      i_ = size;
      return true;
    }
    bool AllowsAliasing() const override { return true; }
    uint64 hash() {
      if (i_ != 0) {
        Mix(buf_, i_);
        i_ = 0;
      }
      return h_;
    }
   private:
    void Mix(const char* p, size_t n) {
      byte_count_ += n;
      h_ = Hash64(p, n, h_);
    }
    char buf_[kBufSize];
    int i_ = 0;
    int64_t byte_count_ = 0;
    uint64 h_ = kDefaultSeed;
  };
  uint64 h_ = HashingOutputStream::kDefaultSeed;
};
size_t OptimizerCSE::NodeHash(const Node* n) {
  Hasher hasher;
  hasher.MixString(n->type_string());
  hasher.MixInteger(n->output_types().size());
  for (DataType dt : n->output_types()) {
    hasher.MixInteger(dt);
  }
  hasher.MixInteger(n->num_inputs());
  absl::InlinedVector<const Node*, 4UL> control_edges;
  absl::InlinedVector<std::pair<const Node*, int>, 4UL> in(n->num_inputs());
  FillInputs(n, &control_edges, &in);
  for (const auto& edge : in) {
    hasher.MixInteger(edge.first->id());
    hasher.MixInteger(edge.second);
  }
#if !defined(__ANDROID__)
  size_t attr_hashes = 0;
  for (const auto& attr : n->attrs()) {
    Hasher h;
    h.MixString(attr.first);
    h.MixProto(attr.second);
    attr_hashes = Hash64CombineUnordered(attr_hashes, h.hash());
  }
  hasher.MixInteger(attr_hashes);
#endif
  return hasher.hash();
}
static bool HasRefInput(const Node* n) {
  for (auto dt : n->input_types()) {
    if (IsRefType(dt)) return true;
  }
  return false;
}
bool OptimizerCSE::Equivalent(const Node* a, const Node* b,
                              AttrSlice::Scratch* scratch) {
  if (a->type_string() != b->type_string()) return false;
  if (a->op_def().is_stateful()) return false;
  if (HasRefInput(a) || HasRefInput(b)) return false;
  if (!a->attrs().EqualAttrs(b->attrs(), scratch)) return false;
  if (a->num_inputs() != b->num_inputs()) return false;
  const int N_in = a->num_inputs();
  absl::InlinedVector<const Node*, 4UL> a_control_edges;
  absl::InlinedVector<const Node*, 4UL> b_control_edges;
  absl::InlinedVector<std::pair<const Node*, int>, 4UL> a_in(N_in);
  absl::InlinedVector<std::pair<const Node*, int>, 4UL> b_in(N_in);
  FillInputs(a, &a_control_edges, &a_in);
  FillInputs(b, &b_control_edges, &b_in);
  if (a_in != b_in) return false;
  if (a_control_edges != b_control_edges) return false;
  return true;
}
bool OptimizerCSE::Optimize(
    const std::function<bool(const Node*)>& consider_fn) {
  std::vector<Node*> order;
  GetReversePostOrder(*g_, &order, NodeComparatorID());
  std::unordered_map<size_t, Node*> available;
  bool changed = false;
  AttrSlice::Scratch scratch;
  for (Node* n : order) {
    if (!n->IsOp()) continue;
    if (n->type_string() == "Placeholder" ||
        n->type_string() == "PlaceholderV2" ||
        n->type_string() == "PlaceholderWithDefault") {
      continue;
    }
    if (consider_fn != nullptr && !consider_fn(n)) continue;
    size_t h = NodeHash(n);
    Node** candidate = &available[h];
    if (*candidate == nullptr) {
      *candidate = n;
    } else if (Equivalent(*candidate, n, &scratch)) {
      VLOG(1) << "CSE: equivalent: " << (*candidate)->name() << " and "
              << n->name();
      for (const Edge* e : n->out_edges()) {
        g_->AddEdge(*candidate, e->src_output(), e->dst(), e->dst_input());
      }
      MergeDebugInfo(NodeDebugInfo(*n), *candidate);
      g_->RemoveNode(n);
      changed = true;
    }
  }
  return changed;
}
bool OptimizeCSE(Graph* g,
                 const std::function<bool(const Node*)>& consider_fn) {
  OptimizerCSE opt(g);
  return opt.Optimize(consider_fn);
}
}  