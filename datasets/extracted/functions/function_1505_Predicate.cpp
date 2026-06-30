#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/deadness_analysis_internal.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/hash/hash.h"
namespace tensorflow {
namespace {
using tsl::StatusOr;
class Predicate {
 public:
  enum class Kind { kAnd, kOr, kNot, kAndRecurrence, kSymbol, kIntSymbol };
  virtual string ToString() const = 0;
  int64_t id() const { return id_; }
  virtual absl::Span<Predicate* const> GetOperands() const = 0;
  virtual Kind kind() const = 0;
  virtual ~Predicate() {}
  template <typename FunctionTy>
  static void Visit(Predicate* p, const FunctionTy& func);
 protected:
  explicit Predicate(int64_t id) : id_(id) {}
 private:
  const int64_t id_;
  Predicate(const Predicate&) = delete;
  void operator=(const Predicate&) = delete;
};
class AndPredicate : public Predicate {
 public:
  explicit AndPredicate(int64_t id, std::vector<Predicate*> operands)
      : Predicate(id), operands_(std::move(operands)) {}
  string ToString() const override {
    if (operands().empty()) {
      return "#true";
    }
    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });
    return absl::StrCat("(", absl::StrJoin(operands_str, " & "), ")");
  }
  Kind kind() const override { return Kind::kAnd; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
  absl::Span<Predicate* const> operands() const { return operands_; }
 private:
  std::vector<Predicate*> operands_;
};
class OrPredicate : public Predicate {
 public:
  explicit OrPredicate(int64_t id, std::vector<Predicate*> operands)
      : Predicate(id), operands_(std::move(operands)) {}
  string ToString() const override {
    if (operands().empty()) {
      return "#false";
    }
    std::vector<string> operands_str;
    std::transform(operands().begin(), operands().end(),
                   std::back_inserter(operands_str),
                   [](Predicate* pred) { return pred->ToString(); });
    return absl::StrCat("(", absl::StrJoin(operands_str, " | "), ")");
  }
  Kind kind() const override { return Kind::kOr; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
  absl::Span<Predicate* const> operands() const { return operands_; }
 private:
  std::vector<Predicate*> operands_;
};
class NotPredicate : public Predicate {
 public:
  explicit NotPredicate(int64_t id, Predicate* operand)
      : Predicate(id), operands_({operand}) {}
  string ToString() const override {
    return absl::StrCat("~", operand()->ToString());
  }
  Kind kind() const override { return Kind::kNot; }
  Predicate* operand() const { return operands_[0]; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
 private:
  std::array<Predicate*, 1> operands_;
};
class AndRecurrencePredicate : public Predicate {
 public:
  explicit AndRecurrencePredicate(int64_t id, Predicate* start, Predicate* step,
                                  std::vector<string> frame)
      : Predicate(id), operands_({start, step}), frame_(std::move(frame)) {}
  Predicate* start() const { return operands_[0]; }
  Predicate* step() const { return operands_[1]; }
  absl::Span<const string> frame() const { return frame_; }
  string ToString() const override {
    return absl::StrCat("{", start()->ToString(), ",&,", step()->ToString(),
                        "}<", absl::StrJoin(frame(), ";"), ">");
  }
  Kind kind() const override { return Kind::kAndRecurrence; }
  absl::Span<Predicate* const> GetOperands() const override {
    return operands_;
  }
 private:
  std::array<Predicate*, 2> operands_;
  std::vector<string> frame_;
};
class SymbolPredicate : public Predicate {
 public:
  explicit SymbolPredicate(int64_t id, TensorId tensor_id, bool must_be_true)
      : Predicate(id),
        tensor_id_(std::move(tensor_id)),
        must_be_true_(must_be_true) {}
  string ToString() const override {
    return must_be_true() ? absl::StrCat("*", tensor_id_.ToString())
                          : tensor_id_.ToString();
  }
  Kind kind() const override { return Kind::kSymbol; }
  absl::Span<Predicate* const> GetOperands() const override { return {}; }
  TensorId tensor_id() const { return tensor_id_; }
  bool must_be_true() const { return must_be_true_; }
 private:
  TensorId tensor_id_;
  bool must_be_true_;
};
class IntSymbolPredicate : public Predicate {
 public:
  explicit IntSymbolPredicate(int64_t id, TensorId tensor_id,
                              std::optional<int> must_have_value)
      : Predicate(id),
        tensor_id_(std::move(tensor_id)),
        must_have_value_(must_have_value) {}
  string ToString() const override {
    return must_have_value().has_value()
               ? absl::StrCat(tensor_id_.ToString(), "=", *must_have_value_)
               : tensor_id_.ToString();
  }
  Kind kind() const override { return Kind::kIntSymbol; }
  absl::Span<Predicate* const> GetOperands() const override { return {}; }
  TensorId tensor_id() const { return tensor_id_; }
  const std::optional<int>& must_have_value() const { return must_have_value_; }
 private:
  TensorId tensor_id_;
  std::optional<int> must_have_value_;
};
template <typename FunctionTy>
 void Predicate::Visit(Predicate* p, const FunctionTy& func) {
  absl::flat_hash_set<Predicate*> visited;
  std::vector<Predicate*> stack;
  stack.push_back(p);
  visited.insert(p);
  while (!stack.empty()) {
    Predicate* current = stack.back();
    stack.pop_back();
    bool done = func(current);
    if (done) {
      return;
    }
    for (Predicate* op : current->GetOperands()) {
      if (visited.insert(op).second) {
        stack.push_back(op);
      }
    }
  }
}
class PredicateFactory {
 public:
  Predicate* MakeAndPredicate(absl::Span<Predicate* const> operands) {
    return MakeAndOrImpl(operands, true);
  }
  Predicate* MakeOrPredicate(absl::Span<Predicate* const> operands) {
    return MakeAndOrImpl(operands, false);
  }
  Predicate* MakeNotPredicate(Predicate* pred) {
    auto it = make_not_predicate_cache_.find(pred);
    if (it != make_not_predicate_cache_.end()) {
      return it->second;
    }
    Predicate* result = MakeNotPredicateImpl(pred);
    bool insert_successful =
        make_not_predicate_cache_.insert({pred, result}).second;
    (void)insert_successful;
    DCHECK(insert_successful);
    return result;
  }
  Predicate* MakeAndRecurrencePredicate(Predicate* start, Predicate* step,
                                        std::vector<string> frame) {
    SignatureForAndRec signature(start, step, std::move(frame));
    auto it = interned_and_rec_instances_.find(signature);
    if (it != interned_and_rec_instances_.end()) {
      return it->second.get();
    }
    std::unique_ptr<Predicate> new_pred = Make<AndRecurrencePredicate>(
        std::get<0>(signature), std::get<1>(signature), std::get<2>(signature));
    Predicate* new_pred_ptr = new_pred.get();
    bool inserted =
        interned_and_rec_instances_.emplace(signature, std::move(new_pred))
            .second;
    (void)inserted;
    DCHECK(inserted);
    return new_pred_ptr;
  }
  Status MakeSymbolPredicate(Node* node, int output_idx, bool must_be_true,
                             Predicate** predicate) {
    TensorId tensor_id(node->name(), output_idx);
    bool is_boolean_tensor =
        BaseType(node->output_type(tensor_id.index())) == DT_BOOL;
    TF_RET_CHECK(!must_be_true || is_boolean_tensor);
    if (node->type_string() == "Const" && must_be_true) {
      const TensorProto* proto = nullptr;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "value", &proto));
      Tensor tensor(proto->dtype());
      TF_RET_CHECK(tensor.FromProto(*proto));
      *predicate = tensor.scalar<bool>()() ? MakeTrue() : MakeFalse();
      return absl::OkStatus();
    }
    SignatureForSymbol signature = {tensor_id, must_be_true};
    auto it = interned_symbol_instances_.find(signature);
    if (it == interned_symbol_instances_.end()) {
      std::unique_ptr<Predicate> new_pred =
          Make<SymbolPredicate>(tensor_id, must_be_true);
      Predicate* new_pred_ptr = new_pred.get();
      interned_symbol_instances_.emplace(std::move(signature),
                                         std::move(new_pred));
      *predicate = new_pred_ptr;
    } else {
      *predicate = it->second.get();
    }
    return absl::OkStatus();
  }
  Status MakeSymbolPredicate(Node* node, int output_idx,
                             std::optional<int> must_have_value,
                             Predicate** predicate) {
    TensorId tensor_id(node->name(), output_idx);
    TF_RET_CHECK(BaseType(node->output_type(tensor_id.index())) == DT_INT32);
    if (must_have_value.has_value() && node->type_string() == "Const") {
      const TensorProto* proto = nullptr;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "value", &proto));
      Tensor tensor(proto->dtype());
      TF_RET_CHECK(tensor.FromProto(*proto));
      *predicate = tensor.scalar<int32>()() == *must_have_value ? MakeTrue()
                                                                : MakeFalse();
      return absl::OkStatus();
    }
    SignatureForIntSymbol signature = {tensor_id, must_have_value};
    auto it = interned_int_symbol_instances_.find(signature);
    if (it == interned_int_symbol_instances_.end()) {
      std::unique_ptr<Predicate> new_pred =
          Make<IntSymbolPredicate>(tensor_id, must_have_value);
      Predicate* new_pred_ptr = new_pred.get();
      interned_int_symbol_instances_.emplace(std::move(signature),
                                             std::move(new_pred));
      *predicate = new_pred_ptr;
    } else {
      *predicate = it->second.get();
    }
    return absl::OkStatus();
  }
  Predicate* MakeTrue() { return MakeAndPredicate({}); }
  Predicate* MakeFalse() { return MakeOrPredicate({}); }
  ~PredicateFactory() {
    DCHECK_EQ(stack_depth_, 0) << "Unnested IncrementStackDepth?";
  }
 private:
  Predicate* MakeNotPredicateImpl(Predicate* pred) {
    IncrementStackDepth stack_frame(this);
    if (!stack_frame.HasOverflowed()) {
      if (Predicate* simplified = SimplifyUsingDeMorgan(pred)) {
        return simplified;
      }
      if (auto* not_pred = dynamic_cast<NotPredicate*>(pred)) {
        return not_pred->operand();
      }
    }
    SignatureForNot signature = pred;
    auto it = interned_not_instances_.find(signature);
    if (it == interned_not_instances_.end()) {
      std::unique_ptr<Predicate> new_pred = Make<NotPredicate>(pred);
      Predicate* new_pred_ptr = new_pred.get();
      interned_not_instances_.emplace(signature, std::move(new_pred));
      return new_pred_ptr;
    } else {
      return it->second.get();
    }
  }
  Predicate* SimplifyUsingDeMorgan(Predicate* pred) {
    Predicate::Kind kind = pred->kind();
    if (kind == Predicate::Kind::kAnd || kind == Predicate::Kind::kOr) {
      std::vector<Predicate*> new_operands;
      absl::c_transform(pred->GetOperands(), std::back_inserter(new_operands),
                        [&](Predicate* p) { return MakeNotPredicate(p); });
      return kind == Predicate::Kind::kOr ? MakeAndPredicate(new_operands)
                                          : MakeOrPredicate(new_operands);
    }
    return nullptr;
  }
  template <typename PredicateT, typename... Args>
  std::unique_ptr<Predicate> Make(Args&&... args) {
    return std::unique_ptr<PredicateT>(
        new PredicateT(id_counter_++, std::forward<Args>(args)...));
  }
  Predicate* MakeAndOrImpl(absl::Span<Predicate* const> operands, bool is_and);
  Predicate* MakeInternedAndOr(std::vector<Predicate*> simplified_ops,
                               Predicate::Kind pred_kind);
  using SignatureForAndOr =
      std::pair<Predicate::Kind, absl::Span<Predicate* const>>;
  using SignatureForNot = Predicate*;
  using SignatureForAndRec =
      std::tuple<Predicate*, Predicate*, std::vector<string>>;
  using SignatureForSymbol = std::pair<SafeTensorId, bool>;
  using SignatureForIntSymbol = std::pair<SafeTensorId, std::optional<int32>>;
  struct HashSignatureForAndOr {
    size_t operator()(const SignatureForAndOr& signature) const {
      size_t hash = ::tensorflow::hash<Predicate::Kind>()(signature.first);
      for (Predicate* p : signature.second) {
        hash = Hash64Combine(hash, ::tensorflow::hash<Predicate*>()(p));
      }
      return hash;
    }
  };
  struct HashSignatureForSymbol {
    size_t operator()(const SignatureForSymbol& signature) const {
      return Hash64Combine(SafeTensorId::Hasher()(signature.first),
                           ::tensorflow::hash<bool>()(signature.second));
    }
  };
  struct HashSignatureForIntSymbol {
    size_t operator()(const SignatureForIntSymbol& signature) const {
      return Hash64Combine(
          SafeTensorId::Hasher()(signature.first),
          Hash64Combine(
              ::tensorflow::hash<bool>()(signature.second.has_value()),
              ::tensorflow::hash<int32>()(
                  signature.second.has_value() ? *signature.second : 0)));
    }
  };
  class IncrementStackDepth {
   public:
    explicit IncrementStackDepth(PredicateFactory* parent) : parent_(parent) {
      parent_->stack_depth_++;
    }
    bool HasOverflowed() const {
      const int kMaxStackDepth = 8;
      return parent_->stack_depth_ >= kMaxStackDepth;
    }
    ~IncrementStackDepth() { parent_->stack_depth_--; }
   private:
    PredicateFactory* parent_;
  };
  absl::flat_hash_map<Predicate*, Predicate*> make_not_predicate_cache_;
  absl::flat_hash_map<SignatureForAndOr, std::unique_ptr<Predicate>,
                      HashSignatureForAndOr>
      interned_and_or_instances_;
  absl::flat_hash_map<SignatureForNot, std::unique_ptr<Predicate>>
      interned_not_instances_;
  absl::flat_hash_map<SignatureForAndRec, std::unique_ptr<Predicate>>
      interned_and_rec_instances_;
  absl::flat_hash_map<SignatureForSymbol, std::unique_ptr<Predicate>,
                      HashSignatureForSymbol>
      interned_symbol_instances_;
  absl::flat_hash_map<SignatureForIntSymbol, std::unique_ptr<Predicate>,
                      HashSignatureForIntSymbol>
      interned_int_symbol_instances_;
  int64_t id_counter_ = 0;
  int stack_depth_ = 0;
};
Predicate* PredicateFactory::MakeInternedAndOr(
    std::vector<Predicate*> simplified_ops, Predicate::Kind pred_kind) {
  std::stable_sort(
      simplified_ops.begin(), simplified_ops.end(),
      [](Predicate* a, Predicate* b) { return a->id() < b->id(); });
  auto it = interned_and_or_instances_.find({pred_kind, simplified_ops});
  if (it != interned_and_or_instances_.end()) {
    return it->second.get();
  }
  simplified_ops.shrink_to_fit();
  absl::Span<Predicate* const> operands_slice = simplified_ops;
  std::unique_ptr<Predicate> new_pred =
      pred_kind == Predicate::Kind::kAnd
          ? Make<AndPredicate>(std::move(simplified_ops))
          : Make<OrPredicate>(std::move(simplified_ops));
  Predicate* new_pred_ptr = new_pred.get();
  interned_and_or_instances_.emplace(
      SignatureForAndOr(pred_kind, operands_slice), std::move(new_pred));
  return new_pred_ptr;
}
Predicate* PredicateFactory::MakeAndOrImpl(
    absl::Span<Predicate* const> operands, bool is_and) {
  Predicate::Kind pred_kind =
      is_and ? Predicate::Kind::kAnd : Predicate::Kind::kOr;
  IncrementStackDepth stack_frame(this);
  if (stack_frame.HasOverflowed()) {
    return MakeInternedAndOr(
        std::vector<Predicate*>(operands.begin(), operands.end()), pred_kind);
  }
  Predicate::Kind other_pred_kind =
      is_and ? Predicate::Kind::kOr : Predicate::Kind::kAnd;
  absl::flat_hash_set<Predicate*> simplified_ops_set;
  std::vector<Predicate*> simplified_ops;
  for (Predicate* op : operands) {
    if (!simplified_ops_set.insert(op).second) {
      continue;
    }
    if (op->kind() == pred_kind) {
      for (Predicate* subop : op->GetOperands()) {
        if (simplified_ops_set.insert(subop).second) {
          simplified_ops.push_back(subop);
        }
      }
    } else {
      simplified_ops.push_back(op);
    }
  }
  if (simplified_ops.size() == 1) {
    return simplified_ops[0];
  }
  absl::flat_hash_set<Predicate*> negated_ops;
  for (Predicate* op : simplified_ops) {
    if (negated_ops.count(op)) {
      return is_and ? MakeFalse() : MakeTrue();
    }
    Predicate* negated_op = MakeNotPredicate(op);
    if (negated_op->kind() == pred_kind) {
      if (absl::c_all_of(negated_op->GetOperands(), [&](Predicate* p) {
            return simplified_ops_set.contains(p);
          })) {
        return is_and ? MakeFalse() : MakeTrue();
      }
    }
    negated_ops.insert(negated_op);
  }
  if (is_and) {
    absl::flat_hash_set<Predicate*> to_remove;
    std::vector<Predicate*> to_add;
    for (Predicate* op : simplified_ops) {
      if (op->kind() == Predicate::Kind::kAndRecurrence) {
        auto* and_rec = static_cast<AndRecurrencePredicate*>(op);
        if (negated_ops.contains(and_rec->step())) {
          to_remove.insert(and_rec);
          to_remove.insert(MakeNotPredicate(and_rec->step()));
          to_add.push_back(and_rec->start());
        }
      }
    }
    auto it = simplified_ops.begin();
    while (it != simplified_ops.end()) {
      if (to_remove.contains(*it)) {
        it = simplified_ops.erase(it);
      } else {
        ++it;
      }
    }
    simplified_ops.insert(simplified_ops.end(), to_add.begin(), to_add.end());
  }
  std::vector<Predicate*> common_inner_operands;
  absl::flat_hash_set<Predicate*> common_inner_operands_set;
  for (Predicate* op : simplified_ops) {
    if (op->kind() != other_pred_kind) {
      common_inner_operands.clear();
      break;
    }
    if (common_inner_operands.empty()) {
      common_inner_operands.insert(common_inner_operands.end(),
                                   op->GetOperands().begin(),
                                   op->GetOperands().end());
    } else {
      common_inner_operands.clear();
      absl::c_copy_if(op->GetOperands(),
                      std::back_inserter(common_inner_operands),
                      [&](Predicate* sub_op) {
                        return common_inner_operands_set.count(sub_op) == 1;
                      });
    }
    if (common_inner_operands.empty()) break;
    common_inner_operands_set.clear();
    common_inner_operands_set.insert(common_inner_operands.begin(),
                                     common_inner_operands.end());
  }
  if (common_inner_operands.empty()) {
    return MakeInternedAndOr(std::move(simplified_ops), pred_kind);
  }
  std::vector<Predicate*> factored_ops;
  for (Predicate* op : simplified_ops) {
    std::vector<Predicate*> new_sub_op_ops;
    absl::c_copy_if(op->GetOperands(), std::back_inserter(new_sub_op_ops),
                    [&](Predicate* sub_op) {
                      return std::find(common_inner_operands.begin(),
                                       common_inner_operands.end(),
                                       sub_op) == common_inner_operands.end();
                    });
    factored_ops.push_back(MakeAndOrImpl(new_sub_op_ops, !is_and));
  }
  Predicate* new_inner_op = MakeAndOrImpl(factored_ops, is_and);
  std::vector<Predicate*> outer_ops;
  outer_ops.push_back(new_inner_op);
  outer_ops.insert(outer_ops.end(), common_inner_operands.begin(),
                   common_inner_operands.end());
  return MakeAndOrImpl(outer_ops, !is_and);
}
class DeadnessAnalysisImpl : public DeadnessAnalysis {
 public:
  explicit DeadnessAnalysisImpl(const Graph* graph)
      : graph_(*graph), vlog_(VLOG_IS_ON(2)) {}
  Status Populate(bool enable_optimistic);
  Status PopulateFrame(absl::Span<Node* const> topo, bool use_optimistic_mode,
                       bool* success);
  absl::StatusOr<DeadnessAnalysis::DeadnessPredicate> GetPredicateFor(
      Node* n, int oidx) const override;
  void Print() const override;
  absl::flat_hash_map<TensorId, string, TensorId::Hasher> PredicateMapAsString()
      const;
 private:
  enum class EdgeKind { kDataAndControl, kDataOnly, kControlOnly };
  Status GetInputPreds(Node* n, EdgeKind edge_kind,
                       std::vector<Predicate*>* result);
  void SetPredicate(Node* n, int output_idx, Predicate* pred,
                    std::vector<bool>* should_revisit) {
    auto insert_result =
        predicate_map_.insert({TensorId(n->name(), output_idx), pred});
    if (!insert_result.second && insert_result.first->second != pred) {
      VLOG(4) << "For " << n->name() << ":" << output_idx << " from "
              << insert_result.first->second->ToString() << " "
              << insert_result.first->second << " to " << pred->ToString()
              << " " << pred;
      insert_result.first->second = pred;
      if (should_revisit != nullptr) {
        for (const Edge* e : n->out_edges()) {
          (*should_revisit)[e->dst()->id()] = true;
        }
      }
    }
  }
  void SetPredicate(Node* n, absl::Span<const int> output_idxs, Predicate* pred,
                    std::vector<bool>* should_revisit) {
    for (int output_idx : output_idxs) {
      SetPredicate(n, output_idx, pred, should_revisit);
    }
  }
  Status HandleSwitch(Node* n, std::vector<bool>* should_revisit);
  Status HandleMerge(Node* n, std::vector<bool>* should_revisit,
                     bool use_optimistic_mode);
  Status HandleRecv(Node* n, std::vector<bool>* should_revisit);
  Status HandleGeneric(Node* n, std::vector<bool>* should_revisit);
  Status HandleNode(Node* n, std::vector<bool>* should_revisit,
                    bool use_optimistic_mode = false);
  Status GetFrameBasedTopologicalOrder(std::vector<Node*>* order);
  bool IsRootEnter(const Node* n) const {
    return IsEnter(n) && control_flow_info_[n->id()].parent_frame->IsSource();
  }
  bool IsRootExit(const Node* n) const {
    return IsExit(n) && control_flow_info_[n->id()].parent_frame->IsSource();
  }
  const Graph& graph_;
  absl::flat_hash_map<TensorId, Predicate*, TensorId::Hasher> predicate_map_;
  PredicateFactory predicate_factory_;
  std::vector<ControlFlowInfo> control_flow_info_;
  bool vlog_;
  absl::flat_hash_map<absl::string_view, Node*> frame_to_merge_node_;
};
TensorId InputEdgeToTensorId(const Edge* e) {
  return TensorId(e->src()->name(), e->src_output());
}
Status DeadnessAnalysisImpl::GetInputPreds(
    Node* n, DeadnessAnalysisImpl::EdgeKind edge_kind,
    std::vector<Predicate*>* result) {
  result->clear();
  for (const Edge* in_edge : n->in_edges()) {
    bool should_process =
        edge_kind == EdgeKind::kDataAndControl ||
        (in_edge->IsControlEdge() && edge_kind == EdgeKind::kControlOnly) ||
        (!in_edge->IsControlEdge() && edge_kind == EdgeKind::kDataOnly);
    if (should_process) {
      auto it = predicate_map_.find(InputEdgeToTensorId(in_edge));
      if (it == predicate_map_.end()) {
        xla::GraphCycles graph_cycles;
        TF_RETURN_IF_ERROR(
            CreateCycleDetectionGraph(&graph_, &graph_cycles).status());
        return errors::Internal("Could not find input ", in_edge->DebugString(),
                                " to ", n->name(),
                                " when visiting the graph in post-order.  Most "
                                "likely indicates a bug in deadness analysis.");
      }
      result->push_back(it->second);
    }
  }
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::HandleSwitch(Node* n,
                                          std::vector<bool>* should_revisit) {
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  const Edge* pred_edge;
  TF_RETURN_IF_ERROR(n->input_edge(1, &pred_edge));
  if (n->type_string() != "_SwitchN") {  
    Predicate* true_switch;
    TF_RETURN_IF_ERROR(predicate_factory_.MakeSymbolPredicate(
        pred_edge->src(), pred_edge->src_output(),
        true, &true_switch));
    Predicate* false_switch = predicate_factory_.MakeNotPredicate(true_switch);
    input_preds.push_back(false_switch);
    SetPredicate(n, 0, predicate_factory_.MakeAndPredicate(input_preds),
                 should_revisit);
    input_preds.pop_back();
    input_preds.push_back(true_switch);
    SetPredicate(n, 1, predicate_factory_.MakeAndPredicate(input_preds),
                 should_revisit);
    input_preds.pop_back();
  } else {  
    Predicate* branch_pred;
    for (int i = 0; i < n->num_outputs() - 1; i++) {
      TF_RETURN_IF_ERROR(predicate_factory_.MakeSymbolPredicate(
          pred_edge->src(), pred_edge->src_output(),
          std::optional<int32>(i), &branch_pred));
      input_preds.push_back(branch_pred);
      SetPredicate(n, i, predicate_factory_.MakeAndPredicate(input_preds),
                   should_revisit);
      input_preds.pop_back();
      input_preds.push_back(predicate_factory_.MakeNotPredicate(branch_pred));
    }
    SetPredicate(n, n->num_outputs() - 1,
                 predicate_factory_.MakeAndPredicate(input_preds),
                 should_revisit);
  }
  SetPredicate(n, Graph::kControlSlot,
               predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);
  return absl::OkStatus();
}
namespace {
Status CreateMultipleNextIterationInputsError(Node* merge) {
  std::vector<string> backedges;
  for (const Edge* backedge : merge->in_edges()) {
    if (backedge->src()->IsNextIteration()) {
      backedges.push_back(absl::StrCat("  ", SummarizeNode(*backedge->src())));
    }
  }
  return errors::InvalidArgument(
      "Multiple NextIteration inputs to merge node ",
      FormatNodeForError(*merge), ": \n", absl::StrJoin(backedges, "\n"),
      "\nMerge nodes can have at most one incoming NextIteration edge.");
}
Status FindUniqueBackedge(Node* merge, const Edge** result) {
  *result = nullptr;
  CHECK(merge->IsMerge());
  for (const Edge* e : merge->in_edges()) {
    if (e->src()->IsNextIteration()) {
      if (*result != nullptr) {
        return CreateMultipleNextIterationInputsError(merge);
      }
      *result = e;
    }
  }
  return absl::OkStatus();
}
Predicate* DeduceStepPredicate(PredicateFactory* predicate_factory,
                               Predicate* symbolic_predicate,
                               Predicate* backedge_predicate) {
  CHECK(dynamic_cast<SymbolPredicate*>(symbolic_predicate));
  if (backedge_predicate->kind() != Predicate::Kind::kAnd) {
    return nullptr;
  }
  std::vector<Predicate*> and_ops;
  absl::Span<Predicate* const> recurrent_pred_ops =
      backedge_predicate->GetOperands();
  bool found_sym = false;
  for (Predicate* and_op : recurrent_pred_ops) {
    if (and_op == symbolic_predicate) {
      found_sym = true;
      continue;
    }
    bool found_sym_as_inner_operand = false;
    auto has_self_as_inner_operand = [&](Predicate* p) {
      if (p == symbolic_predicate) {
        found_sym_as_inner_operand = true;
        return true;  
      }
      return false;
    };
    Predicate::Visit(and_op, has_self_as_inner_operand);
    if (found_sym_as_inner_operand) {
      return nullptr;
    }
    and_ops.push_back(and_op);
  }
  return found_sym ? predicate_factory->MakeAndPredicate(and_ops) : nullptr;
}
Status GetFullFrame(const Node* n, absl::Span<const ControlFlowInfo> cfi_infos,
                    std::vector<string>* frame) {
  int depth = 0;
  for (const ControlFlowInfo* cfi_iter = &cfi_infos[n->id()]; !n->IsSource();
       n = cfi_iter->parent_frame, cfi_iter = &cfi_infos[n->id()]) {
    frame->push_back(cfi_iter->frame_name);
    if (depth++ > 5000) {
      return errors::Internal(
          "Frame of depth > 5000:  Probably malformed graph or a bug in "
          "BuildControlFlowInfo");
    }
  }
  return absl::OkStatus();
}
Status GetRootFrame(const Node* n, absl::Span<const ControlFlowInfo> cfi_infos,
                    absl::string_view* frame) {
  int depth = 0;
  const ControlFlowInfo* cfi_iter = &cfi_infos[n->id()];
  while (!cfi_iter->parent_frame->IsSource()) {
    n = cfi_iter->parent_frame;
    cfi_iter = &cfi_infos[n->id()];
    if (depth++ > 5000) {
      return errors::Internal(
          "Frame of depth > 5000:  Probably malformed graph or a bug in "
          "BuildControlFlowInfo");
    }
  }
  *frame = cfi_iter->frame_name;
  return absl::OkStatus();
}
}  
Status DeadnessAnalysisImpl::HandleMerge(Node* n,
                                         std::vector<bool>* should_revisit,
                                         bool use_optimistic_mode) {
  bool has_unvisited_backedge = false;
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->IsNextIteration()) {
      has_unvisited_backedge |= !predicate_map_.count(InputEdgeToTensorId(e));
    }
  }
  auto it = predicate_map_.find(TensorId(n->name(), 0));
  if (it == predicate_map_.end()) {
    if (has_unvisited_backedge) {
      Predicate* input_data_pred;
      if (use_optimistic_mode) {
        absl::string_view frame_name = control_flow_info_[n->id()].frame_name;
        auto insert_result = frame_to_merge_node_.insert({frame_name, n});
        Node* representative = insert_result.first->second;
        TF_RETURN_IF_ERROR(predicate_factory_.MakeSymbolPredicate(
            representative, 0, false,
            &input_data_pred));
      } else {
        TF_RETURN_IF_ERROR(predicate_factory_.MakeSymbolPredicate(
            n, 0, false, &input_data_pred));
      }
      SetPredicate(n, {0, 1, Graph::kControlSlot}, input_data_pred,
                   should_revisit);
      return absl::OkStatus();
    }
    std::vector<Predicate*> input_preds;
    TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataOnly, &input_preds));
    Predicate* input_data_pred =
        predicate_factory_.MakeOrPredicate(input_preds);
    SetPredicate(n, {0, 1, Graph::kControlSlot}, input_data_pred,
                 should_revisit);
    return absl::OkStatus();
  }
  if (it->second->kind() == Predicate::Kind::kSymbol) {
    const Edge* unique_backedge;
    TF_RETURN_IF_ERROR(FindUniqueBackedge(n, &unique_backedge));
    if (unique_backedge) {
      if (Predicate* step = DeduceStepPredicate(
              &predicate_factory_, it->second,
              predicate_map_[InputEdgeToTensorId(unique_backedge)])) {
        std::vector<Predicate*> non_recurrent_inputs;
        for (const Edge* e : n->in_edges()) {
          if (e != unique_backedge) {
            non_recurrent_inputs.push_back(
                predicate_map_[InputEdgeToTensorId(e)]);
          }
        }
        Predicate* start =
            predicate_factory_.MakeOrPredicate(non_recurrent_inputs);
        std::vector<string> frame;
        TF_RETURN_IF_ERROR(GetFullFrame(n, control_flow_info_, &frame));
        Predicate* and_rec = predicate_factory_.MakeAndRecurrencePredicate(
            start, step, std::move(frame));
        SetPredicate(n, {0, 1, Graph::kControlSlot}, and_rec, should_revisit);
        return absl::OkStatus();
      }
    }
  }
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::HandleRecv(Node* n,
                                        std::vector<bool>* should_revisit) {
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  Predicate* signal_is_alive;
  TF_RETURN_IF_ERROR(predicate_factory_.MakeSymbolPredicate(
      n, 0, false, &signal_is_alive));
  input_preds.push_back(signal_is_alive);
  SetPredicate(n, {0, Graph::kControlSlot},
               predicate_factory_.MakeAndPredicate(input_preds),
               should_revisit);
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::HandleGeneric(Node* n,
                                           std::vector<bool>* should_revisit) {
  std::vector<Predicate*> input_preds;
  TF_RETURN_IF_ERROR(GetInputPreds(n, EdgeKind::kDataAndControl, &input_preds));
  Predicate* pred = predicate_factory_.MakeAndPredicate(input_preds);
  for (int output_idx = 0; output_idx < n->num_outputs(); output_idx++) {
    SetPredicate(n, output_idx, pred, should_revisit);
  }
  SetPredicate(n, Graph::kControlSlot, pred, should_revisit);
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::HandleNode(Node* n,
                                        std::vector<bool>* should_revisit,
                                        bool use_optimistic_mode) {
  if (n->IsSwitch()) {
    TF_RETURN_IF_ERROR(HandleSwitch(n, should_revisit));
  } else if (n->IsMerge()) {
    TF_RETURN_IF_ERROR(HandleMerge(n, should_revisit, use_optimistic_mode));
  } else if (n->IsControlTrigger()) {
    SetPredicate(n, Graph::kControlSlot, predicate_factory_.MakeTrue(),
                 nullptr);
  } else if (n->IsRecv() || n->IsHostRecv()) {
    TF_RETURN_IF_ERROR(HandleRecv(n, should_revisit));
  } else if (n->IsNextIteration()) {
    TF_RETURN_IF_ERROR(HandleGeneric(n, should_revisit));
  } else {
    TF_RETURN_IF_ERROR(HandleGeneric(n, should_revisit));
  }
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::GetFrameBasedTopologicalOrder(
    std::vector<Node*>* order) {
  absl::flat_hash_map<absl::string_view, size_t> num_enters_for_frame;
  absl::flat_hash_map<absl::string_view, size_t> num_exits_for_frame;
  std::vector<size_t> num_ready_inputs(graph_.num_node_ids(), 0);
  Node* src_node = graph_.source_node();
  for (const auto* node : graph_.op_nodes()) {
    const ControlFlowInfo& cf = control_flow_info_[node->id()];
    if (IsRootEnter(node)) {
      ++num_enters_for_frame[cf.frame_name];
    } else if (IsRootExit(node)) {
      ++num_exits_for_frame[cf.frame_name];
    }
    if (IsMerge(node)) {
      for (const Edge* e : node->in_edges()) {
        if (IsNextIteration(e->src())) {
          ++num_ready_inputs[node->id()];
        }
      }
    }
  }
  std::deque<Node*> ready;
  ready.push_back(src_node);
  absl::flat_hash_map<absl::string_view, std::vector<Node*>>
      ready_enters_per_frame;
  std::vector<Node*> ready_exits;
  while (!ready.empty()) {
    Node* curr_node = ready.front();
    ready.pop_front();
    VLOG(4) << "Visiting " << curr_node->name();
    order->push_back(curr_node);
    for (const Edge* out_edge : curr_node->out_edges()) {
      Node* out = out_edge->dst();
      int out_id = out->id();
      if (IsNextIteration(curr_node) && IsMerge(out)) {
        continue;
      }
      ++num_ready_inputs[out->id()];
      if (!out->IsOp()) continue;  
      if (num_ready_inputs[out->id()] != out->in_edges().size()) continue;
      absl::string_view frame_name = control_flow_info_[out_id].frame_name;
      if (IsRootEnter(out)) {
        ready_enters_per_frame[frame_name].push_back(out);
      } else if (IsRootExit(out)) {
        ready_exits.push_back(out);
      } else {
        ready.push_back(out);
      }
    }
    if (ready.empty()) {
      if (!ready_exits.empty()) {
        absl::string_view frame_name =
            control_flow_info_[ready_exits.front()->id()].frame_name;
        CHECK_EQ(ready_exits.size(), num_exits_for_frame[frame_name]);
        ready.insert(ready.end(), ready_exits.begin(), ready_exits.end());
        ready_exits.clear();
      } else {
        for (auto iter = ready_enters_per_frame.begin();
             iter != ready_enters_per_frame.end(); ++iter) {
          absl::string_view frame_name = iter->first;
          const std::vector<Node*>& ready_enters = iter->second;
          if (ready_enters.size() == num_enters_for_frame[frame_name]) {
            ready.insert(ready.end(), ready_enters.begin(), ready_enters.end());
            ready_enters_per_frame.erase(iter);
            break;
          }
        }
      }
    }
  }
  if (!ready_enters_per_frame.empty() || !ready_exits.empty()) {
    return errors::InvalidArgument(
        "Some enters/exits have never been visited in the traversal."
        " Most probably the input graph is malformed.");
  }
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::Populate(bool enable_optimistic) {
  std::vector<string> unreachable_nodes;
  TF_RETURN_IF_ERROR(
      BuildControlFlowInfo(&graph_, &control_flow_info_, &unreachable_nodes));
  if (!unreachable_nodes.empty()) {
    if (unreachable_nodes.size() > 5) {
      unreachable_nodes.erase(unreachable_nodes.begin() + 5,
                              unreachable_nodes.end());
    }
    return errors::InvalidArgument(
        "Found unreachable nodes, most likely source and sink nodes not "
        "connected: ",
        absl::StrJoin(unreachable_nodes, ", "));
  }
  std::vector<Node*> topo;
  TF_RETURN_IF_ERROR(GetFrameBasedTopologicalOrder(&topo));
  size_t frame_start = 0;
  while (frame_start < topo.size()) {
    absl::string_view cur_frame_name;
    TF_RETURN_IF_ERROR(
        GetRootFrame(topo[frame_start], control_flow_info_, &cur_frame_name));
    size_t frame_end = frame_start;
    for (size_t i = frame_start + 1; i < topo.size(); ++i) {
      absl::string_view i_frame_name;
      TF_RETURN_IF_ERROR(
          GetRootFrame(topo[i], control_flow_info_, &i_frame_name));
      if (i_frame_name == cur_frame_name) {
        frame_end = i;
      } else {
        break;
      }
    }
    absl::Span<Node*> sub_topo(topo.data() + frame_start,
                               frame_end - frame_start + 1);
    frame_start = frame_end + 1;
    bool success = false;
    if (enable_optimistic && !cur_frame_name.empty()) {
      TF_RETURN_IF_ERROR(
          PopulateFrame(sub_topo, true, &success));
    }
    if (!success) {
      TF_RETURN_IF_ERROR(
          PopulateFrame(sub_topo, false, nullptr));
    }
    VLOG(2) << "Done populating frame " << cur_frame_name << " using the "
            << (success ? "optimistic" : "pessimistic") << " mode.";
  }
  return absl::OkStatus();
}
Status DeadnessAnalysisImpl::PopulateFrame(absl::Span<Node* const> topo,
                                           bool use_optimistic_mode,
                                           bool* success) {
  CHECK(use_optimistic_mode && success != nullptr ||
        !use_optimistic_mode && success == nullptr);
  std::vector<bool> should_revisit;
  should_revisit.resize(graph_.num_node_ids());
  for (Node* n : topo) {
    VLOG(4) << "Visiting " << n->name();
    TF_RETURN_IF_ERROR(
        HandleNode(n, nullptr, use_optimistic_mode));
    if (n->IsNextIteration()) {
      for (const Edge* e : n->out_edges()) {
        if (e->dst()->IsMerge()) {
          should_revisit[e->dst()->id()] = true;
        }
      }
    }
  }
  for (Node* n : topo) {
    if (should_revisit[n->id()]) {
      VLOG(4) << "Revisiting " << n->name();
      TF_RETURN_IF_ERROR(HandleNode(n, &should_revisit));
    }
  }
  if (use_optimistic_mode) {
    bool is_converged = true;
    absl::flat_hash_map<absl::string_view, Predicate*> frame_to_pred;
    for (Node* n : topo) {
      if (!n->IsMerge()) {
        continue;
      }
      const Edge* e;
      TF_RETURN_IF_ERROR(FindUniqueBackedge(n, &e));
      if (e == nullptr) {
        continue;
      }
      Node* merge = n;
      absl::string_view frame_name = control_flow_info_[merge->id()].frame_name;
      auto it = predicate_map_.find(TensorId(merge->name(), 0));
      Predicate* merge_pred = it->second;
      if (merge_pred->kind() != Predicate::Kind::kAndRecurrence) {
        is_converged = false;
        VLOG(2) << "Running the optimistic mode on frame " << frame_name
                << " does not converge because node " << merge->name()
                << " cannot be mapped into the AndRecurrence form.";
        break;
      }
      auto insert_result = frame_to_pred.insert({frame_name, merge_pred});
      if (!insert_result.second) {
        Predicate* curr_andrec = merge_pred;
        Predicate* prev_andrec = insert_result.first->second;
        if (curr_andrec != prev_andrec) {
          is_converged = false;
          VLOG(2) << "Running the optimistic mode on frame " << frame_name
                  << " does not converge. Seeing different Merge predicates: \n"
                  << curr_andrec->ToString() << " and \n"
                  << prev_andrec->ToString();
          break;
        }
      }
    }
    if (!is_converged) {
      for (Node* n : topo) {
        for (int oid = 0; oid < n->num_outputs(); ++oid) {
          predicate_map_.erase(TensorId(n->name(), oid));
        }
        predicate_map_.erase(TensorId(n->name(), Graph::kControlSlot));
      }
    }
    if (success != nullptr) {
      *success = is_converged;
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<DeadnessAnalysis::DeadnessPredicate>
DeadnessAnalysisImpl::GetPredicateFor(Node* n, int oidx) const {
  auto it = predicate_map_.find(TensorId(n->name(), oidx));
  TF_RET_CHECK(it != predicate_map_.end())
      << "could not find " << TensorId(n->name(), oidx).ToString()
      << " in predicate map";
  return MakeDeadnessPredicate(it->second);
}
void DeadnessAnalysisImpl::Print() const {
  std::vector<TensorId> tensor_ids;
  tensor_ids.reserve(predicate_map_.size());
  for (const auto& kv_pair : predicate_map_) {
    tensor_ids.push_back(kv_pair.first);
  }
  std::sort(tensor_ids.begin(), tensor_ids.end());
  for (TensorId tensor_id : tensor_ids) {
    auto it = predicate_map_.find(tensor_id);
    CHECK(it != predicate_map_.end()) << tensor_id.ToString();
    VLOG(2) << tensor_id.ToString() << " -> " << it->second->ToString();
  }
}
}  
DeadnessAnalysis::~DeadnessAnalysis() {}
 Status DeadnessAnalysis::Run(
    const Graph& graph, std::unique_ptr<DeadnessAnalysis>* result) {
  std::unique_ptr<DeadnessAnalysisImpl> analysis(
      new DeadnessAnalysisImpl(&graph));
  TF_RETURN_IF_ERROR(analysis->Populate(true));
  if (VLOG_IS_ON(2)) {
    analysis->Print();
  }
  *result = std::move(analysis);
  return absl::OkStatus();
}
absl::flat_hash_map<TensorId, string, TensorId::Hasher>
DeadnessAnalysisImpl::PredicateMapAsString() const {
  absl::flat_hash_map<TensorId, string, TensorId::Hasher> result;
  for (const auto& kv_pair : predicate_map_) {
    CHECK(result.insert({kv_pair.first, kv_pair.second->ToString()}).second);
  }
  return result;
}
namespace deadness_analysis_internal {
Status ComputePredicates(const Graph& graph, PredicateMapTy* out_predicate_map,
                         bool enable_optimistic) {
  DeadnessAnalysisImpl impl(&graph);
  TF_RETURN_IF_ERROR(impl.Populate(enable_optimistic));
  *out_predicate_map = impl.PredicateMapAsString();
  return absl::OkStatus();
}
}  
string DeadnessAnalysis::DebugString(DeadnessPredicate predicate) const {
  return static_cast<Predicate*>(predicate.pred_)->ToString();
}
}  