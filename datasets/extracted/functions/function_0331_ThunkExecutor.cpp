#include "xla/backends/cpu/runtime/thunk_executor.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"
namespace xla::cpu {
ThunkExecutor::ThunkExecutor(ThunkSequence thunk_sequence,
                             std::vector<NodeDef> nodes_defs,
                             const ThunkExecutor::Options& options)
    : thunk_sequence_(std::move(thunk_sequence)),
      options_(options),
      num_thunks_(thunk_sequence_.size()),
      nodes_defs_(std::move(nodes_defs)),
      is_sequential_(true) {
  for (NodeId i = 0; i < nodes_defs_.size(); ++i) {
    if (nodes_defs_[i].in_edges.empty()) {
      source_.push_back(i);
    }
    if (nodes_defs_[i].out_edges.empty()) {
      sink_.push_back(i);
    }
  }
  int64_t num_erased_edges = RunTransitiveReductionAndUpdatePriorities();
  for (NodeId i = 1; i < nodes_defs_.size() && is_sequential_; ++i) {
    is_sequential_ &= (absl::c_count(nodes_defs_[i].in_edges, i - 1) != 0);
  }
  auto uses_small_buffers = [&](const std::unique_ptr<Thunk>& thunk) {
    return absl::c_all_of(thunk->buffer_uses(), [&](const BufferUse& use) {
      return use.slice().size() <= options.execute_sequential_buffer_threshold;
    });
  };
  bool small_buffers = absl::c_all_of(thunk_sequence_, uses_small_buffers);
  is_sequential_ |= small_buffers;
  is_sequential_ |=
      thunk_sequence_.size() <= options.execute_sequential_num_thunks_threshold;
  VLOG(2) << absl::StreamFormat(
      "Constructed ThunkExecutor with %d nodes: #source_nodes=%d "
      "#sink_nodes=%d, #erased_edges=%d, is_sequential=%v, small_buffers=%v",
      nodes_defs_.size(), source_.size(), sink_.size(), num_erased_edges,
      is_sequential_, small_buffers);
  DCHECK((!source_.empty() && !sink_.empty() && !thunk_sequence_.empty()) ||
         (source_.empty() && sink_.empty() && thunk_sequence_.empty()));
}
absl::StatusOr<ThunkExecutor> ThunkExecutor::Create(
    ThunkSequence thunk_sequence, const ThunkExecutor::Options& options) {
  std::vector<NodeDef> defs(thunk_sequence.size());
  std::vector<BufferUse::ReadWriteSet> buffer_rwsets(thunk_sequence.size());
  std::vector<ResourceUse::ReadWriteSet> resource_rwsets(thunk_sequence.size());
  for (NodeId i = 0; i < thunk_sequence.size(); ++i) {
    defs[i].id = i;
    Thunk& thunk = *thunk_sequence[i];
    buffer_rwsets[i].AddAll(thunk.buffer_uses());
    resource_rwsets[i].AddAll(thunk.resource_uses());
    for (NodeId j = 0; j < i; ++j) {
      if (buffer_rwsets[j].HasConflicts(buffer_rwsets[i]) ||
          resource_rwsets[j].HasConflicts(resource_rwsets[i])) {
        defs[j].out_edges.push_back(i);
        defs[i].in_edges.push_back(j);
      }
    }
  }
  for (NodeId i = 0; i < defs.size(); ++i) {
    DCHECK(absl::c_is_sorted(defs[i].out_edges));
    DCHECK(absl::c_is_sorted(defs[i].in_edges));
  }
  return ThunkExecutor(std::move(thunk_sequence), std::move(defs), options);
}
ThunkExecutor::ExecuteState::Node::Node(const NodeDef& node_def)
    : counter(node_def.in_edges.size()), out_edges(&node_def.out_edges) {}
ThunkExecutor::ExecuteState::ExecuteState(ThunkExecutor* executor,
                                          Thunk::TaskRunner* runner)
    : executor(executor),
      runner(runner),
      nodes(executor->nodes_defs().size()),
      execute_event(tsl::MakeConstructedAsyncValueRef<ExecuteEvent>()),
      pending_sink_nodes(executor->sink().size()),
      abort(false) {
  DCHECK(runner == nullptr || static_cast<bool>(*runner))
      << "`runner` must be nullptr or a valid TaskRunner";
  NodeStorage* node = nodes.data();
  for (const NodeDef& node_def : executor->nodes_defs()) {
    new (node++) Node(node_def);
  }
}
tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent> ThunkExecutor::Execute(
    const Thunk::ExecuteParams& params) {
  if (ABSL_PREDICT_FALSE(num_thunks_ == 0)) {
    return Thunk::OkExecuteEventSingleton();
  }
  if (ABSL_PREDICT_FALSE(num_thunks_ == 1)) {
    return thunk_sequence_[0]->Execute(params);
  }
  if (is_sequential_) {
    return ExecuteSequential(params);
  }
  auto state = std::make_unique<ExecuteState>(this, params.task_runner);
  if (options_.use_priority_ready_queue) {
    Execute(state.get(), params, PriorityReadyQueue(nodes_defs_, source_),
            nullptr);
  } else {
    Execute(state.get(), params, FifoReadyQueue(source_),
            nullptr);
  }
  if (ABSL_PREDICT_TRUE(state->execute_event.IsAvailable())) {
    return std::move(state->execute_event);
  }
  tsl::AsyncValueRef<ExecuteEvent> execute_event = state->execute_event;
  execute_event.AndThen([state = std::move(state)] {
    auto cnt = state->pending_sink_nodes.load(std::memory_order_acquire);
    DCHECK_EQ(cnt, 0)
        << "All sink nodes must be completed before execute_event is marked "
           "available.";
  });
  return execute_event;
}
tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent>
ThunkExecutor::ExecuteSequential(const Thunk::ExecuteParams& params) {
  for (auto it = thunk_sequence_.begin(); it != thunk_sequence_.end(); ++it) {
    Thunk& thunk = **it;
    auto execute_event = thunk.Execute(params);
    if (ABSL_PREDICT_TRUE(thunk.IsOkExecuteEvent(execute_event))) {
      continue;
    }
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
      execute_event.AndThen([this, &params, it, event](absl::Status status) {
        if (ABSL_PREDICT_FALSE(!status.ok())) {
          event.SetError(std::move(status));
        } else {
          ResumeExecuteSequential(it + 1, params, std::move(event));
        }
      });
      return event;
    }
    if (ABSL_PREDICT_FALSE(execute_event.IsError())) {
      return execute_event;
    }
  }
  return Thunk::OkExecuteEventSingleton();
}
void ThunkExecutor::ResumeExecuteSequential(
    ThunkIterator it, const Thunk::ExecuteParams& params,
    tsl::AsyncValueRef<ExecuteEvent> event) {
  for (; it != thunk_sequence_.end(); ++it) {
    Thunk& thunk = **it;
    auto execute_event = thunk.Execute(params);
    if (ABSL_PREDICT_TRUE(thunk.IsOkExecuteEvent(execute_event))) {
      continue;
    }
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      execute_event.AndThen(
          [this, &params, it, event = std::move(event)](absl::Status status) {
            if (ABSL_PREDICT_FALSE(!status.ok())) {
              event.SetError(std::move(status));
            } else {
              ResumeExecuteSequential(it + 1, params, std::move(event));
            }
          });
      return;
    }
    if (ABSL_PREDICT_FALSE(execute_event.IsError())) {
      event.SetError(execute_event.GetError());
      return;
    }
  }
  event.SetStateConcrete();
}
template <typename ReadyQueue>
void ThunkExecutor::Execute(ExecuteState* state,
                            const Thunk::ExecuteParams& params,
                            ReadyQueue ready_queue,
                            Thunk::ExecuteSession::Lock lock) {
  DCHECK(!ready_queue.Empty()) << "Ready queue must not be empty";
  tsl::profiler::TraceMe trace("ThunkExecutor::Execute");
  bool has_runner = state->runner != nullptr;
  bool has_lock = static_cast<bool>(lock);
  int64_t split_threshold = params.session.split_threshold();
  while (!ready_queue.Empty()) {
    DCHECK_EQ(static_cast<bool>(lock), has_lock)
        << "Execute session lock must not be lost in the middle of the loop";
    NodeId id = ready_queue.Pop();
    ExecuteState::Node& node = state->node(id);
    int64_t cnt = node.counter.load(std::memory_order_acquire);
    DCHECK_EQ(cnt, 0) << "Node counter must be 0";
    int64_t num_ready_thunks = ready_queue.Size();
    if (ABSL_PREDICT_FALSE(has_runner && num_ready_thunks > split_threshold)) {
      SplitReadyQueue(state, params, ready_queue, split_threshold);
    }
    Thunk& thunk = *state->executor->thunk_sequence_[id];
    tsl::AsyncValueRef<ExecuteEvent> execute_event =
        ABSL_PREDICT_FALSE(state->abort.load(std::memory_order_relaxed))
            ? Thunk::OkExecuteEventSingleton()
            : thunk.Execute(params);
    if (ABSL_PREDICT_TRUE(execute_event.IsAvailable())) {
      ProcessOutEdges(state, execute_event.AsPtr(), node, ready_queue);
    } else {
      execute_event.AndThen(
          [&params, &node, state, execute_event = execute_event.AsPtr(),
           ready_queue = ready_queue.CreateEmptyReadyQueue(),
           lock = ready_queue.Empty() ? std::move(lock)
                                      : params.session.Join()]() mutable {
            state->executor->ProcessOutEdges(state, execute_event, node,
                                             ready_queue);
            if (ABSL_PREDICT_TRUE(!ready_queue.Empty())) {
              state->executor->Execute(state, params, std::move(ready_queue),
                                       std::move(lock));
            }
          });
    }
  }
}
template <typename ReadyQueue>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void ThunkExecutor::SplitReadyQueue(
    ExecuteState* state, const Thunk::ExecuteParams& params,
    ReadyQueue& ready_queue, int64_t split_threshold) {
  DCHECK(state->runner) << "TaskRunner must be set";
  while (ready_queue.Size() > split_threshold) {
    Thunk::ExecuteSession::Lock task_runner_lock = params.session.TryJoin();
    if (!task_runner_lock) {
      break;
    }
    (*state->runner)([&params, state, ready_queue = ready_queue.PopHalf(),
                      lock = std::move(task_runner_lock)]() mutable {
      state->executor->Execute(state, params, std::move(ready_queue),
                               std::move(lock));
    });
  }
}
template <typename ReadyQueue>
void ThunkExecutor::ProcessOutEdges(
    ExecuteState* state, tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
    ExecuteState::Node& node, ReadyQueue& ready_queue) {
  if (ABSL_PREDICT_FALSE(node_event.IsError())) {
    absl::MutexLock lock(&state->abort_mutex);
    state->abort = true;
    state->abort_status.Update(node_event.GetError());
  }
  bool is_sink = node.out_edges->empty();
  for (NodeId out_edge : *node.out_edges) {
    ExecuteState::Node& out_node = state->node(out_edge);
    int64_t cnt = out_node.counter.fetch_sub(1, std::memory_order_release);
    DCHECK_GE(cnt, 1) << "Node counter can't drop below 0";
    if (cnt == 1) ready_queue.Push(out_edge);
  }
  if (ABSL_PREDICT_FALSE(is_sink)) {
    bool is_done =
        state->pending_sink_nodes.fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (ABSL_PREDICT_TRUE(!is_done)) return;
    if (ABSL_PREDICT_FALSE(state->abort.load(std::memory_order_relaxed))) {
      auto take_error = [&] {
        absl::MutexLock lock(&state->abort_mutex);
        DCHECK(!state->abort_status.ok())
            << "Abort status must be set if execution is aborted";
        return std::move(state->abort_status);
      };
      state->execute_event.SetError(take_error());
    } else {
      state->execute_event.SetStateConcrete();
    }
  }
}
static int64_t EraseEdge(ThunkExecutor::NodeDef& from,
                         ThunkExecutor::NodeDef& to) {
  DCHECK_NE(from.id, to.id) << "Nodes must be different";
  DCHECK_LT(from.id, to.id) << "Nodes must be ordered";
  if (from.out_edges.empty() || to.in_edges.empty()) {
    DCHECK_EQ(absl::c_count(from.out_edges, to.id), 0) << "Unexpected out edge";
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }
  if (from.out_edges.back() < to.id || to.in_edges.front() > from.id) {
    DCHECK_EQ(absl::c_count(from.out_edges, to.id), 0) << "Unexpected out edge";
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }
  auto out_edges_it = absl::c_lower_bound(from.out_edges, to.id);
  bool has_out_edge =
      out_edges_it != from.out_edges.end() && *out_edges_it == to.id;
  if (!has_out_edge) {
    DCHECK_EQ(absl::c_count(to.in_edges, from.id), 0) << "Unexpected in edge";
    return 0;
  }
  auto in_edges_it = absl::c_lower_bound(to.in_edges, from.id);
  bool has_in_edge =
      in_edges_it != to.in_edges.end() && *in_edges_it == from.id;
  DCHECK(has_in_edge) << "In-edge must exist if out-edge exists";
  from.out_edges.erase(out_edges_it);
  to.in_edges.erase(in_edges_it);
  return 1;
}
int64_t ThunkExecutor::RunTransitiveReductionAndUpdatePriorities() {
  int64_t num_erased_edges = 0;
  std::vector<int64_t> stack;
  std::vector<bool> visited;
  auto add_to_stack = [&](int64_t node_id) {
    if (!visited[node_id]) {
      stack.push_back(node_id);
      visited[node_id] = true;
    }
  };
  for (int64_t i = nodes_defs_.size() - 1; i >= 0; --i) {
    NodeDef& source_node = nodes_defs_[i];
    stack.clear();
    visited.assign(nodes_defs_.size(), false);
    for (int64_t out_id : source_node.out_edges) {
      NodeDef& out_node = nodes_defs_[out_id];
      visited[out_id] = true;
      for (int64_t start_id : out_node.out_edges) add_to_stack(start_id);
    }
    while (!stack.empty()) {
      int64_t node_id = stack.back();
      stack.pop_back();
      NodeDef& node = nodes_defs_[node_id];
      num_erased_edges += EraseEdge(source_node, node);
      for (int64_t out_id : node.out_edges) add_to_stack(out_id);
    }
    source_node.priority = absl::c_count(visited, true);
  }
  return num_erased_edges;
}
std::string ThunkExecutor::ToString() const {
  std::string str = absl::StrFormat(
      "ThunkExecutor: #thunks=%d #source_nodes=%d #sink_nodes=%d", num_thunks_,
      source_.size(), sink_.size());
  std::vector<std::vector<std::string>> in_edges(num_thunks_);
  for (const auto& node_def : nodes_defs_) {
    for (NodeId in_edge : node_def.in_edges) {
      in_edges[node_def.id].push_back(thunk_sequence_[in_edge]->info().op_name);
    }
  }
  for (NodeId i = 0; i < num_thunks_; ++i) {
    const Thunk& thunk = *thunk_sequence_[i];
    bool is_source = absl::c_find(source_, i) != source_.end();
    bool is_sink = absl::c_find(sink_, i) != sink_.end();
    absl::StrAppendFormat(&str,
                          "\n thunk #%05d: op_name=%s, dependencies=[%s], "
                          "source=%v, sink=%v, priority=%d",
                          i, thunk.info().op_name,
                          absl::StrJoin(in_edges[i], ", "), is_source, is_sink,
                          nodes_defs_[i].priority);
  }
  return str;
}
ThunkExecutor::FifoReadyQueue::FifoReadyQueue(
    absl::Span<const NodeId> ready_nodes)
    : queue_(ready_nodes.begin(), ready_nodes.end()) {}
void ThunkExecutor::FifoReadyQueue::Push(NodeId id) { queue_.push_back(id); }
ThunkExecutor::NodeId ThunkExecutor::FifoReadyQueue::Pop() {
  DCHECK(!Empty()) << "Queue must not be empty";
  return queue_[head_++];
}
ThunkExecutor::FifoReadyQueue ThunkExecutor::FifoReadyQueue::PopHalf() {
  DCHECK(!Empty()) << "Queue must not be empty";
  auto mid = queue_.begin() + head_ + Size() / 2;
  FifoReadyQueue popped(absl::MakeConstSpan(&*mid, queue_.end() - mid));
  queue_.resize(mid - queue_.begin());
  return popped;
}
size_t ThunkExecutor::FifoReadyQueue::Size() const {
  return queue_.size() - head_;
}
bool ThunkExecutor::FifoReadyQueue::Empty() const {
  return head_ == queue_.size();
}
ThunkExecutor::FifoReadyQueue
ThunkExecutor::FifoReadyQueue::CreateEmptyReadyQueue() const {
  return FifoReadyQueue(absl::Span<const NodeId>());
}
ThunkExecutor::PriorityReadyQueue::PriorityReadyQueue(
    absl::Span<const NodeDef> nodes_defs, absl::Span<const NodeId> ready_nodes)
    : nodes_defs_(nodes_defs),
      queue_(ready_nodes.begin(), ready_nodes.end(), Compare{nodes_defs}) {}
void ThunkExecutor::PriorityReadyQueue::Push(NodeId id) { queue_.push(id); }
ThunkExecutor::NodeId ThunkExecutor::PriorityReadyQueue::Pop() {
  DCHECK(!Empty()) << "Queue must not be empty";
  NodeId id = queue_.top();
  queue_.pop();
  return id;
}
ThunkExecutor::PriorityReadyQueue ThunkExecutor::PriorityReadyQueue::PopHalf() {
  DCHECK(!Empty()) << "Queue must not be empty";
  int64_t keep_top_nodes = queue_.size() / 2;
  PriorityReadyQueue popped(nodes_defs_, {});
  while (keep_top_nodes-- > 0) {
    popped.queue_.push(queue_.top());
    queue_.pop();
  }
  popped.queue_.swap(queue_);
  return popped;
}
size_t ThunkExecutor::PriorityReadyQueue::Size() const { return queue_.size(); }
bool ThunkExecutor::PriorityReadyQueue::Empty() const { return queue_.empty(); }
ThunkExecutor::PriorityReadyQueue
ThunkExecutor::PriorityReadyQueue::CreateEmptyReadyQueue() const {
  return PriorityReadyQueue(nodes_defs_, {});
}
}  