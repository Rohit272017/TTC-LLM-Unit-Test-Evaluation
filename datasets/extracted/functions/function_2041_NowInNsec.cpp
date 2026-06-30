#include "tensorflow/core/common_runtime/executor.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <utility>
#include <vector>
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/activity_watcher/activity.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/propagator_state.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/simple_propagator_state.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tsl/platform/tracing.h"
namespace tensorflow {
namespace {
static const Tensor* const kEmptyTensor = new Tensor;
namespace nodestats {
inline int64_t NowInNsec() { return EnvTime::NowNanos(); }
void SetScheduled(NodeExecStatsInterface* stats, int64_t micros) {
  if (!stats) return;
  stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
}
void SetAllStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorStarted();
}
void SetOpStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeStarted();
}
void SetOpEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeEnded();
}
void SetAllEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorEnded();
}
void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
  if (!stats) return;
  stats->SetOutput(slot, v);
}
void SetMemory(NodeExecStatsInterface* stats, OpKernelContext* ctx) {
  if (!stats) return;
  stats->SetMemory(ctx);
}
}  
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();
  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};
typedef absl::InlinedVector<TensorValue, 4UL> TensorValueVec;
typedef absl::InlinedVector<AllocatorAttributes, 4UL> AllocatorAttributeVec;
class ExecutorImpl : public Executor {
 public:
  explicit ExecutorImpl(const LocalExecutorParams& p) : immutable_state_(p) {}
  Status Initialize(const Graph& graph) {
    TF_RETURN_IF_ERROR(immutable_state_.Initialize(graph));
    kernel_stats_.Initialize(immutable_state_.graph_view());
    return absl::OkStatus();
  }
 private:
  void RunAsyncInternal(const Args& args, DoneCallback done) override;
  template <class PropagatorStateType>
  friend class ExecutorState;
  class KernelStats {
   public:
    KernelStats() = default;
    void Initialize(const GraphView& gview) {
      is_expensive_.resize(gview.num_nodes());
      cost_estimates_ =
          std::make_unique<std::atomic_uint_fast64_t[]>(gview.num_nodes());
      for (int32_t i = 0; i < gview.num_nodes(); ++i) {
        if (gview.node(i)) {
          is_expensive_[i] =
              gview.node(i)->kernel && gview.node(i)->kernel->IsExpensive();
          cost_estimates_[i] = kInitialCostEstimateCycles;
        }
      }
    }
    bool IsExpensive(const NodeItem& node) const {
      return is_expensive_[node.node_id] &&
             (cost_estimates_[node.node_id].load(std::memory_order_relaxed) >
              kOpIsExpensiveThresholdCycles);
    }
    bool HasExpensiveMarker(const NodeItem& node) const {
      return is_expensive_[node.node_id];
    }
    void UpdateCostEstimate(const NodeItem& node, uint64 elapsed_cycles) {
      std::atomic_uint_fast64_t& cost_estimate = cost_estimates_[node.node_id];
      auto prev_estimate = cost_estimate.load(std::memory_order_relaxed);
      uint64 new_estimate =
          ((kCostDecay - 1) * prev_estimate + elapsed_cycles) / kCostDecay;
      cost_estimate.store(new_estimate, std::memory_order_relaxed);
    }
   private:
    static constexpr uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
    static constexpr uint64 kOpIsExpensiveThresholdCycles = 8000;
    static constexpr uint64 kCostDecay = 10;
    std::vector<bool> is_expensive_;
    std::unique_ptr<std::atomic_uint_fast64_t[]> cost_estimates_;
  };
  ImmutableExecutorState immutable_state_;
  KernelStats kernel_stats_;
  ExecutorImpl(const ExecutorImpl&) = delete;
  void operator=(const ExecutorImpl&) = delete;
};
template <class PropagatorStateType>
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args,
                const ImmutableExecutorState& immutable_state_,
                ExecutorImpl::KernelStats* kernel_stats_);
  ~ExecutorState();
  void RunAsync(Executor::DoneCallback done);
 private:
  typedef typename PropagatorStateType::TaggedNode TaggedNode;
  typedef
      typename PropagatorStateType::TaggedNodeReadyQueue TaggedNodeReadyQueue;
  typedef typename PropagatorStateType::TaggedNodeSeq TaggedNodeSeq;
  struct AsyncState;
  void Process(const TaggedNode& node, int64_t scheduled_nsec);
  void ProcessInline(TaggedNodeReadyQueue* inline_ready,
                     int64_t scheduled_nsec);
  Status ProcessSync(const NodeItem& item, OpKernelContext::Params* params,
                     EntryVector* outputs, NodeExecStatsInterface* stats);
  void ProcessAsync(const NodeItem& item, const OpKernelContext::Params& params,
                    const TaggedNode& tagged_node, Entry* first_input,
                    NodeExecStatsInterface* stats,
                    activity_watcher::ActivityId activity_id);
  void ProcessNoop(NodeExecStatsInterface* stats);
  void ProcessConstTensor(const NodeItem& item, EntryVector* outputs,
                          NodeExecStatsInterface* stats);
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        Entry* outputs, NodeExecStatsInterface* stats);
  bool NodeDone(const Status& s, TaggedNodeSeq* ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);
  void ScheduleReady(TaggedNodeSeq* ready, TaggedNodeReadyQueue* inline_ready);
  template <typename Closure>
  void RunTask(Closure&& c, int sample_rate = 0);
  void Finish();
  void ScheduleFinish();
  DeviceContext* device_context_ = nullptr;
  const bool vlog_;  
  const bool log_memory_;
  int64_t step_id_;
  int64_t trace_id_;  
  int64_t start_time_usecs_ = 0;
  absl::optional<absl::Time> deadline_;
  static constexpr uint64 kInlineScheduleReadyThreshold = 500;
  RendezvousInterface* rendezvous_;
  CollectiveExecutor* collective_executor_ = nullptr;
  const ConfigProto* const session_config_;
  SessionState* session_state_;
  string session_handle_;
  const SessionMetadata* session_metadata_ = nullptr;
  TensorStore* tensor_store_;
  ScopedStepContainer* step_container_;
  StepStatsCollectorInterface* const stats_collector_;
  const tsl::tracing::EventCollector* const event_collector_;
  Context context_;
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
  const ImmutableExecutorState& immutable_state_;
  ExecutorImpl::KernelStats* const kernel_stats_;
  CancellationManager* cancellation_manager_;
  tsl::CoordinationServiceAgent* coordination_service_agent_;
  absl::optional<ManagedStackTrace> stack_trace_ = absl::nullopt;
  std::unique_ptr<DeviceBase> user_device_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;
  const bool run_all_kernels_inline_;
  PropagatorStateType propagator_;
  Executor::DoneCallback done_cb_;
  std::atomic_int_fast32_t num_outstanding_ops_;
  mutex num_deferred_ops_mu_;
  int64_t num_deferred_ops_ TF_GUARDED_BY(num_deferred_ops_mu_) = 0;
  bool finish_when_deferred_ops_done_ TF_GUARDED_BY(num_deferred_ops_mu_) =
      false;
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
};
template <class PropagatorStateType>
ExecutorState<PropagatorStateType>::ExecutorState(
    const Executor::Args& args, const ImmutableExecutorState& immutable_state,
    ExecutorImpl::KernelStats* kernel_stats)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      trace_id_(args.function_trace_id ? *args.function_trace_id : step_id_),
      start_time_usecs_(args.start_time_usecs),
      deadline_(args.deadline),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      session_config_(args.session_config),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      session_metadata_(immutable_state.params().session_metadata),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      event_collector_(tsl::tracing::GetEventCollector(
          tsl::tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      immutable_state_(immutable_state),
      kernel_stats_(kernel_stats),
      cancellation_manager_(args.cancellation_manager),
      coordination_service_agent_(args.coordination_service_agent),
      stack_trace_(args.stack_trace),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      run_all_kernels_inline_(args.run_all_kernels_inline),
      propagator_(immutable_state, step_id_, vlog_),
      num_outstanding_ops_(0) {
  if (args.user_intra_op_threadpool != nullptr) {
    Device* device = immutable_state_.params().device;
    user_device_ = RenamedDevice::NewRenamedDevice(
        device->name(), device, false, false, args.user_intra_op_threadpool);
  }
}
template <class PropagatorStateType>
ExecutorState<PropagatorStateType>::~ExecutorState() {
  if (device_context_) {
    device_context_->Unref();
  }
  delete slice_reader_cache_;
}
template <class PropagatorStateType>
template <typename Closure>
void ExecutorState<PropagatorStateType>::RunTask(Closure&& c, int sample_rate) {
  alignas(64) static std::atomic<int64_t> num_enqueue_ops{0};
  alignas(64) static std::atomic<int64_t> num_dequeue_ops{0};
  auto n_enqueues = num_enqueue_ops.fetch_add(1, std::memory_order_relaxed);
  if (n_enqueues % std::max(16, sample_rate) == 0) {
    auto n_dequeues = num_dequeue_ops.load(std::memory_order_relaxed);
    metrics::UpdateGraphPendingQueueLength(n_enqueues - n_dequeues);
  }
  runner_([c = std::forward<Closure>(c)]() mutable {
    num_dequeue_ops.fetch_add(1, std::memory_order_relaxed);
    std::forward<Closure>(c)();
  });
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::RunAsync(Executor::DoneCallback done) {
  TaggedNodeSeq ready;
  Device* device = immutable_state_.params().device;
  const Status get_context_status =
      device->TryGetDeviceContext(&device_context_);
  if (!get_context_status.ok()) {
    delete this;
    done(get_context_status);
    return;
  }
  ready.reserve(immutable_state_.root_nodes().size());
  propagator_.ActivateRoots(immutable_state_.root_nodes(), &ready);
  num_outstanding_ops_ = ready.size();
  if (ready.empty()) {
    delete this;
    done(absl::OkStatus());
  } else {
    done_cb_ = std::move(done);
    ScheduleReady(&ready, nullptr);
  }
}
template <class PropagatorStateType>
struct ExecutorState<PropagatorStateType>::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsInterface* _stats)
      : saved_inputs(p.inputs.begin(), p.inputs.end()),
        saved_input_alloc_attrs(p.input_alloc_attrs.begin(),
                                p.input_alloc_attrs.end()),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
        stats(_stats) {
    params.inputs = saved_inputs;
    params.input_alloc_attrs = saved_input_alloc_attrs;
  }
  TensorValueVec saved_inputs;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  const NodeItem* item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStatsInterface* stats;
 private:
  OpKernelContext::Params* ParamsButClearingEigenGPUDevice(
      OpKernelContext::Params* p) {
    p->eigen_gpu_device = nullptr;  
    return p;
  }
};
bool MightTrace(const tsl::tracing::EventCollector* event_collector,
                bool is_expensive) {
  if (event_collector != nullptr) {
    return true;
  }
  if (tsl::profiler::ScopedAnnotation::IsEnabled()) return true;
  return tsl::profiler::TraceMe::Active(
      tsl::profiler::GetTFTraceMeLevel(is_expensive));
}
template <class PropagatorStateType>
Status ExecutorState<PropagatorStateType>::ProcessSync(
    const NodeItem& item, OpKernelContext::Params* params, EntryVector* outputs,
    NodeExecStatsInterface* stats) {
  Status s;
  OpKernelContext ctx(params, item.num_outputs);
  nodestats::SetOpStart(stats);
  OpKernel* op_kernel = item.kernel;
  Device* device = immutable_state_.params().device;
  const bool is_expensive = kernel_stats_->IsExpensive(item);
  if (TF_PREDICT_FALSE(MightTrace(event_collector_, is_expensive))) {
    tsl::tracing::ScopedRegion region(tsl::tracing::EventCategory::kCompute,
                                      op_kernel->name_view());
    profiler::AnnotatedTraceMe activity(
        [op_kernel, &ctx] {
          return op_kernel->TraceString(
              ctx, tsl::profiler::TfOpDetailsEnabled());
        },
        tsl::profiler::GetTFTraceMeLevel(is_expensive));
    device->Compute(op_kernel, &ctx);
  } else if (kernel_stats_->HasExpensiveMarker(item)) {
    KernelTimer timer;
    device->Compute(op_kernel, &ctx);
    constexpr int kKernelExecutionTrackingInvocationSkipCount = 16;
    if (is_expensive ||
        timer.start_cycles % kKernelExecutionTrackingInvocationSkipCount == 0) {
      kernel_stats_->UpdateCostEstimate(item, timer.ElapsedCycles());
    }
  } else {
    device->Compute(op_kernel, &ctx);
  }
  nodestats::SetOpEnd(stats);
  if (outputs->size() < item.num_outputs) outputs->resize(item.num_outputs);
  s = ProcessOutputs(item, &ctx, outputs->data(), stats);
  nodestats::SetMemory(stats, &ctx);
  return s;
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessAsync(
    const NodeItem& item, const OpKernelContext::Params& params,
    const TaggedNode& tagged_node, Entry* first_input,
    NodeExecStatsInterface* stats, activity_watcher::ActivityId activity_id) {
  AsyncOpKernel* async_kernel = item.kernel->AsAsync();
  DCHECK(async_kernel != nullptr);
  AsyncState* state =
      new AsyncState(params, tagged_node, &item, first_input, stats);
  nodestats::SetOpStart(stats);
  {
    profiler::AnnotatedTraceMe activity(
        [async_kernel, state] {
          return async_kernel->TraceString(
              state->ctx, tsl::profiler::TfOpDetailsEnabled());
        },
        tsl::profiler::GetTFTraceMeLevel(false));
    tsl::profiler::TraceMeProducer producer(
        [&] {
          return tsl::profiler::TraceMeEncode(
              "ExecutorState::ProcessAsync::Start",
              {{"name", async_kernel->name()},
               {"kernel_type", async_kernel->type_string()},
               {"step_id", step_id_}});
        },
        tsl::profiler::ContextType::kTfExecutor);
    auto done = [this, state, activity_id, ctx_id = producer.GetContextId()]() {
      tsl::profiler::TraceMeConsumer consumer(
          [&] {
            return profiler::TraceMeEncode(
                "ExecutorState::ProcessAsync::Done",
                {{"name", state->item->kernel->name()},
                 {"kernel_type", state->item->kernel->type_string()},
                 {"step_id", step_id_}});
          },
          tsl::profiler::ContextType::kTfExecutor, ctx_id);
      Device* device = immutable_state_.params().device;
      NodeExecStatsInterface* stats = state->stats;  
      Entry* first_input = state->first_input;       
      nodestats::SetOpEnd(stats);
      EntryVector outputs(state->item->num_outputs);
      Status s =
          ProcessOutputs(*state->item, &state->ctx, outputs.data(), stats);
      nodestats::SetMemory(stats, &state->ctx);
      if (vlog_) {
        VLOG(2) << "Async kernel done: " << state->item->node_id << " step "
                << step_id_ << " "
                << SummarizeNodeDef(state->item->kernel->def())
                << (state->tagged_node.get_is_dead() ? " is dead" : "")
                << " device: " << device->name();
      }
      const int num_inputs = state->item->num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      propagator_.MaybeMarkCompleted(state->tagged_node);
      activity_watcher::ActivityEnd(activity_id);
      TaggedNodeSeq ready;
      if (s.ok()) {
        propagator_.PropagateOutputs(state->tagged_node, &outputs, &ready);
      }
      outputs.clear();
      const bool completed = NodeDone(s, &ready, stats, nullptr);
      delete state;
      if (completed) ScheduleFinish();
    };
    immutable_state_.params().device->ComputeAsync(async_kernel, &state->ctx,
                                                   std::move(done));
  }
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessNoop(
    NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessConstTensor(
    const NodeItem& item, EntryVector* outputs, NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
  Entry& output = (*outputs)[0];
  output.state = Entry::State::HAS_CONST_TENSOR;
  output.const_tensor = item.const_tensor;
  output.alloc_attr = item.output_attrs()[0];
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::Process(const TaggedNode& tagged_node,
                                                 int64_t scheduled_nsec) {
  tsl::profiler::TraceMe traceme("ExecutorState::Process Scheduled",
                                 tsl::profiler::TraceMeLevel::kVerbose);
  TaggedNodeReadyQueue inline_ready;
  inline_ready.push_back(tagged_node);
  return ProcessInline(&inline_ready, scheduled_nsec);
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessInline(
    TaggedNodeReadyQueue* inline_ready, int64_t scheduled_nsec) {
  WithContext wc(context_);
  auto ready = std::make_unique<TaggedNodeSeq>();
  auto inputs = std::make_unique<TensorValueVec>();
  AllocatorAttributeVec input_alloc_attrs;
  auto params = std::make_unique<OpKernelContext::Params>();
  params->step_id = step_id_;
  Device* device = immutable_state_.params().device;
  if (user_device_) {
    params->device = user_device_.get();
  } else {
    params->device = device;
  }
  params->start_time_usecs = start_time_usecs_;
  params->deadline = deadline_;
  params->log_memory = log_memory_;
  params->rendezvous = rendezvous_;
  params->collective_executor = collective_executor_;
  params->session_config = session_config_;
  params->session_state = session_state_;
  params->session_handle = session_handle_;
  params->session_metadata = session_metadata_;
  params->tensor_store = tensor_store_;
  params->cancellation_manager = cancellation_manager_;
  params->coordination_service_agent = coordination_service_agent_;
  params->stack_trace = stack_trace_;
  params->call_frame = call_frame_;
  params->function_library = immutable_state_.params().function_library;
  params->resource_manager = device->resource_manager();
  params->step_container = step_container_;
  params->slice_reader_cache = slice_reader_cache_;
  params->runner = &runner_;
  params->run_all_kernels_inline = run_all_kernels_inline_;
  params->stats_collector = stats_collector_;
  params->inc_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_++;
  };
  params->dec_num_deferred_ops_function = [this]() {
    bool finish_when_deferred_ops_done = false;
    {
      mutex_lock lock(num_deferred_ops_mu_);
      num_deferred_ops_--;
      if (num_deferred_ops_ == 0) {
        finish_when_deferred_ops_done = finish_when_deferred_ops_done_;
      }
    }
    if (finish_when_deferred_ops_done) Finish();
  };
  params->op_device_context = device_context_;
  Status s;
  NodeExecStatsInterface* stats = nullptr;
  EntryVector outputs(1);
  bool completed = false;
  int64_t last_iter_num = -1;
  std::unique_ptr<tsl::profiler::TraceMeConsumer> iteration_scope;
  while (!inline_ready->empty()) {
    TaggedNode tagged_node = inline_ready->front();
    int64_t current_iter_num = tagged_node.get_iter_num();
    if (current_iter_num != last_iter_num) {
      iteration_scope = std::make_unique<tsl::profiler::TraceMeConsumer>(
          [&] {
            return profiler::TraceMeEncode(
                "ExecutorState::Process",
                {{"id", step_id_}, {"iter_num", tagged_node.get_iter_num()}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id_,
          tsl::profiler::TraceMeLevel::kInfo);
      last_iter_num = current_iter_num;
    }
    inline_ready->pop_front();
    const NodeItem& item = tagged_node.get_node_item();
    const int id = item.node_id;
    propagator_.MaybeMarkStarted(tagged_node);
    const activity_watcher::ActivityId activity_id =
        activity_watcher::ActivityStart(
            [&]() {
              return std::make_unique<activity_watcher::Activity>(
                  "ExecutorState::Process",
                  activity_watcher::ActivityCategory::kMisc,
                  activity_watcher::Activity::Attributes{
                      {"node_name", item.kernel->def().name()},
                      {"op", item.kernel->def().op()},
                      {"iter_num", absl::StrCat(tagged_node.get_iter_num())},
                      {"step_id", absl::StrCat(params->step_id)},
                      {"node_id", absl::StrCat(id)},
                      {"device", device->name()},
                      {"inputs",
                       absl::StrJoin(item.kernel->def().input(), "; ")},
                      {"original_node_names",
                       absl::StrJoin(item.kernel->def()
                                         .experimental_debug_info()
                                         .original_node_names(),
                                     "; ")},
                      {"original_func_names",
                       absl::StrJoin(item.kernel->def()
                                         .experimental_debug_info()
                                         .original_func_names(),
                                     "; ")},
                  });
            },
            2);
    params->track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !tagged_node.get_is_dead()) {
      stats = stats_collector_->CreateNodeExecStats(&item.kernel->def());
      params->track_allocations = stats ? stats->TrackAllocations() : false;
      nodestats::SetScheduled(stats, scheduled_nsec);
      nodestats::SetAllStart(stats);
    }
    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params->step_id << " "
              << SummarizeNodeDef(item.kernel->def())
              << (tagged_node.get_is_dead() ? " is dead" : "")
              << " device: " << device->name();
    }
    Entry* first_input = propagator_.GetInputTensors(tagged_node);
    bool launched_asynchronously = false;
    if (tagged_node.get_is_dead() && !item.is_transfer_node) {
      if (outputs.size() < item.num_outputs) outputs.resize(item.num_outputs);
    } else if (TF_PREDICT_FALSE(item.is_noop)) {
      ProcessNoop(stats);
    } else if (item.const_tensor != nullptr && !params->track_allocations) {
      ProcessConstTensor(item, &outputs, stats);
    } else {
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, inputs.get(), &input_alloc_attrs,
                        &is_input_dead);
      if (!s.ok()) {
        const int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        propagator_.MaybeMarkCompleted(tagged_node);
        activity_watcher::ActivityEnd(activity_id);
        completed = NodeDone(s, ready.get(), stats, inline_ready);
        continue;
      }
      params->op_kernel = item.kernel;
      params->frame_iter = propagator_.GetFrameAndIter(tagged_node);
      params->is_input_dead = is_input_dead;
      params->output_attr_array = item.output_attrs();
      params->forward_from_array = item.forward_from();
      params->outputs_required_array = item.outputs_required.get();
      params->inputs = *inputs;
      params->input_alloc_attrs = input_alloc_attrs;
      if (item.kernel_is_async) {
        ProcessAsync(item, *params, tagged_node, first_input, stats,
                     activity_id);
        launched_asynchronously = true;
      } else {
        s = ProcessSync(item, params.get(), &outputs, stats);
      }
    }
    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params->step_id << " "
                << SummarizeNodeDef(item.kernel->def())
                << (tagged_node.get_is_dead() ? " is dead: " : "")
                << " device: " << device->name();
      }
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      propagator_.MaybeMarkCompleted(tagged_node);
      activity_watcher::ActivityEnd(activity_id);
      if (s.ok()) {
        propagator_.PropagateOutputs(tagged_node, &outputs, ready.get());
      }
      const int num_outputs = item.num_outputs;
      for (int i = 0; i < num_outputs; ++i) {
        outputs[i].ClearVal();
      }
      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }
      completed = NodeDone(s, ready.get(), stats, inline_ready);
    }
  }  
  if (completed) ScheduleFinish();
}
template <class PropagatorStateType>
Status ExecutorState<PropagatorStateType>::PrepareInputs(
    const NodeItem& item, Entry* first_input, TensorValueVec* inputs,
    AllocatorAttributeVec* input_alloc_attrs, bool* is_input_dead) {
  inputs->resize(item.num_inputs);
  input_alloc_attrs->resize(item.num_inputs);
  *is_input_dead = false;
  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = TF_PREDICT_FALSE(item.is_any_input_ref_typed) &&
                            IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_alloc_attrs)[i] = entry->alloc_attr;
    TensorValue* inp = &(*inputs)[i];
    switch (entry->state) {
      case Entry::State::NO_VALUE: {
        inp->mutex_if_ref = nullptr;
        if (item.is_merge) {
          inp->tensor = nullptr;
        } else {
          DCHECK(item.is_transfer_node)
              << item.kernel->name() << " - input " << i;
          entry->state = Entry::State::HAS_CONST_TENSOR;
          entry->const_tensor = kEmptyTensor;
          inp->tensor = const_cast<Tensor*>(kEmptyTensor);
          *is_input_dead = true;
        }
        break;
      }
      case Entry::State::HAS_VALUE: {
        if (TF_PREDICT_FALSE(expect_ref)) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        inp->mutex_if_ref = nullptr;
        inp->tensor = entry->val.get();
        break;
      }
      case Entry::State::HAS_CONST_TENSOR: {
        if (TF_PREDICT_FALSE(expect_ref)) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        inp->mutex_if_ref = nullptr;
        inp->tensor = const_cast<Tensor*>(entry->const_tensor);
        break;
      }
      case Entry::State::HAS_REF_TENSOR: {
        {
          tf_shared_lock ml(*entry->ref_tensor.mu);
          if (TF_PREDICT_FALSE(!entry->ref_tensor.tensor->IsInitialized() &&
                               !item.is_initialization_op)) {
            return AttachDef(errors::FailedPrecondition(
                                 "Attempting to use uninitialized value ",
                                 item.kernel->requested_input(i)),
                             item.kernel->def());
          }
        }
        if (expect_ref) {
          inp->mutex_if_ref = entry->ref_tensor.mu;
          inp->tensor = entry->ref_tensor.tensor;
        } else {
          {
            mutex* ref_mu = entry->ref_tensor.mu;
            Tensor* ref_tensor = entry->ref_tensor.tensor;
            tf_shared_lock l(*ref_mu);
            entry->val.Init(*ref_tensor);
          }
          entry->state = Entry::State::HAS_VALUE;
          inp->mutex_if_ref = nullptr;
          inp->tensor = entry->val.get();
          if (TF_PREDICT_FALSE(item.input_type(i) != inp->tensor->dtype())) {
            return AttachDef(
                errors::InvalidArgument(
                    i, "-th input expects type ",
                    DataTypeString(item.input_type(i)),
                    " but automatically dereferenced input tensor has type ",
                    DataTypeString(inp->tensor->dtype())),
                item.kernel->def());
          }
        }
        break;
      }
    }
  }
  return absl::OkStatus();
}
template <class PropagatorStateType>
Status ExecutorState<PropagatorStateType>::ProcessOutputs(
    const NodeItem& item, OpKernelContext* ctx, Entry* outputs,
    NodeExecStatsInterface* stats) {
  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err =
            stats_collector_->ReportAllocsOnResourceExhausted(s.message());
        s = errors::CreateWithUpdatedMessage(s,
                                             strings::StrCat(s.message(), err));
      } else {
        s = errors::CreateWithUpdatedMessage(
            s,
            strings::StrCat(
                s.message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info. This isn't "
                "available when running in Eager mode.\n"));
      }
    } else if (s.code() == error::UNAVAILABLE &&
               !item.is_distributed_communication) {
      s = errors::ReplaceErrorFromNonCommunicationOps(s, item.kernel->name());
    }
    return ADD_SOURCE_LOCATION(s);
  }
  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);
    Entry* out = &outputs[i];
    DCHECK(out->state == Entry::State::NO_VALUE);
    if (val.tensor == nullptr) {
      if (!(item.is_recv_or_switch ||
            (item.outputs_required && !item.outputs_required[i]))) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeDefForError(item.kernel->def())));
      }
    } else {
      out->alloc_attr = ctx->output_alloc_attr(i);
      DataType dtype = val.dtype_safe();
      if (dtype == item.output_type(i)) {
        if (stats && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
        if (val.is_ref()) {
          out->state = Entry::State::HAS_REF_TENSOR;
          out->ref_tensor.tensor = val.tensor;
          out->ref_tensor.mu = val.mutex_if_ref;
          if (log_memory_) {
            Tensor to_log;
            {
              tf_shared_lock l(*out->ref_tensor.mu);
              to_log = *out->ref_tensor.tensor;
            }
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
          }
        } else {
          out->state = Entry::State::HAS_VALUE;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }
        }
      } else {
        s.Update(
            errors::Internal("Output ", i, " of type ", DataTypeString(dtype),
                             " does not match declared output type ",
                             DataTypeString(item.output_type(i)), " for node ",
                             FormatNodeDefForError(item.kernel->def())));
      }
    }
    if (!val.is_ref()) {
      delete val.tensor;
    }
  }
  return s;
}
template <class PropagatorStateType>
bool ExecutorState<PropagatorStateType>::NodeDone(
    const Status& s, TaggedNodeSeq* ready, NodeExecStatsInterface* stats,
    TaggedNodeReadyQueue* inline_ready) {
  if (stats) {
    nodestats::SetAllEnd(stats);
    DCHECK_NE(stats_collector_, nullptr);
    stats->Done(immutable_state_.params().device->name());
  }
  if (TF_PREDICT_TRUE(s.ok())) {
    const size_t ready_size = ready->size();
    if (ready_size == 0) {
      return num_outstanding_ops_.fetch_sub(1) == 1;
    } else {
      if (ready_size > 1) {
        num_outstanding_ops_.fetch_add(ready_size - 1,
                                       std::memory_order_relaxed);
      }
      ScheduleReady(ready, inline_ready);
      return false;
    }
  } else {
    bool abort_run = false;
    Status maybe_derived_s(s);
    {
      mutex_lock l(mu_);
      if (status_.ok()) {
        abort_run = true;
        if (cancellation_manager_ && cancellation_manager_->IsCancelled() &&
            (errors::IsCancelled(s) || errors::IsAborted(s))) {
          status_ = StatusGroup::MakeDerived(s);
          maybe_derived_s = status_;
        } else {
          status_ = s;
        }
      }
    }
    if (abort_run) {
      TRACEPRINTF("StartAbort: %s", s.ToString());
      if (cancellation_manager_) {
        VLOG(1) << "[" << immutable_state_.params().device->name()
                << "] Executor start aborting: " << s;
      }
      if (rendezvous_) {
        rendezvous_->StartAbort(s);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancelWithStatus(maybe_derived_s);
      } else if (collective_executor_) {
        collective_executor_->StartAbort(s);
      }
    }
    return num_outstanding_ops_.fetch_sub(1) == 1;
  }
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ScheduleReady(
    TaggedNodeSeq* ready, TaggedNodeReadyQueue* inline_ready) {
  tsl::profiler::TraceMe activity(
      [&]() {
        return strings::StrCat(
            "ExecutorState::ScheduleReady#",
            "ready_size=", (ready == nullptr ? -1 : ready->size()),
            ",inline_ready_size=",
            (inline_ready == nullptr ? -1 : inline_ready->size()), "#");
      },
      tsl::profiler::GetTFTraceMeLevel(false));
  DCHECK(!ready->empty());
  int64_t scheduled_nsec = 0;
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }
  if (run_all_kernels_inline_) {
    if (inline_ready == nullptr) {
      RunTask([this, ready = std::move(*ready), scheduled_nsec]() {
        for (auto& tagged_node : ready) {
          Process(tagged_node, scheduled_nsec);
        }
      });
    } else {
      for (auto& tagged_node : *ready) {
        inline_ready->push_back(tagged_node);
      }
    }
  } else {
    const TaggedNode* curr_expensive_node = nullptr;
    TaggedNodeSeq expensive_nodes;
    if (inline_ready == nullptr) {
      for (auto& tagged_node : *ready) {
        RunTask([=]() { Process(tagged_node, scheduled_nsec); },
                ready->size());
      }
    } else {
      for (auto& tagged_node : *ready) {
        const NodeItem& item = *tagged_node.node_item;
        if (tagged_node.get_is_dead() || !kernel_stats_->IsExpensive(item)) {
          inline_ready->push_back(tagged_node);
        } else {
          if (curr_expensive_node) {
            expensive_nodes.push_back(*curr_expensive_node);
          }
          curr_expensive_node = &tagged_node;
        }
      }
    }
    if (curr_expensive_node) {
      if (inline_ready->empty()) {
        inline_ready->push_back(*curr_expensive_node);
      } else {
        expensive_nodes.push_back(*curr_expensive_node);
      }
    }
    if (!expensive_nodes.empty()) {
      if (expensive_nodes.size() < kInlineScheduleReadyThreshold) {
        for (auto& tagged_node : expensive_nodes) {
          RunTask(std::bind(&ExecutorState::Process, this, tagged_node,
                            scheduled_nsec),
                  expensive_nodes.size());
        }
      } else {
        auto it = expensive_nodes.begin();
        while (it < expensive_nodes.end()) {
          auto end = it;
          std::advance(end, kInlineScheduleReadyThreshold);
          if (end > expensive_nodes.end()) {
            end = expensive_nodes.end();
          }
          TaggedNodeSeq ready_chunk{it, end};
          RunTask(
              [this, ready_chunk = std::move(ready_chunk), scheduled_nsec]() {
                tsl::profiler::TraceMe activity(
                    [&]() {
                      return strings::StrCat(
                          "ExecutorState::ScheduleReady::"
                          "ChildThreadExpensiveNodes#",
                          "ready_chunk_size=", ready_chunk.size(), "#");
                    },
                    tsl::profiler::GetTFTraceMeLevel(false));
                for (auto& tagged_node : ready_chunk) {
                  RunTask(std::bind(&ExecutorState::Process, this, tagged_node,
                                    scheduled_nsec),
                          ready_chunk.size());
                }
              });
          it = end;
        }
      }
    }
  }
  ready->clear();
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ScheduleFinish() {
  {
    mutex_lock lock(num_deferred_ops_mu_);
    if (num_deferred_ops_ > 0) {
      finish_when_deferred_ops_done_ = true;
      return;
    }
  }
  Finish();
}
template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();
  int64_t trace_id = trace_id_;
  int64_t step_id = step_id_;
  CHECK(done_cb != nullptr);
  Device* device = immutable_state_.params().device;
  if (vlog_ && !status.ok() && VLOG_IS_ON(1)) {
    propagator_.DumpState();
  }
  if (!device->AllowsSyncOnCompletion()) {
    status.Update(device->RefreshStatus());
    if (!status.ok()) {
      if (rendezvous_) {
        rendezvous_->StartAbort(status);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancelWithStatus(status);
      } else if (collective_executor_) {
        collective_executor_->StartAbort(status);
      }
    }
    delete this;
    runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
      tsl::profiler::TraceMeConsumer activity(
          [&] {
            return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                {{"id", step_id}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id,
          tsl::profiler::TraceMeLevel::kInfo);
      done_cb(status);
    });
    return;
  }
  if (sync_on_finish_ && status.ok()) {
    device->Sync([this, step_id, trace_id, runner = std::move(runner),
                  done_cb = std::move(done_cb)](const Status& status) mutable {
      delete this;
      runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
        tsl::profiler::TraceMeConsumer activity(
            [&] {
              return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                  {{"id", step_id}});
            },
            tsl::profiler::ContextType::kTfExecutor, trace_id,
            tsl::profiler::TraceMeLevel::kInfo);
        done_cb(status);
      });
    });
  } else {
    delete this;
    runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
      tsl::profiler::TraceMeConsumer activity(
          [&] {
            return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                {{"id", step_id}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id,
          tsl::profiler::TraceMeLevel::kInfo);
      done_cb(status);
    });
  }
}
void ExecutorImpl::RunAsyncInternal(const Args& args, DoneCallback done) {
  if (OpOrderDeterminismRequired()) {
    (new ExecutorState<OrderedPropagatorState>(args, immutable_state_,
                                               &kernel_stats_))
        ->RunAsync(std::move(done));
  } else if (immutable_state_.requires_control_flow_support()) {
    (new ExecutorState<PropagatorState>(args, immutable_state_, &kernel_stats_))
        ->RunAsync(std::move(done));
  } else {
    (new ExecutorState<SimplePropagatorState>(args, immutable_state_,
                                              &kernel_stats_))
        ->RunAsync(std::move(done));
  }
}
}  
Status NewLocalExecutor(const LocalExecutorParams& params, const Graph& graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params);
  const Status s = impl->Initialize(graph);
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const std::shared_ptr<const NodeProperties>& props,
                             int graph_def_version, OpKernel** kernel) {
  const auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib,
                        device->resource_manager(), props, graph_def_version,
                        kernel);
}
void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }
namespace {
class DefaultExecutorRegistrar {
 public:
  DefaultExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("", factory);
    ExecutorFactory::Register("DEFAULT", factory);
  }
 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return absl::OkStatus();
    }
  };
};
static DefaultExecutorRegistrar registrar;
}  
}  