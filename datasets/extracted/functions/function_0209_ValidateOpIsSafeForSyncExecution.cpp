#include "tensorflow/core/common_runtime/single_threaded_executor.h"
#include <utility>
#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
namespace tensorflow {
Status ValidateOpIsSafeForSyncExecution(
    const Node& n, bool allow_control_flow_sync_execution) {
  for (DataType dt : n.output_types()) {
    if (IsRefType(dt)) {
      return errors::Unimplemented(
          "Single-threaded executor does not support reference-typed "
          "edges.  But saw type ",
          DataTypeString(dt), " in outputs of node ", n.name());
    }
  }
  if (n.IsSwitch()) {
    return errors::FailedPrecondition(
        "Single-threaded executor does not support switch op, but saw node ",
        n.name(),
        ". Perhaps your graph contains old-style control flow primitives? "
        "Try using tf.compat.v1.enable_control_flow_v2().");
  }
  if (n.IsControlFlow() && !allow_control_flow_sync_execution) {
    return errors::FailedPrecondition(
        "Single-threaded executor does not support low level control flow, "
        " but saw control flow node ",
        n.name(),
        ".  Perhaps your graph contains old-style control flow primitives? "
        "Try using tf.compat.v1.enable_control_flow_v2().");
  }
  return absl::OkStatus();
}
namespace {
typedef absl::InlinedVector<TensorValue, 4UL> TensorValueVec;
typedef absl::InlinedVector<AllocatorAttributes, 4UL> AllocatorAttributeVec;
static const string& kSingleThreadedExecutor =
    *new string("SINGLE_THREADED_EXECUTOR");
class SingleThreadedExecutorImpl : public Executor {
 public:
  explicit SingleThreadedExecutorImpl(const LocalExecutorParams& params)
      : params_(params) {}
  ~SingleThreadedExecutorImpl() override {
    for (const KernelState& kernel_state : kernels_) {
      params_.delete_kernel(kernel_state.kernel);
    }
    for (const ConstTensorKernelState& kernel_state : const_tensor_kernels_) {
      params_.delete_kernel(kernel_state.kernel);
    }
  }
  Status Initialize(const Graph& graph) {
    std::vector<Node*> ordered_nodes;
    ordered_nodes.reserve(graph.num_nodes());
    GetReversePostOrder(graph, &ordered_nodes);
    int ordered_nodes_size = ordered_nodes.size();
    if (ordered_nodes_size != graph.num_nodes()) {
      return errors::InvalidArgument("Graph had ", graph.num_nodes(),
                                     " but reverse post-order had ",
                                     ordered_nodes.size());
    }
    kernels_.reserve(ordered_nodes.size() - 2);
    std::vector<Node*> nodes_with_kernels;
    std::vector<Node*> nodes_with_const_tensor_kernels;
    nodes_with_kernels.reserve(ordered_nodes.size() - 2);
    std::map<size_t, Node*> arg_index_to_node_map;
    absl::flat_hash_map<Node*, size_t> node_to_index_map;
    for (Node* n : ordered_nodes) {
      if (n->IsSource() || n->IsSink()) {
        continue;
      }
      TF_RETURN_IF_ERROR(ValidateOpIsSafeForSyncExecution(
          *n, params_.allow_control_flow_sync_execution));
      if (n->IsArg()) {
        int32_t arg_index;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &arg_index));
        if (arg_index < 0) {
          return errors::InvalidArgument("Invalid argument index ", arg_index,
                                         " in node ", n->name());
        }
        arg_index_to_node_map[arg_index] = n;
        continue;
      }
      OpKernel* kernel;
      TF_RETURN_IF_ERROR(params_.create_kernel(n->properties(), &kernel));
      const Tensor* const_tensor;
      if (n->num_outputs() == 1 && (const_tensor = kernel->const_tensor())) {
        const size_t kernel_index = const_tensor_kernels_.size();
        const_tensor_kernels_.push_back({});
        nodes_with_const_tensor_kernels.push_back(n);
        ConstTensorKernelState& kernel_state =
            const_tensor_kernels_[kernel_index];
        kernel_state.kernel = kernel;
        kernel_state.const_tensor = *const_tensor;
      } else {
        const size_t kernel_index = kernels_.size();
        kernels_.push_back({});
        nodes_with_kernels.push_back(n);
        KernelState& kernel_state = kernels_[kernel_index];
        kernel_state.kernel = kernel;
        kernel_state.num_inputs = n->num_inputs();
        kernel_state.num_outputs = n->num_outputs();
        node_to_index_map[n] = kernel_index;
        if (kernel_index == 0) {
          kernel_state.input_start_index = 0;
        } else {
          const KernelState& previous_kernel_state = kernels_[kernel_index - 1];
          kernel_state.input_start_index =
              previous_kernel_state.input_start_index +
              previous_kernel_state.num_inputs;
        }
      }
    }
    if (!arg_index_to_node_map.empty()) {
      const size_t num_args = arg_index_to_node_map.rbegin()->first + 1;
      arg_output_locations_.resize(num_args);
      for (const auto& arg_index_node_pair : arg_index_to_node_map) {
        const size_t arg_index = arg_index_node_pair.first;
        const Node* arg_node = arg_index_node_pair.second;
        arg_output_locations_[arg_index].reserve(arg_node->out_edges().size());
        for (const Edge* e : arg_node->out_edges()) {
          if (e->src_output() == Graph::kControlSlot) {
            continue;
          } else if (e->src_output() != 0) {
            return errors::Internal("Invalid output index ", e->src_output(),
                                    " from argument node ", arg_index);
          }
          arg_output_locations_[arg_index].push_back(
              kernels_[node_to_index_map[e->dst()]].input_start_index +
              e->dst_input());
        }
      }
    }
    for (size_t i = 0; i < const_tensor_kernels_.size(); ++i) {
      Node* n = nodes_with_const_tensor_kernels[i];
      ConstTensorKernelState& kernel_state = const_tensor_kernels_[i];
      for (const Edge* e : n->out_edges()) {
        if (e->src_output() == Graph::kControlSlot) {
          continue;
        } else if (e->src_output() != 0) {
          return errors::Internal("Invalid output index ", e->src_output(),
                                  " from node ", n->DebugString());
        }
        kernel_state.output_locations.push_back(
            kernels_[node_to_index_map[e->dst()]].input_start_index +
            e->dst_input());
      }
      bool on_host =
          kernel_state.kernel->output_memory_types()[0] == HOST_MEMORY;
      kernel_state.output_alloc_attr.set_on_host(on_host);
    }
    for (size_t i = 0; i < kernels_.size(); ++i) {
      Node* n = nodes_with_kernels[i];
      KernelState& kernel_state = kernels_[i];
      kernel_state.output_locations.resize(kernel_state.num_outputs);
      for (const Edge* e : n->out_edges()) {
        if (!e->IsControlEdge()) {
          kernel_state.output_locations[e->src_output()].push_back(
              kernels_[node_to_index_map[e->dst()]].input_start_index +
              e->dst_input());
        }
      }
      kernel_state.output_alloc_attrs.resize(kernel_state.num_outputs);
      AllocatorAttributes* attrs = kernel_state.output_alloc_attrs.data();
      OpKernel* op_kernel = kernel_state.kernel;
      for (int out = 0; out < n->num_outputs(); out++) {
        DCHECK_LT(out, op_kernel->output_memory_types().size());
        bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
        if (on_host) {
          AllocatorAttributes h;
          h.set_on_host(on_host);
          attrs[out].Merge(h);
        }
      }
    }
    if (!kernels_.empty()) {
      const KernelState& last_kernel_state = kernels_.back();
      total_num_inputs_ =
          last_kernel_state.input_start_index + last_kernel_state.num_inputs;
      input_alloc_attrs_.resize(total_num_inputs_);
      for (size_t i = 0; i < kernels_.size(); ++i) {
        for (size_t j = 0; j < kernels_[i].output_locations.size(); ++j) {
          for (size_t output_location : kernels_[i].output_locations[j]) {
            input_alloc_attrs_[output_location] =
                kernels_[i].output_alloc_attrs[j];
          }
        }
      }
    } else {
      total_num_inputs_ = 0;
    }
    return absl::OkStatus();
  }
  Status Run(const Args& args) override {
    std::vector<Entry> inputs(total_num_inputs_);
    TensorValueVec node_inputs;
    AllocatorAttributeVec input_alloc_attrs;
    Device* device = params_.device;
    std::unique_ptr<Device> user_device;
    if (args.user_intra_op_threadpool != nullptr) {
      user_device = RenamedDevice::NewRenamedDevice(
          device->name(), device, false,
          false, args.user_intra_op_threadpool);
      device = user_device.get();
    }
    OpKernelContext::Params params;
    params.step_id = args.step_id;
    params.device = device;
    params.log_memory = false;  
    params.rendezvous = args.rendezvous;
    params.session_state = args.session_state;
    params.session_metadata = params_.session_metadata;
    params.tensor_store = args.tensor_store;
    params.cancellation_manager = args.cancellation_manager;
    params.session_config = args.session_config;
    params.call_frame = args.call_frame;
    params.function_library = params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = args.step_container;
    params.collective_executor = args.collective_executor;
    params.stack_trace = args.stack_trace;
    params.slice_reader_cache = nullptr;  
    Args::Runner runner_copy = args.runner;
    params.runner = &runner_copy;
    params.run_all_kernels_inline = args.run_all_kernels_inline;
    params.stats_collector = args.stats_collector;
    params.executor_type = &kSingleThreadedExecutor;
    params.frame_iter = FrameAndIter(0, 0);
    params.is_input_dead = false;
    device->TryGetDeviceContext(&params.op_device_context).IgnoreError();
    auto context_cleanup = gtl::MakeCleanup([&params] {
      if (params.op_device_context != nullptr) {
        params.op_device_context->Unref();
      }
    });
    params.forward_from_array = nullptr;
    const size_t received_args =
        args.call_frame ? args.call_frame->num_args() : 0;
    if (TF_PREDICT_FALSE(arg_output_locations_.size() > received_args)) {
      return errors::InvalidArgument("Expected ", arg_output_locations_.size(),
                                     " arguments, but only received ",
                                     received_args, ".");
    }
    for (size_t i = 0; i < arg_output_locations_.size(); ++i) {
      const size_t num_destinations = arg_output_locations_[i].size();
      if (num_destinations > 0) {
        if (args.call_frame->CanConsumeArg(i)) {
          Entry& first_input = inputs[arg_output_locations_[i][0]];
          first_input.state = Entry::State::HAS_VALUE;
          first_input.val.Init();
          args.call_frame->ConsumeArg(i, first_input.val.get());
          for (size_t j = 1; j < num_destinations; ++j) {
            Entry& input = inputs[arg_output_locations_[i][j]];
            input.state = Entry::State::HAS_VALUE;
            input.val.Init(*first_input.val);
          }
        } else {
          const Tensor* arg;
          TF_RETURN_IF_ERROR(args.call_frame->GetArg(i, &arg));
          for (size_t j = 0; j < num_destinations; ++j) {
            Entry& input = inputs[arg_output_locations_[i][j]];
            input.state = Entry::State::HAS_VALUE;
            input.val.Init(*arg);
          }
        }
      }
    }
    for (const ConstTensorKernelState& kernel_state : const_tensor_kernels_) {
      for (size_t i = 0; i < kernel_state.output_locations.size(); ++i) {
        Entry& input = inputs[kernel_state.output_locations[i]];
        input.state = Entry::State::HAS_CONST_TENSOR;
        input.const_tensor = &kernel_state.const_tensor;
      }
    }
    for (size_t i = 0; i < kernels_.size(); ++i) {
      const KernelState& kernel_state = kernels_[i];
      const size_t input_start_index = kernel_state.input_start_index;
      const size_t num_inputs = kernel_state.num_inputs;
      const size_t num_outputs = kernel_state.num_outputs;
      node_inputs.clear();
      node_inputs.resize(num_inputs);
      input_alloc_attrs.clear();
      input_alloc_attrs.resize(num_inputs);
      for (size_t j = 0; j < num_inputs; ++j) {
        Entry& input = inputs[input_start_index + j];
        switch (input.state) {
          case Entry::State::HAS_CONST_TENSOR:
            node_inputs[j].tensor = const_cast<Tensor*>(input.const_tensor);
            break;
          case Entry::State::HAS_VALUE:
            node_inputs[j].tensor = input.val.get();
            break;
          default:
            DCHECK(false) << "Input did not have a valid value.";
        }
        input_alloc_attrs[j] = input_alloc_attrs_[input_start_index + j];
      }
      params.inputs = node_inputs;
      params.input_alloc_attrs = input_alloc_attrs;
      params.op_kernel = kernel_state.kernel;
      params.output_attr_array = kernel_state.output_alloc_attrs.data();
      OpKernelContext ctx(&params, num_outputs);
      device->Compute(kernel_state.kernel, &ctx);
      TF_RETURN_IF_ERROR(ctx.status());
      for (size_t j = 0; j < num_inputs; ++j) {
        inputs[input_start_index + j].ClearVal();
      }
      for (size_t j = 0; j < num_outputs; ++j) {
        TensorValue val = ctx.release_output(j);
        const size_t num_destinations = kernel_state.output_locations[j].size();
        if (num_destinations > 0) {
          for (size_t k = 0; k < num_destinations - 1; ++k) {
            Entry& input = inputs[kernel_state.output_locations[j][k]];
            input.state = Entry::State::HAS_VALUE;
            if (val.tensor != nullptr) {
              input.val.Init(*val.tensor);
            } else {
              input.val.Init(Tensor(kernel_state.kernel->output_type(j)));
            }
          }
          Entry& input =
              inputs[kernel_state.output_locations[j][num_destinations - 1]];
          input.state = Entry::State::HAS_VALUE;
          if (val.tensor != nullptr) {
            input.val.Init(std::move(*val.tensor));
          } else {
            input.val.Init(Tensor(kernel_state.kernel->output_type(j)));
          }
        }
        delete val.tensor;
      }
    }
    return absl::OkStatus();
  }
 private:
  void RunAsyncInternal(const Args& args, DoneCallback done) override {
    args.runner([this, args, done]() { done(Run(args)); });
  }
  const LocalExecutorParams params_;
  size_t total_num_inputs_;
  struct KernelState {
    OpKernel* kernel;
    size_t input_start_index;
    size_t num_inputs;
    size_t num_outputs;
    std::vector<std::vector<size_t>>
        output_locations;  
    std::vector<AllocatorAttributes>
        output_alloc_attrs;  
  };
  std::vector<KernelState> kernels_;
  std::vector<std::vector<size_t>>
      arg_output_locations_;  
  struct ConstTensorKernelState {
    OpKernel* kernel;
    Tensor const_tensor;
    std::vector<size_t> output_locations;  
    AllocatorAttributes output_alloc_attr;
  };
  std::vector<ConstTensorKernelState> const_tensor_kernels_;
  std::vector<AllocatorAttributes>
      input_alloc_attrs_;  
};
class SingleThreadedExecutorRegistrar {
 public:
  SingleThreadedExecutorRegistrar() {
    ExecutorFactory::Register(kSingleThreadedExecutor, new Factory());
  }
 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret;
      TF_RETURN_IF_ERROR(NewSingleThreadedExecutor(params, graph, &ret));
      out_executor->reset(ret);
      return absl::OkStatus();
    }
  };
};
static SingleThreadedExecutorRegistrar registrar;
}  
Status NewSingleThreadedExecutor(const LocalExecutorParams& params,
                                 const Graph& graph, Executor** executor) {
  auto impl = std::make_unique<SingleThreadedExecutorImpl>(params);
  TF_RETURN_IF_ERROR(impl->Initialize(graph));
  *executor = impl.release();
  return absl::OkStatus();
}
}  