#define EIGEN_USE_THREADS
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/single_threaded_cpu_device.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"
namespace tensorflow {
namespace {
class SimpleRendezvous : public RendezvousInterface {
 public:
  explicit SimpleRendezvous() {}
  Status Send(const ParsedKey& parsed, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    if (is_dead) {
      return errors::Internal("Send of a dead tensor");
    }
    mutex_lock l(mu_);
    string edge_name(parsed.edge_name);
    if (table_.count(edge_name) > 0) {
      return errors::Internal("Send of an already sent tensor");
    }
    table_[edge_name] = val;
    return absl::OkStatus();
  }
  void RecvAsync(const ParsedKey& parsed, const Args& recv_args,
                 DoneCallback done) override {
    Tensor tensor;
    Status status = absl::OkStatus();
    {
      string key(parsed.edge_name);
      mutex_lock l(mu_);
      if (table_.count(key) <= 0) {
        status = errors::Internal("Did not find key ", key);
      } else {
        tensor = table_[key];
      }
    }
    done(status, Args{}, recv_args, tensor, false);
  }
  void StartAbort(const Status& status) override {}
 private:
  typedef std::unordered_map<string, Tensor> Table;
  mutex mu_;
  Table table_ TF_GUARDED_BY(mu_);
};
}  
GraphRunner::GraphRunner(Env* env)
    : device_deleter_(NewSingleThreadedCpuDevice(env)),
      device_(device_deleter_.get()) {}
GraphRunner::GraphRunner(Device* device) : device_(device) {}
GraphRunner::~GraphRunner() {}
Status GraphRunner::Run(Graph* graph, FunctionLibraryRuntime* function_library,
                        const NamedTensorList& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
  if (device_ == nullptr) {
    return errors::NotFound("Cannot find a device for GraphRunner.");
  }
  if (function_library && function_library->device() &&
      function_library->device()->device_type() != device_->device_type()) {
    VLOG(1) << "Cannot run on: " << device_->device_type()
            << " with a function library for a "
            << function_library->device()->device_type() << " device.";
    function_library = nullptr;
  }
  std::unique_ptr<Graph> graph_to_run(new Graph(graph->op_registry()));
  CopyGraph(*graph, graph_to_run.get());
  SimpleRendezvous rendez;
  std::vector<string> input_names;
  for (const auto& in : inputs) {
    const string& tensor_name = in.first;
    input_names.emplace_back(tensor_name);
    string full_key = Rendezvous::CreateKey("/device:CPU:0", 1, "/device:CPU:1",
                                            tensor_name, FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(full_key, &parsed));
    TF_RETURN_IF_ERROR(rendez.Send(parsed, Rendezvous::Args(), in.second,
                                   false ));
  }
  subgraph::RewriteGraphMetadata metadata;
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph_to_run.get(), input_names, output_names, {} ,
      device_->attributes(), false , &metadata));
  auto runner = [](Executor::Args::Closure c) { c(); };
  LocalExecutorParams params;
  params.device = device_;
  params.function_library = function_library;
  const int producer = graph_to_run->versions().producer();
  params.create_kernel = [this, function_library, producer](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
    return CreateNonCachedKernel(device_, function_library, props, producer,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) { delete kernel; };
  Executor* executor;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, *graph_to_run, &executor));
  std::unique_ptr<Executor> executor_unref(executor);
  Executor::Args args;
  args.step_id = LogMemory::CONSTANT_FOLDING_STEP_ID;
  args.runner = runner;
  args.rendezvous = &rendez;
  args.collective_executor = nullptr;
  CancellationManager cancellation_manager;
  args.cancellation_manager = &cancellation_manager;
  if (function_library != nullptr) {
    args.session_config = function_library->config_proto();
  }
  TF_RETURN_IF_ERROR(executor->Run(args));
  outputs->resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); ++i) {
    const string& output_key =
        Rendezvous::CreateKey("/device:CPU:0", 1, "/device:CPU:1",
                              output_names[i], FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(output_key, &parsed));
    bool is_dead;
    Tensor output_tensor;
    TF_RETURN_IF_ERROR(
        rendez.Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead));
    (*outputs)[i] = tensor::DeepCopy(output_tensor);
  }
  return absl::OkStatus();
}
}  