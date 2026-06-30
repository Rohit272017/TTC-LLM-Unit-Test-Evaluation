#include "tensorflow/core/grappler/clusters/single_machine.h"
#include <atomic>
#include <memory>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
namespace tensorflow {
namespace grappler {
static std::atomic<bool> already_provisioned(false);
SingleMachine::SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus)
    : Cluster(timeout_s), expected_init_time_s_(0), closing_(false) {
  VLOG(1) << "Number of CPU cores: " << num_cpu_cores
          << " Number of GPUs: " << num_gpus;
  thread_pool_ = std::make_unique<thread::ThreadPool>(
      Env::Default(), SanitizeThreadSuffix("single_machine"), 2);
  (*options_.config.mutable_device_count())["CPU"] = 1;
  if (num_gpus > 0) {
    (*options_.config.mutable_device_count())["GPU"] = num_gpus;
  }
  CHECK_GE(num_cpu_cores, 1);
  options_.config.set_intra_op_parallelism_threads(num_cpu_cores);
  options_.config.add_session_inter_op_thread_pool()->set_num_threads(
      num_cpu_cores);
  if (timeout_s > 0) {
    options_.config.set_operation_timeout_in_ms(timeout_s * 1000);
  }
}
SingleMachine::~SingleMachine() {
  CloseSession(false ).IgnoreError();
  thread_pool_.reset();
}
Status SingleMachine::Provision() {
  if (already_provisioned) {
    return absl::UnavailableError(
        "Can't provision more than one single cluster at a time");
  }
  TF_RETURN_IF_ERROR(ResetSession());
  std::vector<DeviceAttributes> devices;
  TF_RETURN_IF_ERROR(session_->ListDevices(&devices));
  for (const auto& dev : devices) {
    DeviceProperties attr;
    if (dev.device_type() == "CPU") {
      attr = GetLocalCPUInfo();
    } else if (dev.device_type() == "GPU") {
      DeviceNameUtils::ParsedName parsed;
      if (!DeviceNameUtils::ParseFullName(dev.name(), &parsed)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Not able to parse GPU device name: ", dev.name()));
      }
      TfDeviceId tf_device_id(parsed.id);
      PlatformDeviceId platform_device_id;
      Status s =
          GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id);
      if (!s.ok()) {
        return absl::UnavailableError(
            absl::StrCat("Unknown TF GPU device with id ", tf_device_id.value(),
                         ": ", s.message()));
      }
      attr = GetLocalGPUInfo(platform_device_id);
    } else if (dev.device_type().find("XLA") == string::npos) {
      attr.set_type(dev.device_type());
    }
    attr.set_memory_size(dev.memory_limit());
    devices_[dev.name()] = attr;
  }
  already_provisioned = true;
  if (cpu_allocator_stats_enabled_) {
    TF_RETURN_IF_ERROR(ClearAllocatorStats());
  }
  return absl::OkStatus();
}
Status SingleMachine::Initialize(const GrapplerItem& item) {
  mutex_lock l(this->last_graph_mu_);
  if (last_graph_ != &item.graph || last_graph_id_ != item.id) {
    init_ops_ = item.init_ops;
    expected_init_time_s_ = item.expected_init_time;
    last_graph_ = nullptr;
    queue_runner_defs_ = item.queue_runners;
    last_graph_id_ = item.id;
  }
  return absl::OkStatus();
}
Status SingleMachine::Shutdown() {
  TF_RETURN_IF_ERROR(ShutdownSession());
  mutex_lock l(this->last_graph_mu_);
  last_graph_ = nullptr;
  already_provisioned = false;
  return absl::OkStatus();
}
Status SingleMachine::Run(const GraphDef& graph_def,
                          const std::vector<std::pair<string, Tensor>>& feed,
                          const std::vector<string>& fetch,
                          RunMetadata* metadata) {
  mutex_lock l(this->last_graph_mu_);
  if (last_graph_ != &graph_def) {
    TF_RETURN_IF_ERROR(ResetSession());
    TF_RETURN_IF_ERROR(session_->Create(graph_def));
    if (!init_ops_.empty()) {
      init_metadata_ = RunMetadata();
      int64_t timeout_s = timeout_s_ + expected_init_time_s_;
      TF_RETURN_IF_ERROR(
          RunWithTimeout({}, init_ops_, &init_metadata_, timeout_s));
      for (auto node : *init_metadata_.mutable_cost_graph()->mutable_node()) {
        node.clear_compute_cost();
      }
      init_metadata_.clear_step_stats();
    }
    RunOptions queue_options = run_options_;
    if (queue_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
      queue_options.set_trace_level(RunOptions::SOFTWARE_TRACE);
    }
    for (size_t i = 0; i < queue_runner_defs_.size(); ++i) {
      std::unique_ptr<QueueRunner> queue_runner;
      TF_RETURN_IF_ERROR(QueueRunner::New(queue_runner_defs_[i],
                                          coordinator_.get(), &queue_runner));
      TF_RETURN_IF_ERROR(queue_runner->StartAndCollectCostGraph(session_.get(),
                                                                queue_options));
      TF_RETURN_IF_ERROR(coordinator_->RegisterRunner(std::move(queue_runner)));
      TF_RETURN_IF_ERROR(coordinator_->GetStatus());
    }
    for (int i = 0; i < NumWarmupSteps(); ++i) {
      TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, nullptr));
    }
  }
  if (metadata) {
    TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, metadata));
    CostGraphDef queue_costs;
    TF_RETURN_IF_ERROR(coordinator_->ExportCostGraph(&queue_costs));
    MergeCosts(metadata->mutable_cost_graph(), init_metadata_.cost_graph(),
               queue_costs);
  } else {
    TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, nullptr));
  }
  last_graph_ = &graph_def;
  return absl::OkStatus();
}
Status SingleMachine::EnablePeakMemoryStats() {
  EnableCPUAllocatorStats();
  cpu_allocator_stats_enabled_ = true;
  return absl::OkStatus();
}
Status SingleMachine::GetPeakMemoryUsage(
    std::unordered_map<string, uint64>* device_peak_memory) const {
  if (!cpu_allocator_stats_enabled_) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Tracking allocation for CPU is not enabled.");
  }
  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&device_mgr));
  std::vector<Device*> devices = device_mgr->ListDevices();
  device_peak_memory->clear();
  for (Device* device : devices) {
    auto* allocator = device->GetAllocator(AllocatorAttributes());
    if (!allocator->TracksAllocationSizes()) {
      return Status(absl::StatusCode::kInvalidArgument,
                    "Tracking allocation is not enabled.");
    }
    absl::optional<AllocatorStats> stats = allocator->GetStats();
    (*device_peak_memory)[device->name()] =
        (stats ? stats->peak_bytes_in_use : 0);
  }
  return absl::OkStatus();
}
Status SingleMachine::RunWithTimeout(
    const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* run_metadata) {
  return RunWithTimeout(feed, fetch, run_metadata, timeout_s_);
}
Status SingleMachine::RunWithTimeout(
    const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* run_metadata,
    int64_t timeout_s) {
  {
    mutex_lock l(close_mu_);
    CHECK(!closing_);
  }
  auto status = std::make_shared<Status>();
  auto local_metadata = std::make_shared<RunMetadata>();
  const bool executed_in_time = ExecuteWithTimeout(
      [this, status, local_metadata, feed, fetch]() {
        *status = session_->Run(run_options_, feed, {}, fetch, nullptr,
                                local_metadata.get());
      },
      timeout_s * 1000, thread_pool_.get());
  if (!executed_in_time) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Failed to run the graph after ", timeout_s, " seconds, aborting"));
  } else if (run_metadata && status->ok()) {
    *run_metadata = *local_metadata;
  }
  return *status;
}
Status SingleMachine::CloseSession(bool use_timeout) {
  if (!session_ || !thread_pool_) {
    return absl::OkStatus();
  }
  {
    mutex_lock l(close_mu_);
    if (!closing_) {
      closing_ = true;
    }
  }
  const bool executed_in_time = ExecuteWithTimeout(
      [&]() {
        if (this->coordinator_) {
          this->coordinator_->RequestStop().IgnoreError();
          while (!this->coordinator_->AllRunnersStopped()) {
            Env::Default()->SleepForMicroseconds(1000000);
          }
          this->session_->Close().IgnoreError();
          this->coordinator_.reset();
        } else {
          this->session_->Close().IgnoreError();
        }
        mutex_lock l2(close_mu_);
        closing_ = false;
      },
      use_timeout ? timeout_s_ * 1000 : -1, thread_pool_.get());
  if (!executed_in_time) {
    return absl::UnavailableError(
        absl::StrCat("Failed to close the previous session after ", timeout_s_,
                     " seconds, aborting"));
  }
  return absl::OkStatus();
}
Status SingleMachine::ShutdownSession() {
  TF_RETURN_IF_ERROR(CloseSession(true ));
  auto n = std::make_shared<Notification>();
  Env::Default()->SchedClosure([this, n]() {
    thread_pool_.reset();
    n->Notify();
  });
  int64_t timeout_us = 1000000ll * timeout_s_;
  const bool notified = WaitForNotificationWithTimeout(n.get(), timeout_us);
  if (!notified) {
    return absl::UnavailableError(absl::StrCat(
        "The session is still running graphs after ", timeout_s_, " seconds"));
  }
  return absl::OkStatus();
}
Status SingleMachine::ResetSession() {
  if (session_) {
    LOG(INFO) << "Cleaning up previous session";
    TF_RETURN_IF_ERROR(ShutdownSession());
    session_.reset();
  }
  LOG(INFO) << "Starting new session";
  thread_pool_ = std::make_unique<thread::ThreadPool>(
      Env::Default(), SanitizeThreadSuffix("single_machine"), 2);
  session_.reset(NewSession(options_));
  if (!session_) {
    return absl::UnknownError("Failed to create session");
  }
  coordinator_ = std::make_unique<Coordinator>();
  device_set_ = std::make_unique<DeviceSet>();
  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&device_mgr));
  for (auto d : device_mgr->ListDevices()) {
    device_set_->AddDevice(d);
  }
  return absl::OkStatus();
}
void SingleMachine::MergeCosts(CostGraphDef* graph_costs,
                               const CostGraphDef& init_costs,
                               const CostGraphDef& queue_costs) {
  graph_costs->mutable_node()->Reserve(graph_costs->node_size() +
                                       init_costs.node_size() +
                                       queue_costs.node_size());
  std::unordered_set<string> nodes_seen;
  int queue_costs_id_offset = graph_costs->node_size();
  for (const auto& node : graph_costs->node()) {
    nodes_seen.insert(node.name());
    if (node.id() >= queue_costs_id_offset) {
      queue_costs_id_offset = node.id() + 1;
    }
  }
  int init_costs_id_offset = queue_costs_id_offset + queue_costs.node_size();
  for (const auto& node : queue_costs.node()) {
    if (nodes_seen.find(node.name()) != nodes_seen.end()) {
      continue;
    }
    auto* new_node = graph_costs->add_node();
    new_node->MergeFrom(node);
    new_node->set_id(node.id() + queue_costs_id_offset);
    if (new_node->id() >= init_costs_id_offset) {
      init_costs_id_offset = new_node->id() + 1;
    }
    for (auto& input_info : *new_node->mutable_input_info()) {
      input_info.set_preceding_node(input_info.preceding_node() +
                                    queue_costs_id_offset);
    }
    for (auto& control_input : *new_node->mutable_control_input()) {
      control_input += queue_costs_id_offset;
    }
  }
  for (const auto& node : init_costs.node()) {
    if (nodes_seen.find(node.name()) != nodes_seen.end()) {
      continue;
    }
    auto* new_node = graph_costs->add_node();
    new_node->MergeFrom(node);
    new_node->set_id(node.id() + init_costs_id_offset);
    for (auto& input_info : *new_node->mutable_input_info()) {
      input_info.set_preceding_node(input_info.preceding_node() +
                                    init_costs_id_offset);
    }
    for (auto& control_input : *new_node->mutable_control_input()) {
      control_input += init_costs_id_offset;
    }
  }
}
Status SingleMachine::ClearAllocatorStats() const {
  if (!cpu_allocator_stats_enabled_) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Tracking allocation for CPU is not enabled.");
  }
  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&device_mgr));
  std::vector<Device*> devices = device_mgr->ListDevices();
  for (Device* device : devices) {
    auto* allocator = device->GetAllocator(AllocatorAttributes());
    if (!allocator->TracksAllocationSizes()) {
      return Status(absl::StatusCode::kInvalidArgument,
                    "Tracking allocation is not enabled.");
    }
    if (!allocator->ClearStats()) {
      return Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Clearing allocation stats is not supported for ",
                       device->name()));
    }
  }
  return absl::OkStatus();
}
}  
}  