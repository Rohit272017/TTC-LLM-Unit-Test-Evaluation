#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"
#include <memory>
#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
namespace tensorflow {
namespace {
class RecvBufCall : public CancellableCall {
 public:
  RecvBufCall(int64_t step_id, const string& peer_device,
              const string& peer_task, const string& key, Device* to_device,
              DeviceContext* to_device_ctx,
              const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
              const DeviceLocality& client_locality,
              const DeviceAttributes& server_attributes,
              CancellationManager* cancel_mgr, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, peer_task, wc) {
    req_.set_step_id(step_id);
    req_.set_buf_rendezvous_key(key);
    *req_.mutable_client_locality() = client_locality;
    *req_.mutable_server_locality() = server_attributes.locality();
    req_.set_num_bytes(to_tensor->TotalBytes());
    req_.set_buf_ptr(reinterpret_cast<int64_t>(DMAHelper::base(to_tensor)));
    req_.set_src_device(peer_device);
    req_.set_src_incarnation(server_attributes.incarnation());
    req_.set_dst_device(to_device->name());
    req_.set_request_id(GetUniqueRequestId());
  }
  ~RecvBufCall() override {}
  void IssueCall(const StatusCallback& done) override {
    wi_->RecvBufAsync(&opts_, &req_, &resp_, done);
  }
  RecvBufRequest req_;
  RecvBufResponse resp_;
};
void PopulateTensorFromExtra(const RecvBufRespExtra& extra,
                             Tensor* cpu_tensor) {
  char* head = reinterpret_cast<char*>(DMAHelper::base(cpu_tensor));
  for (const auto& tensor_content_chunk : extra.tensor_content()) {
    memcpy(head, std::string(tensor_content_chunk).data(),
           tensor_content_chunk.size());
    head += tensor_content_chunk.size();
  }
}
Status PopulateTensorFromResponse(const RecvBufResponse& response,
                                  Tensor* cpu_tensor) {
  const bool has_transport_options = response.has_transport_options();
  if (!has_transport_options) return absl::OkStatus();
  const int64_t total_bytes = cpu_tensor->TotalBytes();
  int64_t num_bytes = 0;
  RecvBufRespExtra extra;
  response.transport_options().UnpackTo(&extra);
  for (const auto& chunk : extra.tensor_content()) {
    num_bytes += chunk.size();
  }
  if (num_bytes != total_bytes) {
    return errors::Internal("Tensor Size Mismatch: RecvBufResponse returned ",
                            num_bytes,
                            " bytes, expected: ", cpu_tensor->TotalBytes());
  }
  PopulateTensorFromExtra(extra, cpu_tensor);
  return absl::OkStatus();
}
}  
void CollectiveRemoteAccessDistributed::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  if (peer_is_local) {
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
        cancellation_manager, done);
    return;
  }
  struct State {
    DeviceAttributes server_attributes;
    std::unique_ptr<RecvBufCall> call;
    std::unique_ptr<Tensor> cpu_tensor;
  };
  State* state = new State;
  Status s = dev_resolver_->GetDeviceAttributes(peer_device,
                                                &state->server_attributes);
  if (!s.ok()) {
    delete state;
    done(s);
    return;
  }
  Tensor* dst_tensor = nullptr;
  Device* cpu_dev = nullptr;
  if (to_device->tensorflow_accelerator_device_info()) {
    Status status = dev_mgr_->LookupDevice("CPU:0", &cpu_dev);
    if (!status.ok()) {
      delete state;
      done(s);
      return;
    }
    AllocatorAttributes cpu_attr;
    cpu_attr.set_gpu_compatible(true);
    tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
        "CollectiveRemoteAccessDistributed::RecvFromPeer"
        "::recv_buf_callback",
        step_id_, "dynamic", to_tensor->dtype(),
        [to_tensor]() { return to_tensor->shape().DebugString(); });
    state->cpu_tensor =
        std::make_unique<Tensor>(cpu_dev->GetAllocator(cpu_attr),
                                 to_tensor->dtype(), to_tensor->shape());
    dst_tensor = state->cpu_tensor.get();
  } else {
    dst_tensor = to_tensor;
  }
  auto recv_buf_callback =
      [this, state, to_device, to_alloc_attr, to_device_ctx, to_tensor, cpu_dev,
       dev_to_dev_stream_index, dst_tensor, done](const Status& s) {
        if (s.ok()) {
          Status status =
              PopulateTensorFromResponse(state->call->resp_, dst_tensor);
          if (!status.ok()) {
            done(status);
            delete state;
            return;
          }
          if (to_device->tensorflow_accelerator_device_info()) {
            AllocatorAttributes cpu_attr;
            cpu_attr.set_gpu_compatible(true);
            CopyTensor::ViaDMA("",  
                               nullptr , to_device_ctx, cpu_dev,
                               to_device, cpu_attr, to_alloc_attr, dst_tensor,
                               to_tensor, dev_to_dev_stream_index,
                               [this, state, done](const Status& s) {
                                 delete state;
                                 work_queue_->Schedule([s, done] { done(s); });
                               });
            return;
          }
        }
        delete state;
        done(s);
      };
  state->call = std::make_unique<RecvBufCall>(
      step_id_, peer_device, peer_task, key, to_device, to_device_ctx,
      to_alloc_attr, dst_tensor, client_locality, state->server_attributes,
      cancellation_manager, worker_cache_);
  CancellationToken abortion_token =
      abortion_cancel_mgr_.get_cancellation_token();
  bool already_aborted = !abortion_cancel_mgr_.RegisterCallback(
      abortion_token, [state] { state->call->Cancel(); });
  if (already_aborted) {
    recv_buf_callback(errors::Cancelled("collective ops already aborted"));
  } else {
    state->call->Start([this, abortion_token,
                        done = std::move(recv_buf_callback)](const Status& s) {
      abortion_cancel_mgr_.DeregisterCallback(abortion_token);
      done(s);
    });
  }
}
void CollectiveRemoteAccessDistributed::CheckPeerHealth(
    const string& peer_task, int64_t timeout_in_ms,
    const StatusCallback& done) {
  if (peer_task == task_name_) {
    done(absl::OkStatus());
    return;
  }
  WorkerInterface* wi = worker_cache_->GetOrCreateWorker(peer_task);
  if (wi == nullptr) {
    done(errors::InvalidArgument(peer_task,
                                 " not found. It's probably invalid. The "
                                 "valid form is /job:xxx/replica:0/task:N"));
    return;
  }
  auto opts = new CallOptions();
  opts->SetTimeout(timeout_in_ms);
  auto req = new GetStatusRequest();
  auto resp = new GetStatusResponse();
  wi->GetStatusAsync(
      opts, req, resp,  true,
      [this, opts, req, resp, wi, peer_task, done](Status s) {
        std::vector<DeviceAttributes> cached_attrs;
        if (s.ok()) {
          s = dev_resolver_->GetAllDeviceAttributes(peer_task, &cached_attrs);
        }
        if (s.ok()) {
          absl::flat_hash_set<uint64> remote_incarnations;
          for (const DeviceAttributes& da : resp->device_attributes()) {
            remote_incarnations.insert(da.incarnation());
          }
          for (const DeviceAttributes& attr : cached_attrs) {
            if (!remote_incarnations.contains(attr.incarnation())) {
              s = errors::FailedPrecondition(
                  attr.name(), " with incarnation ", attr.incarnation(),
                  " is not available. This usually means ", peer_task,
                  " has restarted");
              break;
            }
          }
        } else if (absl::IsNotFound(s)) {
          s = absl::OkStatus();
        }
        delete opts;
        delete req;
        delete resp;
        worker_cache_->ReleaseWorker(peer_task, wi);
        done(s);
      });
}
void CollectiveRemoteAccessDistributed::StartAbort(const Status& s) {
  CollectiveRemoteAccessLocal::StartAbort(s);
  abortion_cancel_mgr_.StartCancel();
}
}  