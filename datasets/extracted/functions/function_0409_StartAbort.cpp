#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
namespace tensorflow {
void CollectiveRemoteAccessLocal::StartAbort(const Status& s) {
  buf_rendezvous_.StartAbort(s);
}
void CollectiveRemoteAccessLocal::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  VLOG(1) << "RecvFromPeer " << this << " from " << peer_device << " key "
          << key;
  if (!peer_is_local) {
    done(
        errors::Internal("CollectiveRemoteAccessLocal::RecvFromPeer "
                         "called with peer_is_local=false"));
    return;
  }
  Device* from_device;
  Status status = dev_mgr_->LookupDevice(peer_device, &from_device);
  if (!status.ok()) {
    done(status);
    return;
  }
  auto consumer_callback = [to_tensor, to_device_ctx, to_device, to_alloc_attr,
                            dev_to_dev_stream_index,
                            done](const Status& status,
                                  BufRendezvous::Hook* hook) {
    Status s = status;
    if (s.ok()) {
      if (hook == nullptr) {
        s = errors::Internal("Invalid null hook in ConsumeBuf callback");
      }
    } else {
      if (hook != nullptr) {
        LOG(ERROR) << "Got hook " << hook << " with status " << s
                   << " from ConsumeBuf";
      }
    }
    if (s.ok()) {
      int64_t recv_bytes = to_tensor->TotalBytes();
      CHECK_EQ(recv_bytes, hook->prod_value->TotalBytes());
      MemCpyAsync(hook->prod_ctx,    
                  to_device_ctx,     
                  hook->prod_dev,    
                  to_device,         
                  hook->prod_attr,   
                  to_alloc_attr,     
                  hook->prod_value,  
                  to_tensor,         
                  dev_to_dev_stream_index,
                  [hook, done](const Status& memcpy_status) {
                    done(memcpy_status);
                    BufRendezvous::DoneWithHook(hook);
                  });
    } else {
      done(s);
      if (hook != nullptr) {
        BufRendezvous::DoneWithHook(hook);
      }
    }
  };
  buf_rendezvous_.ConsumeBuf(key, from_device->name(),
                             from_device->attributes().incarnation(),
                             consumer_callback, cancellation_manager);
}
void CollectiveRemoteAccessLocal::PostToPeer(
    const string& peer_device, const string& peer_task, const string& key,
    Device* from_device, DeviceContext* from_device_ctx,
    const AllocatorAttributes& from_alloc_attr, const Tensor* from_tensor,
    const DeviceLocality& client_locality,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  VLOG(1) << "PostToPeer " << this << " key " << key
          << " step_id_=" << step_id_;
  buf_rendezvous_.ProvideBuf(key, from_device, from_device_ctx, from_tensor,
                             from_alloc_attr, done, cancellation_manager);
}
void CollectiveRemoteAccessLocal::CheckPeerHealth(const string& peer_task,
                                                  int64_t timeout_in_ms,
                                                  const StatusCallback& done) {
  done(errors::Internal(
      "CheckPeerHealth is not supposed to be called for local collectives"));
}
void CollectiveRemoteAccessLocal::MemCpyAsync(
    DeviceContext* src_dev_ctx, DeviceContext* dst_dev_ctx, Device* src_dev,
    Device* dst_dev, const AllocatorAttributes& src_attr,
    const AllocatorAttributes& dst_attr, const Tensor* src, Tensor* dst,
    int dev_to_dev_stream_index, const StatusCallback& done) {
  const DeviceType src_device_type(
      src_attr.on_host() ? DEVICE_CPU : src_dev->attributes().device_type());
  const DeviceType dst_device_type(
      dst_attr.on_host() ? DEVICE_CPU : dst_dev->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);
  if (src_dev_ctx == nullptr && src_device_type == DEVICE_GPU) {
    const DeviceBase::AcceleratorDeviceInfo* dev_info =
        src_dev->tensorflow_accelerator_device_info();
    CHECK(dev_info);
    src_dev_ctx = dev_info->default_context;
  }
  if (dst_dev_ctx == nullptr && dst_device_type == DEVICE_GPU) {
    const DeviceBase::AcceleratorDeviceInfo* dev_info =
        src_dev->tensorflow_accelerator_device_info();
    CHECK(dev_info);
    dst_dev_ctx = dev_info->default_context;
  }
  if (non_cpu_src) CHECK(src_dev_ctx);
  if (non_cpu_dst) CHECK(dst_dev_ctx);
  if (non_cpu_src || non_cpu_dst) {
    CopyTensor::ViaDMA("",  
                       src_dev_ctx, dst_dev_ctx, src_dev, dst_dev, src_attr,
                       dst_attr, src, dst, dev_to_dev_stream_index, done);
  } else {
    int64_t bytes = src->TotalBytes();
    DCHECK_EQ(dst->TotalBytes(), bytes);
    memcpy(DMAHelper::base(dst), DMAHelper::base(src), bytes);
    done(absl::OkStatus());
  }
}
}  