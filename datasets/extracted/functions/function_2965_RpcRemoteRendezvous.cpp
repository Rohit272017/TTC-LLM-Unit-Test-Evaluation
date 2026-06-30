#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace {
class RpcRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RpcRemoteRendezvous(const WorkerEnv* env, int64_t step_id)
      : BaseRemoteRendezvous(env, step_id) {}
 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;
 private:
  ~RpcRemoteRendezvous() override {}
  RpcRemoteRendezvous(const RpcRemoteRendezvous&) = delete;
  void operator=(const RpcRemoteRendezvous&) = delete;
};
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}
  void Init(WorkerInterface* wi, int64_t step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
    req_.set_request_id(GetUniqueRequestId());
  }
  void Reset() {
    DCHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall::Reset().";
    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = absl::OkStatus();
    }
    done_ = nullptr;
  }
  ~RpcRecvTensorCall() override {
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall destructor.";
  }
  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
  }
  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }
  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }
  void ReleaseWorker(WorkerCacheInterface* worker_cache) {
    DCHECK_NE(static_cast<WorkerInterface*>(nullptr), wi_)
        << "RpcRecvTensorCall::ReleaseWorker() called twice.";
    worker_cache->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
  }
  const Tensor& tensor() const { return resp_.tensor(); }
  bool is_dead() const { return resp_.metadata().is_dead(); }
  Device* dst_device() const { return dst_device_; }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }
 private:
  friend class RpcRemoteRendezvous;
  void StartRTCall(std::function<void()> recv_done) {
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    auto abort_checked = std::make_shared<Notification>();
    auto cb = [this, abort_checked,
               recv_done = std::move(recv_done)](const Status& s) {
      abort_checked->WaitForNotification();
      if (!s.ok()) {
        mutex_lock l(mu_);
        status_.Update(s);
      }
      recv_done();
    };
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
    Status s;
    {
      mutex_lock l(mu_);
      s = status_;
    }
    if (!s.ok()) {
      opts_.StartCancel();
    }
    abort_checked->Notify();
  }
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;  
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;
  mutable mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
  RpcRecvTensorCall(const RpcRecvTensorCall&) = delete;
  void operator=(const RpcRecvTensorCall&) = delete;
};
class RpcRecvTensorFreeList {
 public:
  RpcRecvTensorFreeList() {}
  ~RpcRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }
  RpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcRecvTensorCall;
  }
  void Release(RpcRecvTensorCall* obj) {
    obj->Reset();
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }
 private:
  static constexpr int kMaxObjects = 1000;
  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ TF_GUARDED_BY(mu_);
};
static RpcRecvTensorFreeList* get_call_freelist() {
  static RpcRecvTensorFreeList* call_freelist = new RpcRecvTensorFreeList();
  return call_freelist;
}
void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  CHECK(is_initialized());
  Status s;
  RpcRecvTensorCall* call = get_call_freelist()->New();
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  std::shared_ptr<WorkerCacheInterface> worker_cache =
      sess->GetSharedWorkerCache();
  WorkerInterface* rwi = worker_cache->GetOrCreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }
  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache()->ReleaseWorker(call->src_worker_, rwi);
    }
    get_call_freelist()->Release(call);
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));
  RegisterCall(call, recv_args);
  if (!call->status().ok()) {
    DeregisterCall(call, recv_args);
    call->ReleaseWorker(sess->worker_cache());
    call->done()(call->status(), Args(), Args(), Tensor(), false);
    get_call_freelist()->Release(call);
    return;
  }
  Ref();
  call->Start([this, call, recv_args, worker_cache]() {
    DeregisterCall(call, recv_args);
    Status s = call->status();
    call->ReleaseWorker(session()->worker_cache());
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    get_call_freelist()->Release(call);
    Unref();
  });
}
}  
RpcRendezvousMgr::RpcRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}
tsl::core::RefCountPtr<BaseRemoteRendezvous> RpcRendezvousMgr::Create(
    int64_t step_id, const WorkerEnv* worker_env) {
  return tsl::core::RefCountPtr<BaseRemoteRendezvous>(
      new RpcRemoteRendezvous(worker_env, step_id));
}
}  