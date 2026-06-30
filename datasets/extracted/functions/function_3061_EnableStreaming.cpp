#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"
#include <cstdint>
#include <string>
#include "grpcpp/generic/generic_stub.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/util/env_var.h"
namespace tensorflow {
namespace eager {
namespace {
bool EnableStreaming() {
  bool result;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE",
                                 true, &result));
  return result;
}
class GrpcEagerClientThread : public core::RefCounted {
 public:
  GrpcEagerClientThread() {
    Ref();
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "eager_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcEagerClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcEagerClientThread blocking for next tag";
            if (RefCountIsOne()) {
              break;
            }
          }
          VLOG(4) << "GrpcEagerClientThread exiting";
          completion_queue_.Shutdown();
          Env::Default()->SchedClosure([this]() { this->Unref(); });
        }));
  }
  ~GrpcEagerClientThread() override {}
  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }
 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};
class GrpcEagerClient : public EagerClient {
 public:
  GrpcEagerClient(const tensorflow::SharedGrpcChannelPtr& channel,
                  GrpcEagerClientThread* thread, const string& target)
      : stub_(channel), thread_(thread), target_(target) {
    thread_->Ref();
    cq_ = thread->completion_queue();
  }
  ~GrpcEagerClient() override { thread_->Unref(); }
  bool allow_multiple_pending_requests() const override {
    return EnableStreaming();
  }
#define CLIENT_METHOD(method)                                             \
  void method##Async(const method##Request* request,                      \
                     method##Response* response, StatusCallback done)     \
      override {                                                          \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));      \
    new RPCState<protobuf::Message>(                                      \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request, \
        response, std::move(done_wrapped), nullptr,         \
        nullptr, 0, true,    \
        &target_);                                                        \
  }
  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(UpdateContext);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);
#undef CLIENT_METHOD
#define CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES(method)                       \
  void method##Async(const method##Request* request,                         \
                     method##Response* response, StatusCallback done,        \
                     int64_t init_timeout_in_ms, int retries) override {     \
    CallOptions* call_ops = nullptr;                                         \
    StatusCallback done_wrapped;                                             \
    if (init_timeout_in_ms > 0) {                                            \
      call_ops = new CallOptions;                                            \
      call_ops->SetTimeout(init_timeout_in_ms);                              \
      auto new_done = [call_ops, done = std::move(done)](const Status& s) {  \
        done(s);                                                             \
        delete call_ops;                                                     \
      };                                                                     \
      done_wrapped = callback_wrapper(new_done);                             \
    } else {                                                                 \
      done_wrapped = callback_wrapper(std::move(done));                      \
    }                                                                        \
    new RPCState<protobuf::Message>(                                         \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request,    \
        response, std::move(done_wrapped), call_ops, nullptr, \
        retries, true, &target_);              \
  }
  CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES(CreateContext);
#undef CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES
#define CLIENT_CANCELABLE_METHOD(method)                                      \
  void method##Async(CallOptions* call_opts, const method##Request* request,  \
                     method##Response* response, StatusCallback done)         \
      override {                                                              \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));          \
    new RPCState<protobuf::Message>(                                          \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request,     \
        response, std::move(done_wrapped), call_opts, nullptr, \
        0, true, &target_);                     \
  }
  CLIENT_CANCELABLE_METHOD(Enqueue);
  CLIENT_CANCELABLE_METHOD(RunComponentFunction);
#undef CLIENT_CANCELABLE_METHOD
  void CloseContextAsync(const CloseContextRequest* request,
                         CloseContextResponse* response,
                         StatusCallback done) override {
    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.eager.EagerService/CloseContext", *request,
        response, std::move(done_wrapped), nullptr,
        nullptr, 0, true,
        &target_);
    VLOG(1) << "Sending RPC to close remote eager context "
            << request->DebugString();
    mutex_lock l(mu_);
    const auto& it = enqueue_dispatchers_.find(request->context_id());
    if (it != enqueue_dispatchers_.end()) {
      it->second.CancelCall();
      enqueue_dispatchers_.erase(it);
    } else if (EnableStreaming()) {
      LOG(ERROR) << "Remote EagerContext with id " << request->context_id()
                 << " does not seem to exist.";
    }
  }
  void StreamingEnqueueAsync(bool enable_streaming_enqueue,
                             CallOptions* call_opts,
                             const EnqueueRequest* request,
                             EnqueueResponse* response,
                             StatusCallback done) override {
    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    if (EnableStreaming() && enable_streaming_enqueue) {
      mutex_lock l(mu_);
      auto it = enqueue_dispatchers_.find(request->context_id());
      if (it == enqueue_dispatchers_.end()) {
        auto it_and_bool = enqueue_dispatchers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(request->context_id()),
            std::forward_as_tuple(
                &stub_, cq_,
                "/tensorflow.eager.EagerService/StreamingEnqueue"));
        it = it_and_bool.first;
      }
      it->second.SendNextRequest(*request, response, std::move(done_wrapped));
    } else {
      Notification n;
      Status status;
      EnqueueAsync(call_opts, request, response,
                   [&n, &status](const Status& s) {
                     status.Update(s);
                     n.Notify();
                   });
      n.WaitForNotification();
      done_wrapped(status);
    }
  }
 private:
  ::grpc::GenericStub stub_;
  const GrpcEagerClientThread* thread_;
  const string target_;
  ::grpc::CompletionQueue* cq_;
  mutable mutex mu_;
  std::unordered_map<uint64, StreamingRPCDispatcher<EnqueueResponse>>
      enqueue_dispatchers_ TF_GUARDED_BY(mu_);
  StatusCallback callback_wrapper(StatusCallback done) {
    Ref();
    return [this, done = std::move(done)](const Status& status) {
      done(status);
      this->Unref();
      if (TF_PREDICT_FALSE(!status.ok())) {
        auto error_source_payload = status.GetPayload(kErrorSource);
        if (error_source_payload.has_value()) {
          tensorflow::core::platform::ErrorSourceProto error_source_proto;
          error_source_proto.ParseFromString(
              std::string(*error_source_payload));  
          metrics::UpdateEagerClientErrorCounter(
              error_source_proto.ErrorSource_Name(
                  error_source_proto.error_source()),
              absl::StatusCodeToString(status.code()));
        } else {
          metrics::UpdateEagerClientErrorCounter(
              "unknown", absl::StatusCodeToString(status.code()));
        }
      }
    };
  }
};
class GrpcEagerClientCache : public EagerClientCache {
 public:
  explicit GrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> cache)
      : next_round_robin_assignment_(0), cache_(cache), threads_(4) {
    for (int i = 0, end = threads_.size(); i < end; i++) {
      threads_[i].reset(new GrpcEagerClientThread());
    }
  }
  ~GrpcEagerClientCache() override { threads_.clear(); }
  Status GetClient(const string& target,
                   core::RefCountPtr<EagerClient>* client) override {
    mutex_lock l(clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      tensorflow::SharedGrpcChannelPtr shared =
          cache_->FindWorkerChannel(target);
      if (shared == nullptr) {
        return errors::InvalidArgument("Client for target ", target,
                                       " not found.");
      }
      int assigned_index = AssignClientToThread(target);
      GrpcEagerClientThread* thread = threads_[assigned_index].get();
      core::RefCountPtr<EagerClient> worker(
          new GrpcEagerClient(shared, thread, target));
      it = clients_.emplace(target, std::move(worker)).first;
    }
    it->second->Ref();
    client->reset(it->second.get());
    return absl::OkStatus();
  }
 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);
  size_t AssignClientToThread(const string& target) {
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }
  std::shared_ptr<tensorflow::GrpcChannelCache> cache_;
  mutable mutex clients_mu_;
  std::unordered_map<string, core::RefCountPtr<EagerClient>> clients_
      TF_GUARDED_BY(clients_mu_);
  std::vector<core::RefCountPtr<GrpcEagerClientThread>> threads_;
};
}  
EagerClientCache* NewGrpcEagerClientCache(
    std::shared_ptr<tensorflow::GrpcChannelCache> channel) {
  return new GrpcEagerClientCache(channel);
}
}  
}  