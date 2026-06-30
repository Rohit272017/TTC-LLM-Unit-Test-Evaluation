#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"
namespace tensorflow {
namespace {
class GrpcWorkerCache : public WorkerCachePartial {
 public:
  explicit GrpcWorkerCache(std::shared_ptr<GrpcChannelCache> channel_cache,
                           WorkerInterface* local_worker,
                           const string& local_target,
                           GrpcWorkerEnv* worker_env)
      : local_target_(local_target),
        local_worker_(local_worker),
        channel_cache_(channel_cache),
        worker_env_(worker_env),
        next_round_robin_assignment_(0) {}
  void ListWorkers(std::vector<string>* workers) const override {
    channel_cache_->ListWorkers(workers);
  }
  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    channel_cache_->ListWorkersInJob(job_name, workers);
  }
  WorkerInterface* GetOrCreateWorker(const string& target) override {
    if (target == local_target_) {
      return local_worker_;
    } else {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (!channel) {
        return nullptr;
      }
      size_t index = AssignWorkerToThread(target);
      return NewGrpcRemoteWorker(
          channel, worker_env_->GetCompletionQueue(index),
          worker_env_->GetThreadPool(), &logger_, target);
    }
  }
  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
          << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }
  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
    eager_client_cache->reset(eager::NewGrpcEagerClientCache(channel_cache_));
    return absl::OkStatus();
  }
  Status GetCoordinationClientCache(std::unique_ptr<CoordinationClientCache>*
                                        coordination_client_cache) override {
    coordination_client_cache->reset(
        NewGrpcCoordinationClientCache(channel_cache_));
    return absl::OkStatus();
  }
  void SetLogging(bool v) override { logger_.SetLogging(v); }
  void ClearLogs() override { logger_.ClearLogs(); }
  bool RetrieveLogs(int64_t step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }
 private:
  size_t AssignWorkerToThread(const string& target) {
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(target,
                                      (next_round_robin_assignment_++) %
                                          worker_env_->CompletionQueueSize()))
               .first;
    }
    return it->second;
  }
  const string local_target_;
  WorkerInterface* const local_worker_;  
  std::shared_ptr<GrpcChannelCache> channel_cache_;
  WorkerCacheLogger logger_;
  GrpcWorkerEnv* worker_env_;  
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);
};
}  
GrpcWorkerEnv::GrpcWorkerEnv(size_t num_completion_queues, size_t num_threads)
    : threadpool_(new thread::ThreadPool(
          Env::Default(), ThreadOptions(), "GrpcWorkerEnvQueues", num_threads,
          false, nullptr)),
      threads_(num_completion_queues) {}
GrpcWorkerEnv::~GrpcWorkerEnv() { threads_.clear(); }
GrpcWorkerEnv::GrpcWorkerCacheThread::GrpcWorkerCacheThread() {
  thread_.reset(Env::Default()->StartThread(
      ThreadOptions(), "GrpcWorkerEnvPool", [this]() {
        void* tag;
        bool ok;
        while (completion_queue_.Next(&tag, &ok)) {
          GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
          callback_tag->OnCompleted(ok);
        }
      }));
}
GrpcWorkerEnv::GrpcWorkerCacheThread::~GrpcWorkerCacheThread() {
  completion_queue_.Shutdown();
  thread_.reset();
}
GrpcWorkerEnv* CreateGrpcWorkerEnv() {
  int num_cpus = port::NumSchedulableCPUs();
  int64_t num_completion_queues;
  Status status = ReadInt64FromEnvVar("TF_GRPC_WORKER_CACHE_QUEUES", 64,
                                      &num_completion_queues);
  if (!status.ok()) {
    LOG(ERROR) << "Error parsing TF_GRPC_WORKER_CACHE_QUEUES: " << status;
  }
  int64_t num_threads;
  status = ReadInt64FromEnvVar("TF_GRPC_WORKER_CACHE_THREADS", num_cpus,
                               &num_threads);
  if (!status.ok()) {
    LOG(ERROR) << "Error parsing TF_GRPC_WORKER_CACHE_THREADS: " << status;
  }
  return new GrpcWorkerEnv(num_completion_queues, num_threads);
}
WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc,
                                         GrpcWorkerEnv* worker_env) {
  return new GrpcWorkerCache(cc, nullptr, "",
                             worker_env);
}
WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, GrpcWorkerEnv* worker_env,
    WorkerInterface* local_worker, const string& local_target) {
  return new GrpcWorkerCache(cc, local_worker, local_target, worker_env);
}
}  