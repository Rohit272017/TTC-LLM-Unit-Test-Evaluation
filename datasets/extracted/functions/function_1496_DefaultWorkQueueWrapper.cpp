#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "tfrt/host_context/execution_context.h"  
namespace tensorflow {
namespace tfrt_stub {
namespace {
class DefaultWorkQueueWrapper : public WorkQueueInterface {
 public:
  explicit DefaultWorkQueueWrapper(
      std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue)
      : WorkQueueInterface(0),
        work_queue_owner_(std::move(work_queue)),
        work_queue_(work_queue_owner_.get()) {}
  DefaultWorkQueueWrapper(std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
                          thread::ThreadPoolInterface* intra_thread_pool)
      : WorkQueueInterface(0, intra_thread_pool),
        work_queue_owner_(std::move(work_queue)),
        work_queue_(work_queue_owner_.get()) {}
  DefaultWorkQueueWrapper(int64_t request_id,
                          tfrt::ConcurrentWorkQueue* work_queue,
                          thread::ThreadPoolInterface* intra_thread_pool)
      : WorkQueueInterface(request_id, intra_thread_pool),
        work_queue_(work_queue) {}
  ~DefaultWorkQueueWrapper() override = default;
 private:
  std::string name() const override { return work_queue_->name(); }
  void AddTask(tfrt::TaskFunction work) override {
    work_queue_->AddTask(WrapWork(id(), "inter", std::move(work)));
  }
  std::optional<tfrt::TaskFunction> AddBlockingTask(
      tfrt::TaskFunction work, bool allow_queuing) override {
    return work_queue_->AddBlockingTask(
        WrapWork(id(), "blocking", std::move(work)), allow_queuing);
  }
  void Await(
      llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) override {
    work_queue_->Await(values);
  }
  void Quiesce() override { work_queue_->Quiesce(); }
  int GetParallelismLevel() const override {
    return work_queue_->GetParallelismLevel();
  }
  bool IsInWorkerThread() const override {
    return work_queue_->IsInWorkerThread();
  }
  absl::StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      int64_t request_id) const override {
    return {std::make_unique<DefaultWorkQueueWrapper>(request_id, work_queue_,
                                                      GetIntraOpThreadPool())};
  }
 private:
  std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue_owner_;
  tfrt::ConcurrentWorkQueue* work_queue_ = nullptr;
};
}  
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue) {
  return std::make_unique<DefaultWorkQueueWrapper>(std::move(work_queue));
}
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
    thread::ThreadPoolInterface* intra_thread_pool) {
  return std::make_unique<DefaultWorkQueueWrapper>(std::move(work_queue),
                                                   intra_thread_pool);
}
}  
}  