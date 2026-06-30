#include "xla/stream_executor/host/host_stream.h"
#include <string.h>
#include <cfenv>  
#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/host/host_event.h"
#include "xla/stream_executor/host/host_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/env.h"
#include "tsl/platform/setround.h"
namespace stream_executor {
namespace host {
HostStream::HostStream(StreamExecutor* executor)
    : StreamCommon(executor),
      thread_(tsl::Env::Default()->StartThread({}, "host_executor",
                                               [this]() { WorkLoop(); })) {}
HostStream::~HostStream() {
  {
    absl::MutexLock lock(&mu_);
    work_queue_.push(nullptr);
  }
  thread_.reset();
  parent()->DeallocateStream(this);
}
absl::Status HostStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  EnqueueTask([src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return absl::OkStatus();
}
absl::Status HostStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  EnqueueTask([host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return absl::OkStatus();
}
absl::Status HostStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  void* dst_mem = gpu_dst->opaque();
  EnqueueTask([dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return absl::OkStatus();
}
absl::Status HostStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                  uint64_t size) {
  void* gpu_mem = location->opaque();
  EnqueueTask([gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return absl::OkStatus();
}
absl::Status HostStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  void* gpu_mem = location->opaque();
  EnqueueTask([gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return absl::OkStatus();
}
absl::Status HostStream::WaitFor(Stream* other) {
  auto event = std::make_shared<absl::Notification>();
  static_cast<HostStream*>(other)->EnqueueTask([event]() { event->Notify(); });
  EnqueueTask([event]() { event->WaitForNotification(); });
  return absl::OkStatus();
}
absl::Status HostStream::WaitFor(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  EnqueueTask([notification]() { notification->WaitForNotification(); });
  return absl::OkStatus();
}
bool HostStream::EnqueueTask(absl::AnyInvocable<void() &&> task) {
  return EnqueueTaskWithStatus([task = std::move(task)]() mutable {
    std::move(task)();
    return absl::OkStatus();
  });
}
absl::Status HostStream::RecordEvent(Event* event) {
  std::shared_ptr<absl::Notification> notification =
      static_cast<HostEvent*>(event)->notification();
  EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return absl::OkStatus();
}
absl::Status HostStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  if (EnqueueTaskWithStatus(std::move(callback))) {
    return absl::OkStatus();
  }
  return absl::InternalError("Failed to host callback.");
}
bool HostStream::EnqueueTaskWithStatus(
    absl::AnyInvocable<absl::Status() &&> task) {
  CHECK(task != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(std::move(task));
  return true;
}
bool HostStream::WorkAvailable() { return !work_queue_.empty(); }
void HostStream::WorkLoop() {
  tsl::port::ScopedFlushDenormal flush;
  tsl::port::ScopedSetRound round(FE_TONEAREST);
  while (true) {
    std::queue<absl::AnyInvocable<absl::Status() &&>> queue;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &HostStream::WorkAvailable));
      std::swap(queue, work_queue_);
    }
    while (!queue.empty()) {
      absl::AnyInvocable<absl::Status() &&>& fn = queue.front();
      if (!fn) {
        return;
      }
      status_.Update(std::move(fn)());
      queue.pop();
    }
  }
}
absl::Status HostStream::BlockUntilDone() {
  absl::Notification done;
  absl::Status status;
  EnqueueTask([&done, &status, this]() {
    status = status_;
    status_ = absl::OkStatus();
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}
absl::Status HostStream::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const Kernel& kernel, const KernelArgs& args) {
  const HostKernel* host_kernel = AsHostKernel(&kernel);
  const KernelArgsDeviceMemoryArray* device_mem =
      DynCast<KernelArgsDeviceMemoryArray>(&args);
  if (device_mem != nullptr) {
    return host_kernel->Launch(thread_dims, device_mem->device_memory_args());
  }
  return absl::UnimplementedError(
      "Host kernel implements Launch method only for DeviceMemoryArray "
      "arguments.");
}
}  
}  