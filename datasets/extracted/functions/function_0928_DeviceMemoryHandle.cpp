#include "xla/stream_executor/device_memory_handle.h"
#include <utility>
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
namespace stream_executor {
DeviceMemoryHandle::DeviceMemoryHandle(StreamExecutor *executor,
                                       DeviceMemoryBase memory)
    : memory_(std::move(memory)), executor_(executor) {}
DeviceMemoryHandle::DeviceMemoryHandle(DeviceMemoryHandle &&other) noexcept
    : memory_(std::move(other.memory_)), executor_(other.executor_) {
  other.memory_ = DeviceMemoryBase();
}
DeviceMemoryHandle::~DeviceMemoryHandle() { Free(); }
void DeviceMemoryHandle::Free() {
  if (!memory_.is_null()) {
    executor_->Deallocate(&memory_);
  }
}
DeviceMemoryHandle &DeviceMemoryHandle::operator=(
    DeviceMemoryHandle &&other) noexcept {
  Free();
  memory_ = std::move(other.memory_);
  other.memory_ = DeviceMemoryBase();
  executor_ = other.executor_;
  return *this;
}
}  