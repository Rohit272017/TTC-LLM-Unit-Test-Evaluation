#include "xla/service/gpu/runtime/command_buffer_thunk.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/command_buffer_cmd.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
namespace xla::gpu {
using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;
CommandBufferThunk::ExecutorCommandBuffer::ExecutorCommandBuffer(
    std::unique_ptr<se::CommandBuffer> command_buffer)
    : command_buffer(std::move(command_buffer)) {}
CommandBufferThunk::CommandBufferThunk(
    CommandBufferCmdSequence commands, ThunkInfo thunk_info,
    std::unique_ptr<SequentialThunk> thunks,
    bool enable_command_buffers_during_profiling)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      thunks_(std::move(thunks)),
      enable_command_buffers_during_profiling_(
          enable_command_buffers_during_profiling),
      state_(std::make_shared<State>()) {
  EvictCommandBuffers();
  TrackCommandBuffers(state_);
}
bool CommandBufferThunk::ExecutorCommandBuffer::ShouldUpdateCommandBuffer(
    const CommandBufferCmdSequence& commands,
    const Thunk::ExecuteParams& params) {
  if (commands.force_update()) {
    return true;
  }
  bool should_update = false;
  const BufferAllocations* allocs = params.buffer_allocations;
  for (BufferAllocation::Index index : commands.allocs_indices()) {
    se::DeviceMemoryBase alloc = allocs->GetDeviceAddress(index);
    if (recorded_allocs.size() <= index) {
      recorded_allocs.resize(index + 1);
      should_update = true;
    }
    if (!recorded_allocs[index].IsSameAs(alloc)) {
      recorded_allocs[index] = alloc;
      should_update = true;
    }
  }
  return should_update;
}
absl::Status CommandBufferThunk::Prepare(const PrepareParams& params,
                                         ResourceRequests& resource_requests) {
  if (commands_.empty()) return absl::OkStatus();
  TF_RETURN_IF_ERROR(commands_.Prepare(params, resource_requests));
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}
absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  if (commands_.empty()) return absl::OkStatus();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(params.executor));
  absl::MutexLock lock(&cmd_buffer->mutex);
  TF_RETURN_IF_ERROR(commands_.Initialize(params, cmd_buffer->state));
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Initialize(params));
  }
  Thunk::ExecuteParams execute_params(
      params.buffer_allocations, params.stream,
      params.command_buffer_trace_stream, params.collective_params,
      params.collective_cliques, nullptr,
      nullptr,
      nullptr,
      nullptr, params.ffi_execution_context);
  if (cmd_buffer->command_buffer->state() ==
          se::CommandBuffer::State::kCreate &&
      cmd_buffer->ShouldUpdateCommandBuffer(commands_, execute_params)) {
    VLOG(3) << "Initialize command buffer on device #"
            << params.executor->device_ordinal()
            << " by recoding command buffer cmd sequence"
            << "; num_commands=" << commands_.size();
    TraceMe trace([&] {
      return TraceMeEncode("command_buffer::initialize",
                           {{"device", params.executor->device_ordinal()},
                            {"num_commands", commands_.size()}});
    });
    uint64_t start_micros = tsl::Env::Default()->NowMicros();
    CommandBufferCmd::RecordParams record_params = {cmd_buffer->state};
    TF_RETURN_IF_ERROR(commands_.Record(execute_params, record_params,
                                        cmd_buffer->command_buffer.get()));
    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    VLOG(3) << "Initialized command buffer on device #"
            << params.executor->device_ordinal() << " in "
            << (end_micros - start_micros)
            << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }
  return absl::OkStatus();
}
absl::Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  if (commands_.empty()) return absl::OkStatus();
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Execute command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TF_RETURN_IF_ERROR(thunks_->ExecuteOnStream(params));
    return absl::OkStatus();
  }
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(executor));
  absl::MutexLock lock(&cmd_buffer->mutex);
  if (cmd_buffer->ShouldUpdateCommandBuffer(commands_, params)) {
    VLOG(3) << "Update command buffer on device #" << executor->device_ordinal()
            << " by recoding command buffer cmd sequence" << " after "
            << cmd_buffer->num_executions << " executions since last update"
            << "; num_commands=" << commands_.size();
    TraceMe trace([&] {
      cmd_buffer->mutex.AssertHeld();
      return TraceMeEncode("command_buffer::update",
                           {{"device", executor->device_ordinal()},
                            {"num_commands", commands_.size()},
                            {"num_executions", cmd_buffer->num_executions}});
    });
    uint64_t start_micros = tsl::Env::Default()->NowMicros();
    CommandBufferCmd::RecordParams record_params = {cmd_buffer->state};
    TF_RETURN_IF_ERROR(commands_.Record(params, record_params,
                                        cmd_buffer->command_buffer.get()));
    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    VLOG(3) << "Updated command buffer in " << (end_micros - start_micros)
            << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }
  ++cmd_buffer->num_executions;
  VLOG(3) << "Execute command buffer on device #" << executor->device_ordinal()
          << "; num_executions=" << cmd_buffer->num_executions;
  TraceMe trace([&] {
    cmd_buffer->mutex.AssertHeld();
    return TraceMeEncode("command_buffer::execute",
                         {{"device", executor->device_ordinal()},
                          {"num_commands", commands_.size()},
                          {"num_executions", cmd_buffer->num_executions}});
  });
  return cmd_buffer->command_buffer->Submit(params.stream);
}
absl::StatusOr<std::shared_ptr<CommandBufferThunk::ExecutorCommandBuffer>>
CommandBufferThunk::GetOrCreateCommandBuffer(se::StreamExecutor* executor) {
  absl::MutexLock lock(&state_->mutex);
  if (auto it = state_->command_buffers.find(executor);
      it != state_->command_buffers.end()) {
    return it->second;
  }
  TF_ASSIGN_OR_RETURN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  auto emplaced = state_->command_buffers.emplace(
      executor,
      std::make_shared<ExecutorCommandBuffer>(std::move(command_buffer)));
  return emplaced.first->second;
}
struct CommandBufferThunk::GlobalState {
  absl::Mutex mutex;
  std::vector<std::weak_ptr<CommandBufferThunk::State>> state
      ABSL_GUARDED_BY(mutex);
};
CommandBufferThunk::GlobalState* CommandBufferThunk::GetGlobalState() {
  static auto* global_state = new GlobalState();
  return global_state;
}
void CommandBufferThunk::TrackCommandBuffers(
    std::weak_ptr<CommandBufferThunk::State> state) {
  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(&global_state->mutex);
  global_state->state.push_back(state);
}
void CommandBufferThunk::EvictCommandBuffers() {
  TraceMe trace([&] { return "EvictCommandBuffers"; });
  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(&global_state->mutex);
  VLOG(3) << "Evict command buffer thunk command buffers; tracked thunks = "
          << global_state->state.size();
  global_state->state.erase(
      std::remove_if(global_state->state.begin(), global_state->state.end(),
                     [](auto& weak_ptr) { return weak_ptr.expired(); }),
      global_state->state.end());
  int64_t num_evicted = 0;
  for (auto& weak_ptr : global_state->state) {
    auto ptr = weak_ptr.lock();
    if (!ptr) continue;
    absl::MutexLock state_lock(&ptr->mutex);
    num_evicted += ptr->command_buffers.size();
    ptr->command_buffers.clear();
  }
  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}
}  