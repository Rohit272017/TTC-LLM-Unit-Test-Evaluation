#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include <forward_list>
#include <functional>
#include <memory>
#include <utility>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/env_var.h"
namespace tensorflow {
namespace {
bool IsAsyncWaitForRemoteFunctionEnabled() {
  bool enabled = true;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_ASYNC_WAIT_FOR_REMOTE_FUNCTION",
                                 true, &enabled));
  return enabled;
}
}  
EagerExecutor::EagerExecutor(bool async, bool enable_streaming_enqueue,
                             int in_flight_nodes_limit)
    : next_node_id_(0),
      ok_(true),
      thread_(async ? tensorflow::Env::Default()->StartThread(
                          tensorflow::ThreadOptions(), "eager_async_executor",
                          std::bind(&EagerExecutor::Run, this))
                    : nullptr),
      last_eager_client_(nullptr),
      enable_async_wait_for_remote_function_(
          IsAsyncWaitForRemoteFunctionEnabled()),
      enable_streaming_enqueue_(enable_streaming_enqueue),
      in_flight_nodes_limit_(in_flight_nodes_limit) {
  if (async && in_flight_nodes_limit_ > 0) {
    VLOG(4) << "EagerExecutor InFlightNodes limit is set to "
            << in_flight_nodes_limit_;
  }
}
EagerExecutor::~EagerExecutor() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  state_ = ExecutorState::kShutDown;
  nodes_pending_.notify_all();
  for (const auto& cleanups_for_key : cleanups_) {
    for (const std::function<void()>& cleanup : cleanups_for_key.second) {
      cleanup();
    }
  }
}
Status EagerExecutor::ShutDown() {
  {
    bool has_thread;
    Status status;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      if (state_ != ExecutorState::kShutDown) {
        state_ = ExecutorState::kShuttingDown;
      }
      WaitForAllPendingNodesLocked(&l).IgnoreError();
      state_ = ExecutorState::kShutDown;
      has_thread = thread_ != nullptr;
      status = status_;
      if (has_thread) {
        nodes_pending_.notify_all();
      }
    }
    if (!has_thread) {
      return status;
    }
  }
  thread_exited_notification_.WaitForNotification();
  return status();
}
const char* EagerExecutor::StateStringLocked() {
  switch (state_) {
    case ExecutorState::kActive:
      return "Active";
    case ExecutorState::kShuttingDown:
      return "ShuttingDown";
    case ExecutorState::kShutDown:
      return "ShutDown";
  }
}
Status EagerExecutor::SyncExecute(EagerNode* node) {
  if (Async()) {
    return errors::Internal("SyncExecute does not support async execution.");
  }
  if (node->AsAsync() != nullptr) {
    return errors::Internal("Executor does not support executing async nodes");
  }
  uint64 id = next_node_id_++;
  Status s = node->Prepare();
  if (!s.ok()) {
    return s;
  }
  s = node->Run();
  tensorflow::mutex_lock l(node_queue_mutex_);
  NotifyWaiters(id);
  return s;
}
Status EagerExecutor::AddOrExecute(std::unique_ptr<EagerNode> node) {
  Status status;
  core::RefCountPtr<NodeItem> item(new NodeItem);
  item->id = next_node_id_++;
  item->node = std::move(node);
  item->state = NodeState::kPENDING;
  status = item->node->Prepare();
  if (!status.ok()) {
    item->node->Abort(status);
    return status;
  }
  if (!Async()) {
    return RunItem(std::move(item), false);
  } else {
    tensorflow::mutex_lock l(node_queue_mutex_);
    DVLOG(3) << "Add node [id " << item->id << "]" << item->node->DebugString()
             << " with status: " << status_;
    if (state_ != ExecutorState::kActive) {
      status = errors::FailedPrecondition(
          "EagerExecutor accepts new EagerNodes to run only in Active state. "
          "Current state is '",
          StateStringLocked(), "'");
    } else {
      status = status_;
      if (status.ok()) {
        node_queue_.push(std::move(item));
        if (node_queue_.size() == 1) {
          nodes_pending_.notify_all();
        }
        if (in_flight_nodes_limit_ == 0) {
          return absl::OkStatus();
        }
        while (true) {
          int64_t in_flight_nodes_count =
              node_queue_.size() + unfinished_nodes_.size();
          if (in_flight_nodes_count < in_flight_nodes_limit_) {
            break;
          }
          VLOG(4) << "Hitting in-flight node limit node_queue_.size() = "
                  << node_queue_.size()
                  << " unfinished_nodes_.size() = " << unfinished_nodes_.size()
                  << ".";
          nodes_done_.wait(l);
        }
        return absl::OkStatus();
      }
    }
  }
  item->node->Abort(status);
  return status;
}
tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  return WaitForAllPendingNodesLocked(&l);
}
tensorflow::Status EagerExecutor::WaitForAllPendingNodesLocked(
    mutex_lock* lock) {
  tensorflow::condition_variable cond;
  if (!status_.ok()) return status_;
  if (node_queue_.empty() && unfinished_nodes_.empty()) return absl::OkStatus();
  DCHECK(Async() || node_queue_.empty());
  auto last_id = next_node_id_ - 1;
  DVLOG(3) << "Wait for Node: [id " << last_id << "] ";
  node_done_notifications_.insert(std::make_pair(last_id, &cond));
  cond.wait(*lock);
  return status_;
}
void EagerExecutor::ClearError() {
  if (ok()) return;
  tensorflow::mutex_lock l(node_queue_mutex_);
  DCHECK(node_done_notifications_.empty());
  DCHECK(node_queue_.empty());
  status_ = absl::OkStatus();
  ok_ = true;
  last_eager_client_ = nullptr;
  nodes_pending_.notify_all();
}
void EagerExecutor::NodeDone(const core::RefCountPtr<NodeItem>& item,
                             const Status& status, bool from_queue) {
  DVLOG(3) << "Node Done: [id " << item->id << "] " << item->node->DebugString()
           << " with status: " << status;
  DCHECK(item->state != NodeState::kDONE);
  item->state = NodeState::kDONE;
  bool async = item->node->AsAsync() != nullptr;
  if (status.ok() && !from_queue && !async) {
    return;
  }
  std::forward_list<core::RefCountPtr<NodeItem>> items_to_destroy;
  {
    mutex_lock l(node_queue_mutex_);
    if (!status_.ok()) return;
    bool need_notification = from_queue;
    if (from_queue) {
      DCHECK(!node_queue_.empty() && item.get() == node_queue_.front().get());
      node_queue_.pop();
    } else if (async) {
      need_notification = item->id == unfinished_nodes_.begin()->first;
      auto result = unfinished_nodes_.erase(item->id);
      if (result == 0) return;
    }
    if (!status.ok() && item->node->Fatal()) {
      need_notification = true;
      status_ = status;
      ok_ = false;
      if (Async()) {
        errors::AppendToMessage(&status_,
                                "Encountered when executing an operation using "
                                "EagerExecutor. This error cancels all future "
                                "operations and poisons their output tensors.");
      }
      while (!node_queue_.empty()) {
        items_to_destroy.push_front(std::move(node_queue_.front()));
        node_queue_.pop();
      }
      for (auto& it : unfinished_nodes_) {
        items_to_destroy.push_front(std::move(it.second));
      }
      unfinished_nodes_.clear();
    }
    if (need_notification) {
      NotifyWaiters(item->id);
    }
    nodes_done_.notify_all();
  }
  for (auto& item : items_to_destroy) {
    item->node->Abort(status);
  }
}
void EagerExecutor::NotifyWaiters(uint64 id) {
  if (!node_done_notifications_.empty()) {
    uint64 upperbound_id = 0;
    if (!unfinished_nodes_.empty()) {
      upperbound_id = unfinished_nodes_.begin()->first - 1;
    } else if (!node_queue_.empty()) {
      upperbound_id = node_queue_.front()->id - 1;
    } else {
      upperbound_id = next_node_id_ - 1;
    }
    if (upperbound_id < id) {
      return;
    }
    DVLOG(3) << "Notify node done: [id " << id << " to " << upperbound_id
             << "] ";
    const auto range =
        status_.ok() ? std::make_pair(
                           node_done_notifications_.lower_bound(id),
                           node_done_notifications_.upper_bound(upperbound_id))
                     : std::make_pair(node_done_notifications_.begin(),
                                      node_done_notifications_.end());
    for (auto it = range.first; it != range.second; ++it) {
      it->second->notify_all();
    }
    node_done_notifications_.erase(range.first, range.second);
  }
}
void EagerExecutor::Run() {
  auto thread_exited_notifier =
      gtl::MakeCleanup([this] { thread_exited_notification_.Notify(); });
  while (true) {
    core::RefCountPtr<NodeItem> curr_item;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      while (node_queue_.empty() || !status_.ok()) {
        if (state_ == ExecutorState::kShutDown) return;
        nodes_pending_.wait(l);
      }
      curr_item.reset(node_queue_.front().get());
      curr_item->Ref();
    }
    Status status = RunItem(std::move(curr_item), true);
    if (!status.ok()) {
      VLOG(1) << "Failed to run item: " << status;
    }
  }
}
Status EagerExecutor::RunItem(core::RefCountPtr<NodeItem> item,
                              bool from_queue) {
  DVLOG(3) << "Running Node: [id " << item->id << "] "
           << item->node->DebugString();
  AsyncRemoteExecuteNode* async_remote_node =
      item->node->AsAsyncRemoteExecuteNode();
  if (enable_async_wait_for_remote_function_) {
    if (async_remote_node != nullptr) {
      if (last_eager_client_ != nullptr &&
          async_remote_node->eager_client() != nullptr &&
          last_eager_client_ != async_remote_node->eager_client()) {
        DVLOG(3) << "Executing Sync Executor for node" << item->id;
        tensorflow::Status status = async_remote_node->SyncExecutors();
        if (!status.ok()) {
          NodeDone(item, status, from_queue);
          return status;
        }
        last_eager_client_ = nullptr;
      }
      if (async_remote_node->eager_client() != nullptr &&
          async_remote_node->needs_remote_inputs() &&
          async_remote_node->allow_multiple_pending_requests()) {
        last_eager_client_ = async_remote_node->eager_client();
      }
    }
  }
  AsyncEagerNode* async_node = item->node->AsAsync();
  if (async_node == nullptr) {
    tensorflow::Status status = item->node->Run();
    NodeDone(item, status, from_queue);
    return status;
  }
  item->state = NodeState::kSCHEDULED;
  auto async_ref = item.get();
  async_ref->Ref();
  TF_RETURN_IF_ERROR(MoveToUnfinished(std::move(item), from_queue));
  async_node->RunAsync([this, async_ref](const Status& status) {
    core::RefCountPtr<NodeItem> async_item(async_ref);
    NodeDone(async_item, status, false);
  });
  return status();
}
Status EagerExecutor::MoveToUnfinished(core::RefCountPtr<NodeItem> item,
                                       bool from_queue) {
  tensorflow::mutex_lock l(node_queue_mutex_);
  if (!status_.ok()) {
    return status_;
  }
  if (from_queue) {
    DCHECK(!node_queue_.empty() && item.get() == node_queue_.front().get());
    node_queue_.pop();
  }
  DVLOG(3) << "Add Node: [id " << item->id << "] to unfinished map.";
  unfinished_nodes_.emplace_hint(unfinished_nodes_.end(), item->id,
                                 std::move(item));
  return absl::OkStatus();
}
void EagerExecutor::AddCleanup(intptr_t key, std::function<void()> callback) {
  cleanups_[key].push_back(callback);
}
void EagerExecutor::RemoveCleanups(intptr_t key) { cleanups_.erase(key); }
}  