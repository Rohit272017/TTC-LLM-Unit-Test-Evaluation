#include "tensorflow/core/common_runtime/hierarchical_tree_broadcaster.h"
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#define READABLE_KEYS false
namespace tensorflow {
namespace {
string BroadcastBufKey(const string& exec_key, int subdiv, int src_rank,
                       int dst_rank) {
  if (READABLE_KEYS) {
    return strings::StrCat("broadcast(", exec_key, "):subdiv(", subdiv,
                           "):src(", src_rank, "):dst(", dst_rank, ")");
  } else {
    return strings::StrCat(exec_key, ":", subdiv, ":", src_rank, ":", dst_rank);
  }
}
}  
HierarchicalTreeBroadcaster::HierarchicalTreeBroadcaster()
    : col_ctx_(nullptr),
      col_params_(nullptr),
      done_(nullptr),
      is_source_(false) {}
int HierarchicalTreeBroadcaster::GetDeviceTask(
    int device_rank, const std::vector<int>& dev_per_task) {
  int num_tasks = static_cast<int>(dev_per_task.size());
  int task_lo = 0;
  int task_hi = -1;
  for (int ti = 0; ti < num_tasks; ti++) {
    task_hi = task_lo + dev_per_task[ti];
    if (task_lo <= device_rank && device_rank < task_hi) return ti;
    task_lo = task_hi;
  }
  LOG(FATAL) << "Unexpected device rank " << device_rank << " for " << task_hi
             << " devices";
  return -1;
}
Status HierarchicalTreeBroadcaster::InitializeCollectiveParams(
    CollectiveParams* col_params) {
  CHECK_EQ(col_params->instance.type, BROADCAST_COLLECTIVE);
  CHECK_EQ(col_params->instance.impl_details.collective_name,
           "HierarchicalTreeBroadcast");
  const string& device_name =
      col_params->group.members[col_params->default_rank].device.name();
  std::vector<int> dev_per_task;
  const string* prior_task_name = &col_params->group.members[0].task;
  int dev_count = 1;
  for (int di = 1; di < col_params->group.group_size; ++di) {
    if (col_params->group.members[di].task != *prior_task_name) {
      dev_per_task.push_back(dev_count);
      dev_count = 1;
      prior_task_name = &col_params->group.members[di].task;
    } else {
      ++dev_count;
    }
  }
  dev_per_task.push_back(dev_count);
  CHECK_EQ(col_params->group.num_tasks, dev_per_task.size());
  if (VLOG_IS_ON(2)) {
    string dpt_buf;
    for (int dpt : dev_per_task) strings::StrAppend(&dpt_buf, dpt, ";");
    VLOG(2) << "HierarchicalTreeBroadcaster::InitializeCollectiveParams device="
            << device_name << " source_rank=" << col_params->source_rank
            << " dev_per_task=" << dpt_buf;
  }
  int num_tasks = col_params->group.num_tasks;
  int num_subdivs = num_tasks + (num_tasks > 1 ? 1 : 0);
  int total_num_devices = 0;
  for (int num_dev : dev_per_task) total_num_devices += num_dev;
  col_params->instance.impl_details.subdiv_permutations.resize(num_subdivs);
  col_params->subdiv_rank.reserve(num_subdivs);
  col_params->instance.impl_details.subdiv_source_rank.reserve(num_subdivs);
  if (num_tasks > 1) {
    const int sdi = 0;
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    int device_count = 0;
    int source_task = GetDeviceTask(col_params->source_rank, dev_per_task);
    for (int ti = 0; ti < col_params->group.num_tasks; ti++) {
      bool participate = false;
      if (source_task == ti) {
        perm.push_back(col_params->source_rank);
        participate =
            col_params->group.members[col_params->source_rank].device.name() ==
            device_name;
      } else {
        perm.push_back(device_count);
        participate = col_params->group.members[device_count].device.name() ==
                      device_name;
      }
      if (participate) col_params->subdiv_rank.push_back(ti);
      device_count += dev_per_task[ti];
    }
    if (col_params->subdiv_rank.empty()) col_params->subdiv_rank.push_back(-1);
    col_params->instance.impl_details.subdiv_source_rank.push_back(source_task);
  }
  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);
  int abs_di = 0;
  for (int ti = 0; ti < col_params->group.num_tasks; ti++) {
    const int sdi = ti + (num_tasks > 1 ? 1 : 0);
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    bool participate = false;
    int subdiv_source = 0;
    for (int di = 0; di < dev_per_task[ti]; di++) {
      perm.push_back(abs_di);
      if (col_params->group.members[abs_di].device.name() == device_name) {
        participate = true;
        col_params->subdiv_rank.push_back(di);
      }
      if (abs_di == col_params->source_rank) subdiv_source = di;
      abs_di++;
    }
    if (!participate) col_params->subdiv_rank.push_back(-1);
    col_params->instance.impl_details.subdiv_source_rank.push_back(
        subdiv_source);
  }
  for (int sri = 0; sri < num_subdivs; sri++) {
    CHECK_GE(col_params->instance.impl_details.subdiv_source_rank[sri], 0);
  }
  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);
  return absl::OkStatus();
}
Status HierarchicalTreeBroadcaster::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
  CHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}
void HierarchicalTreeBroadcaster::Run(StatusCallback done) {
  CHECK(col_ctx_);
  CHECK(col_params_);
  done_ = std::move(done);
  is_source_ = col_params_->is_source;
  RunTree();
}
int HierarchicalTreeBroadcaster::TreeRecvFrom(const CollectiveParams& cp,
                                              int subdiv) {
  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return -1;
  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];
  if (my_rank == source_rank) return -1;
  if (source_rank == 0) {
    return (my_rank - 1) / 2;
  } else {
    int predecessor_rank = (my_rank / 2) - 1;
    return (predecessor_rank < 0) ? source_rank : predecessor_rank;
  }
}
void HierarchicalTreeBroadcaster::TreeSendTo(const CollectiveParams& cp,
                                             int subdiv,
                                             std::vector<int>* targets) {
  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return;
  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];
  int group_size = 0;
  for (int i = 0; i < impl.subdiv_permutations[subdiv].size(); i++) {
    if (impl.subdiv_permutations[subdiv][i] >= 0) {
      group_size++;
    }
  }
  targets->clear();
  int successor_rank = 0;
  if (source_rank == 0) {
    successor_rank = (2 * my_rank) + 1;
  } else {
    successor_rank = (2 * (my_rank + 1));
  }
  DCHECK_NE(successor_rank, my_rank);
  if (cp.is_source && source_rank != 0) {
    if (group_size > 1) {
      targets->push_back(0);
    }
    if (group_size > 2 && source_rank != 1) {
      targets->push_back(1);
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (successor_rank < group_size && successor_rank != source_rank) {
      targets->push_back(successor_rank);
    }
    ++successor_rank;
  }
}
void HierarchicalTreeBroadcaster::RunTree() {
  int num_subdivs = static_cast<int>(col_params_->subdiv_rank.size());
  for (int si = 0; si < num_subdivs; si++) {
    int my_rank = col_params_->subdiv_rank[si];
    if (-1 == my_rank) continue;
    int source_rank = col_params_->instance.impl_details.subdiv_source_rank[si];
    if (VLOG_IS_ON(1)) {
      string subdiv_buf;
      for (int r : col_params_->instance.impl_details.subdiv_permutations[si]) {
        strings::StrAppend(&subdiv_buf, r, ",");
      }
      VLOG(1) << "Running Broadcast tree device=" << col_ctx_->device_name
              << " subdiv=" << si << " perm=" << subdiv_buf
              << " my_rank=" << my_rank << " source_rank=" << source_rank;
    }
    mutex mu;               
    int pending_count = 0;  
    condition_variable all_done;
    if (my_rank >= 0 && my_rank != source_rank) {
      tsl::profiler::TraceMe activity(
          [&] { return strings::StrCat("ReceiveValue:", si); },
          tsl::profiler::TraceMeLevel::kInfo);
      int recv_from_rank = TreeRecvFrom(*col_params_, si);
      Notification note;
      DispatchRecv(si, recv_from_rank, my_rank, col_ctx_->output,
                   [this, &mu, &note](const Status& s) {
                     mutex_lock l(mu);
                     status_.Update(s);
                     note.Notify();
                   });
      note.WaitForNotification();
    }
    {
      tsl::profiler::TraceMe activity(
          [&] { return strings::StrCat("ForwardValue:", si); },
          tsl::profiler::TraceMeLevel::kInfo);
      if (my_rank >= 0 && status_.ok()) {
        std::vector<int> send_to_ranks;
        TreeSendTo(*col_params_, si, &send_to_ranks);
        for (int i = 0; i < send_to_ranks.size(); ++i) {
          int target_rank = send_to_ranks[i];
          {
            mutex_lock l(mu);
            ++pending_count;
          }
          DispatchSend(si, target_rank, my_rank,
                       (is_source_ ? col_ctx_->input : col_ctx_->output),
                       [this, &mu, &pending_count, &all_done](const Status& s) {
                         mutex_lock l(mu);
                         status_.Update(s);
                         --pending_count;
                         if (pending_count == 0) {
                           all_done.notify_all();
                         }
                       });
        }
      }
      if (status_.ok() && is_source_ && (1 == num_subdivs || 0 != si)) {
        VLOG(2) << "copying input to output for device="
                << col_ctx_->device_name << " subdiv=" << si;
        if (col_ctx_->input != col_ctx_->output &&
            (DMAHelper::base(col_ctx_->input) !=
             DMAHelper::base(col_ctx_->output))) {
          {
            mutex_lock l(mu);
            ++pending_count;
          }
          DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
          CollectiveRemoteAccessLocal::MemCpyAsync(
              op_dev_ctx, op_dev_ctx, col_ctx_->device, col_ctx_->device,
              col_ctx_->op_ctx->input_alloc_attr(0),
              col_ctx_->op_ctx->output_alloc_attr(0), col_ctx_->input,
              col_ctx_->output, 0, 
              [this, &mu, &pending_count, &all_done](const Status& s) {
                mutex_lock l(mu);
                status_.Update(s);
                --pending_count;
                if (0 == pending_count) {
                  all_done.notify_all();
                }
              });
        }
      }
      {
        mutex_lock l(mu);
        if (pending_count > 0) {
          all_done.wait(l);
        }
      }
    }
  }
  VLOG(2) << "device=" << col_ctx_->device_name << " return status " << status_;
  done_(status_);
}
void HierarchicalTreeBroadcaster::DispatchSend(int subdiv, int dst_rank,
                                               int src_rank,
                                               const Tensor* src_tensor,
                                               const StatusCallback& done) {
  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
      col_params_->name.data(), col_ctx_->step_id, "dynamic",
      src_tensor->dtype(),
      [src_tensor]() { return src_tensor->shape().DebugString(); });
  string send_buf_key =
      BroadcastBufKey(col_ctx_->exec_key, subdiv, src_rank, dst_rank);
  int dst_idx =
      col_params_->instance.impl_details.subdiv_permutations[subdiv][dst_rank];
  VLOG(3) << "DispatchSend " << send_buf_key << " from_device "
          << col_ctx_->device_name << " to_device "
          << col_params_->group.members[dst_idx].device.name()
          << " subdiv=" << subdiv << " dst_rank=" << dst_rank
          << " dst_idx=" << dst_idx;
  col_ctx_->col_exec->remote_access()->PostToPeer(
      col_params_->group.members[dst_idx].device.name(),
      col_params_->group.members[dst_idx].task, send_buf_key, col_ctx_->device,
      col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), src_tensor,
      col_ctx_->device_locality, col_ctx_->op_ctx->cancellation_manager(),
      done);
}
void HierarchicalTreeBroadcaster::DispatchRecv(int subdiv, int src_rank,
                                               int dst_rank, Tensor* dst_tensor,
                                               const StatusCallback& done) {
  string recv_buf_key =
      BroadcastBufKey(col_ctx_->exec_key, subdiv, src_rank, dst_rank);
  int src_idx =
      col_params_->instance.impl_details.subdiv_permutations[subdiv][src_rank];
  VLOG(3) << "DispatchRecv " << recv_buf_key << " from_device "
          << col_params_->group.members[src_idx].device.name() << " to_device "
          << col_ctx_->device_name << " subdiv=" << subdiv
          << " src_rank=" << src_rank << " src_idx=" << src_idx;
  col_ctx_->col_exec->remote_access()->RecvFromPeer(
      col_params_->group.members[src_idx].device.name(),
      col_params_->group.members[src_idx].task,
      col_params_->group.members[src_idx].is_local, recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), dst_tensor,
      col_ctx_->device_locality, 0 ,
      col_ctx_->op_ctx->cancellation_manager(), done);
}
namespace {
REGISTER_COLLECTIVE(HierarchicalTreeBroadcast, HierarchicalTreeBroadcaster);
}  
}  