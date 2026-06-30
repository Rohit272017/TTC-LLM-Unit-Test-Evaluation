#include "tensorflow/core/nccl/nccl_manager.h"
#include <utility>
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "absl/base/call_once.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#if GOOGLE_CUDA
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif
namespace tensorflow {
using stream_executor::gpu::ScopedActivateContext;
#if TENSORFLOW_USE_ROCM
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
int NcclManager::instance_count = 0;
#endif
#define NCCL_RETURN_IF_ERROR(...)                                        \
  do {                                                                   \
    ncclResult_t nccl_status = (__VA_ARGS__);                            \
    if (nccl_status != ncclSuccess) {                                    \
      return errors::Internal("NCCL: ", ncclGetErrorString(nccl_status), \
                              ". Set NCCL_DEBUG=WARN for detail.");      \
    }                                                                    \
  } while (0)
#define CUDA_RETURN_IF_ERROR(...)                                         \
  do {                                                                    \
    cudaError_t cuda_status = (__VA_ARGS__);                              \
    if (cuda_status != cudaSuccess) {                                     \
      return errors::Internal("CUDA: ", cudaGetErrorString(cuda_status)); \
    }                                                                     \
  } while (0)
struct NcclManager::NcclStream : public core::RefCounted {
 public:
  NcclStream() = default;
  ~NcclStream() = default;
  se::StreamExecutor* executor = nullptr;
#if TENSORFLOW_USE_ROCM
  se::Stream* stream = nullptr;
#else
  std::unique_ptr<se::Stream> stream;
#endif
  mutex mu;
  condition_variable cv;
  std::deque<std::pair<Collective*, int>> pending_launches_ TF_GUARDED_BY(mu);
  bool shutdown_requested TF_GUARDED_BY(mu) = false;
};
struct NcclManager::CommunicatorMember {
 public:
  CommunicatorMember() {}
  ~CommunicatorMember() {
    if (nccl_comm != nullptr) ncclCommDestroy(nccl_comm);
  }
  ncclComm_t nccl_comm = nullptr;
  NcclStream* nccl_stream = nullptr;
};
struct NcclManager::Communicator {
 public:
  explicit Communicator(std::vector<CommunicatorMember> members,
                        const string& key)
      : num_devices(members.size()), members(std::move(members)), key(key) {}
  const int num_devices;
  std::vector<CommunicatorMember> members;
  const string key;
};
namespace {
static constexpr DataTypeSet kValidDataTypes =
    ToSet(DT_HALF) | ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) |
    ToSet(DT_INT64);
ncclDataType_t ToNcclType(DataType t) {
  switch (t) {
    case DT_HALF:
      return ncclHalf;
    case DT_FLOAT:
      return ncclFloat;
    case DT_DOUBLE:
      return ncclDouble;
    case DT_INT32:
      return ncclInt;
    case DT_INT64:
      return ncclInt64;
    default:
      return ncclFloat;
  }
}
void StringToNcclUniqueId(const string& str_id, ncclUniqueId* nccl_id) {
  if (str_id.size() == NCCL_UNIQUE_ID_BYTES) {
    memcpy(nccl_id->internal, str_id.data(), NCCL_UNIQUE_ID_BYTES);
  }
}
}  
struct NcclManager::Collective : public core::RefCounted {
  Collective(const string& collective_key_in, DataType data_type_in,
             CollectiveType type_in, ncclRedOp_t reduction_op_in,
             int num_local_devices_in, int num_global_devices_in,
             const string& communicator_key_in)
      : collective_key(collective_key_in),
        data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        num_local_devices(num_local_devices_in),
        num_global_devices(num_global_devices_in),
        single_node(num_local_devices_in == num_global_devices_in),
        communicator_key(communicator_key_in) {
    participants.reserve(num_local_devices_in);
#if TENSORFLOW_USE_ROCM
    if (NcclManager::instance_count > 1) {
      status = errors::Internal(
          "ROCm cannot use multi-node NCCL collectives on a single node");
    }
#endif
  }
  const string collective_key;  
  const DataType data_type;
  const CollectiveType type;
  const ncclRedOp_t reduction_op;  
  const int num_local_devices;     
  const int num_global_devices;    
  const bool single_node;          
  const string communicator_key;
  Communicator* communicator = nullptr;
  std::vector<std::unique_ptr<Participant>> participants;
  int root_rank = -1;
  int available_participants = 0;
  bool multi_node_ready = false;
  uint64 trace_context = 0;
  Status status;
};
NcclManager::NcclManager() {
  VLOG(2) << "New NcclManager " << this;
#if TENSORFLOW_USE_ROCM
  ++instance_count;
#endif
}
NcclManager::~NcclManager() {
  VLOG(2) << "~NcclManager " << this;
#if TENSORFLOW_USE_ROCM
  --instance_count;
#endif
  for (auto& it : device_to_comm_streams_) {
    for (NcclStream* nccl_stream : it.second) {
      {
        mutex_lock l(nccl_stream->mu);
        nccl_stream->shutdown_requested = true;
        nccl_stream->cv.notify_all();
      }
      nccl_stream->Unref();
    }
  }
}
NcclManager* NcclManager::instance() {
  static NcclManager* instance = new NcclManager();
#if TENSORFLOW_USE_ROCM
  static absl::once_flag once;
  absl::call_once(once, [] { --NcclManager::instance_count; });
#endif
  return instance;
}
string NcclManager::GenerateCommunicatorKey() {
  ncclUniqueId nccl_id;
  ncclGetUniqueId(&nccl_id);
  return string(nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
}
Status NcclManager::GetCommunicator(NcclManager::Collective* collective,
                                    NcclManager::Communicator** communicator) {
  std::sort(collective->participants.begin(), collective->participants.end(),
            [](const std::unique_ptr<Participant>& a,
               const std::unique_ptr<Participant>& b) {
              if (a->gpu_device_id != b->gpu_device_id) {
                return a->gpu_device_id < b->gpu_device_id;
              }
              if (a->executor != b->executor) {
                return a->executor < b->executor;
              }
              return a->global_rank < b->global_rank;
            });
  mutex_lock l(mu_);
  if (!status_.ok()) {
    return status_;
  }
  if (collective->communicator_key.empty()) {
    for (auto& comm : communicators_) {
      if (comm->num_devices == collective->num_global_devices) {
        int i;
        for (i = 0; i < collective->num_local_devices; ++i) {
          if (comm->members[i].nccl_stream->executor !=
              collective->participants[i]->executor) {
            break;
          }
        }
        if (i == collective->num_local_devices) {
          *communicator = comm.get();
          return OkStatus();
        }
      }
    }
  } else {
#if NCCL_MAJOR < 2
    return errors::Internal(
        "Cannot use multi-node NCCL collectives with NCCL 1.x");
#endif
    if (collective->communicator_key.size() != NCCL_UNIQUE_ID_BYTES) {
      return errors::Internal("Expected communicator_key of size ",
                              NCCL_UNIQUE_ID_BYTES, " but found size ",
                              collective->communicator_key.size());
    }
    for (auto& comm : communicators_) {
      if (comm->key == collective->communicator_key) {
        *communicator = comm.get();
        return OkStatus();
      }
    }
  }
  auto* env = Env::Default();
  std::set<NcclStream*> used_streams;
  std::vector<CommunicatorMember> members(collective->num_local_devices);
  std::vector<int> devices(collective->num_local_devices);
  for (int i = 0; i < collective->num_local_devices; ++i) {
    auto* executor = collective->participants[i]->executor;
    auto& streams = device_to_comm_streams_[executor];
    NcclStream* nccl_stream = nullptr;
    for (const auto& s : streams) {
      if (used_streams.insert(s).second) {
        nccl_stream = s;
        break;
      }
    }
    if (nccl_stream == nullptr) {
      nccl_stream = new NcclStream();
      nccl_stream->executor = executor;
#if TENSORFLOW_USE_ROCM
      nccl_stream->stream = collective->participants[i]->context->nccl_stream();
#else
      TF_ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
      nccl_stream->stream = std::move(stream);
#endif
      streams.emplace_back(nccl_stream);
      used_streams.insert(nccl_stream);
      nccl_stream->Ref();
      env->SchedClosure([this, nccl_stream]() {
        LoopKernelLaunches(nccl_stream);
        nccl_stream->Unref();
      });
    }
    members[i].nccl_stream = nccl_stream;
    devices[i] = collective->participants[i]->gpu_device_id;
  }
  std::vector<ncclComm_t> nccl_comms(collective->num_local_devices);
  VLOG(2) << "Created nccl Communicator with "
          << "num_global_devices = " << collective->num_global_devices
          << " num_local_devices = " << collective->num_local_devices
          << " communicator_key ="
          << absl::StrJoin(
                 std::vector<int>{collective->communicator_key.begin(),
                                  collective->communicator_key.end()},
                 " ");
#if NCCL_MAJOR >= 2
  ncclUniqueId nccl_id;
  if (collective->single_node) {
    NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
  } else {
    StringToNcclUniqueId(collective->communicator_key, &nccl_id);
  }
  int saved_device = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&saved_device));
  NCCL_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < collective->num_local_devices; ++i) {
    const int rank = collective->participants[i]->global_rank >= 0
                         ? collective->participants[i]->global_rank
                         : i;
    CUDA_RETURN_IF_ERROR(cudaSetDevice(devices[i]));
    NCCL_RETURN_IF_ERROR(ncclCommInitRank(
        nccl_comms.data() + i, collective->num_global_devices, nccl_id, rank));
  }
  NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  CUDA_RETURN_IF_ERROR(cudaSetDevice(saved_device));
#else
  NCCL_RETURN_IF_ERROR(ncclCommInitAll(
      nccl_comms.data(), collective->num_local_devices, devices.data()));
#endif
  for (int i = 0; i < collective->num_local_devices; ++i) {
    members[i].nccl_comm = nccl_comms[i];
  }
  communicators_.emplace_back(
      new Communicator(std::move(members), collective->communicator_key));
  *communicator = communicators_.back().get();
  return OkStatus();
}
void NcclManager::AddToAllReduce(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kAllReduce, reduction_op);
}
void NcclManager::AddToAllGather(std::unique_ptr<Participant> participant,
                                 const Context& context) {
  AddParticipant(std::move(participant), context, kAllGather,
                 ncclSum );
}
void NcclManager::AddToReduceScatter(std::unique_ptr<Participant> participant,
                                     const Context& context,
                                     ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kReduceScatter, reduction_op);
}
void NcclManager::AddToAllToAll(std::unique_ptr<Participant> participant,
                                const Context& context) {
  AddParticipant(std::move(participant), context, kAllToAll,
                 ncclSum );
}
void NcclManager::AddBroadcastSend(std::unique_ptr<Participant> participant,
                                   const Context& context) {
  participant->root = true;
  AddParticipant(std::move(participant), context, kBroadcast,
                 ncclSum );
}
void NcclManager::AddBroadcastRecv(std::unique_ptr<Participant> participant,
                                   const Context& context) {
  AddParticipant(std::move(participant), context, kBroadcast,
                 ncclSum );
}
void NcclManager::AddReduceSend(std::unique_ptr<Participant> participant,
                                const Context& context,
                                ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kReduce, reduction_op);
}
void NcclManager::AddReduceRecv(std::unique_ptr<Participant> participant,
                                const Context& context,
                                ncclRedOp_t reduction_op) {
  participant->root = true;
  AddParticipant(std::move(participant), context, kReduce, reduction_op);
}
void NcclManager::SignalMultiNodeReady(const string& collective_key) {
  Collective* to_run = nullptr;
  {
    mutex_lock l(mu_);
    auto collective_it = collectives_.find(collective_key);
    if (collective_it != collectives_.end()) {
      Collective* collective = collective_it->second;
      collective->multi_node_ready = true;
      if (CheckReady(collective_key, collective)) {
        to_run = collective;
      }
      VLOG(2) << "SignalMultiNodeReady collective " << collective_key
              << " to_run " << to_run;
    }
  }
  if (to_run != nullptr) RunCollective(to_run);
}
void NcclManager::AddParticipant(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 CollectiveType collective_type,
                                 ncclRedOp_t reduction_op) {
  Collective* to_run = nullptr;
  DataType data_type;
  Status nccl_manager_status;
  if (participant->input != nullptr) {
    data_type = participant->input->dtype();
  } else {
    data_type = participant->output->dtype();
  }
  {
    mutex_lock l(mu_);
    nccl_manager_status = status_;
    if (nccl_manager_status.ok()) {
      auto collective_it = collectives_.find(context.collective_key);
      Collective* collective = nullptr;
      if (collective_it == collectives_.end()) {
        collective = new Collective(
            context.collective_key, data_type, collective_type, reduction_op,
            context.num_local_devices, context.num_global_devices,
            context.communicator_key);
        collectives_.emplace(context.collective_key, collective);
      } else {
        collective = collective_it->second;
      }
      if (collective->status.ok() && !collective->single_node &&
          collective->communicator_key.empty()) {
        collective->status = errors::Internal(
            "Collective ", reduction_op,
            " is multi node with num_local_devices=",
            collective->num_local_devices,
            " and num_global_devices=", collective->num_global_devices,
            " but has an empty communicator_key");
      }
      if (collective->status.ok() && collective->communicator_key.size() !=
                                         context.communicator_key.size()) {
        collective->status =
            errors::Internal("Collective ", reduction_op,
                             " mismatch in member communicator_key with size ",
                             collective->communicator_key.size(),
                             " and arg communicator_key with size ",
                             context.communicator_key.size());
      }
      if (collective->status.ok() && collective->type != collective_type) {
        collective->status = errors::Internal(
            "Collective ", reduction_op, " previously initialized with type ",
            collective->type, " but now got type ", collective_type);
      }
      if (collective->status.ok() &&
          collective->num_global_devices != context.num_global_devices) {
        collective->status =
            errors::Internal("Collective ", reduction_op,
                             " previously initialized with num_global_devices ",
                             collective->num_global_devices, " but now got ",
                             context.num_global_devices);
      }
      if (collective->status.ok() &&
          collective->num_local_devices != context.num_local_devices) {
        collective->status =
            errors::Internal("Collective ", reduction_op,
                             "previously initialized with num_local_devices ",
                             collective->num_local_devices, " but now got ",
                             context.num_local_devices);
      }
      if (collective->status.ok() &&
          collective->participants.size() >= collective->num_local_devices) {
        collective->status = errors::Internal(
            "Collective ", reduction_op, " expected ",
            collective->num_local_devices, " participants but now has ",
            collective->participants.size(),
            " with one more participant being added");
      }
      if (collective->status.ok() && collective->root_rank >= 0 &&
          context.source_rank >= 0 &&
          collective->root_rank != context.source_rank) {
        collective->status = errors::Internal(
            "Collective ", collective->collective_key,
            " already has root_rank ", collective->root_rank,
            " but new participant has root_rank ", context.source_rank);
      }
      if (collective->status.ok() &&
          !kValidDataTypes.Contains(collective->data_type)) {
        collective->status = errors::Internal(
            "Collective ", collective->collective_key,
            " expected data types compatible with NCCL but instead got ",
            DataTypeString(collective->data_type));
      }
      if (context.source_rank >= 0) {
        collective->root_rank = context.source_rank;
      }
      collective->participants.emplace_back(std::move(participant));
      ++collective->available_participants;
      if (CheckReady(context.collective_key, collective)) {
        to_run = collective;
      }
    }
  }
  if (!nccl_manager_status.ok()) {
    participant->done_callback(nccl_manager_status);
    return;
  }
  if (to_run != nullptr) RunCollective(to_run);
}
bool NcclManager::CheckReady(const string& collective_key,
                             Collective* collective) {
  if (collective->available_participants == collective->num_local_devices) {
    if (collective->num_global_devices == collective->num_local_devices ||
        collective->multi_node_ready) {
      collectives_.erase(collective_key);
      return true;
    }
  }
  return false;
}
void NcclManager::RunCollective(Collective* collective) {
  tensorflow::profiler::TraceMeProducer traceme("Schedule Collective");
  collective->trace_context = traceme.GetContextId();
  static mutex collective_mu(LINKER_INITIALIZED);
  Status status = collective->status;
  if (status.ok()) {
    status = GetCommunicator(collective, &collective->communicator);
  }
  for (int i = 0; status.ok() && i < collective->num_local_devices; ++i) {
    Participant* p = collective->participants[i].get();
    NcclStream* nccl_stream = collective->communicator->members[i].nccl_stream;
    CHECK(nccl_stream != nullptr);
    const int rank = p->global_rank >= 0 ? p->global_rank : i;
    if (p->input != nullptr) {
      status = nccl_stream->stream->WaitFor(p->tensor_stream);
    }
    if (p->root) {
      if (collective->root_rank == -1) {
        collective->root_rank = rank;
      } else if (collective->root_rank != rank) {
        status = errors::Internal(
            "Inconsistent root rank ", collective->root_rank, " and GPU id ",
            p->gpu_device_id, " rank ", rank, " also marked as root.");
      }
    }
    VLOG(2) << "RunCollective rank " << rank << " global_rank "
            << p->global_rank << " root_rank " << collective->root_rank;
  }
  if (status.ok() && collective->type == kBroadcast &&
      collective->root_rank < 0) {
    status = errors::Internal("Root rank not indicated for collective ",
                              collective->collective_key);
  }
  if (!status.ok()) {
    for (int i = 0; i < collective->num_local_devices; ++i) {
      collective->participants[i]->done_callback(status);
    }
    collective->Unref();
    return;
  }
  {
    mutex_lock l(collective_mu);
    for (int i = 0; i < collective->num_local_devices; ++i) {
      NcclStream* nccl_stream =
          collective->communicator->members[i].nccl_stream;
      mutex_lock l(nccl_stream->mu);
      nccl_stream->pending_launches_.push_front(std::make_pair(collective, i));
      collective->Ref();
      nccl_stream->cv.notify_all();
    }
  }
  collective->Unref();
}
namespace {
size_t ComputeBufferSize(const NcclManager::Participant* p,
                         DataType data_type) {
  size_t num_elements = 0;
  if (p->output) {
    num_elements += p->output->NumElements();
  } else if (p->input) {
    num_elements += p->input->NumElements();
  }
  return num_elements * DataTypeSize(data_type);
}
}  
void NcclManager::LoopKernelLaunches(NcclStream* nccl_stream) {
#if TENSORFLOW_USE_ROCM
  se::Stream* comm_stream = nccl_stream->stream;
#else
  se::Stream* comm_stream = nccl_stream->stream.get();
#endif
  ScopedActivateContext scoped_context(nccl_stream->executor);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(
      comm_stream->platform_specific_handle().stream);
  while (true) {
    std::pair<Collective*, int> next_launch;
    {
      VLOG(3) << "Locking mutex nccl_stream " << nccl_stream;
      mutex_lock l(nccl_stream->mu);
      while (nccl_stream->pending_launches_.empty()) {
        if (nccl_stream->shutdown_requested) {
          return;
        }
        nccl_stream->cv.wait(l);
      }
      next_launch = nccl_stream->pending_launches_.back();
      nccl_stream->pending_launches_.pop_back();
    }
    Collective* collective = next_launch.first;
    tensorflow::profiler::TraceMeConsumer traceme("Run Collective",
                                                  collective->trace_context);
    ncclDataType_t data_type = ToNcclType(collective->data_type);
    int p_idx = next_launch.second;
    Participant* p = collective->participants[p_idx].get();
    auto nccl_comm = collective->communicator->members[p_idx].nccl_comm;
    ncclResult_t nccl_result = ncclSuccess;
    switch (collective->type) {
      case kAllReduce: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->output->tensor_data().data());
        VLOG(2) << "call NcclAllReduce collective_key "
                << collective->collective_key << " participant " << p_idx
                << " num_participants " << collective->participants.size()
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "ncclAllReduce",
              {{"buffer_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "all_reduce"}});
        });
        nccl_result = ncclAllReduce(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, collective->reduction_op,
                                    nccl_comm, cu_stream);
        break;
      }
      case kBroadcast: {
        const void* sendbuff = nullptr;
        void* recvbuff = nullptr;
        int num_elements = -1;
        if (p->input) {
          sendbuff = p->input->tensor_data().data();
          num_elements = p->input->NumElements();
        }
        if (p->output) {
          recvbuff = const_cast<char*>(p->output->tensor_data().data());
          num_elements = p->output->NumElements();
        } else {
          recvbuff = const_cast<void*>(sendbuff);
        }
        if (num_elements < 0) {
          p->done_callback(errors::Internal(
              "Both input and output are null in ncclBroadcast"));
          collective->Unref();
          continue;
        }
        VLOG(2) << "call NcclBroadcast collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "ncclBroadcast",
              {{"buffer_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "broadcast"}});
        });
        nccl_result =
            ncclBroadcast(sendbuff, recvbuff, num_elements, data_type,
                          collective->root_rank, nccl_comm, cu_stream);
        break;
      }
      case kReduce: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff =
            p->output ? const_cast<char*>(p->output->tensor_data().data())
                      : nullptr;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "buffer_size",
              {{"output_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "reduce"}});
        });
        nccl_result = ncclReduce(sendbuff, recvbuff, p->input->NumElements(),
                                 data_type, collective->reduction_op,
                                 collective->root_rank, nccl_comm, cu_stream);
        break;
      }
      case kAllGather: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->output->tensor_data().data());
        VLOG(2) << "call NcclAllGather collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " sendcount "
                << p->input->NumElements() << " recvbuff " << recvbuff
                << " recvcount " << p->output->NumElements() << " nccl_comm "
                << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "ncclAllGather",
              {{"buffer_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "all_gather"}});
        });
        nccl_result = ncclAllGather(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, nccl_comm, cu_stream);
        break;
      }
      case kReduceScatter: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->output->tensor_data().data());
        VLOG(2) << "call NcclReduceScatter collective_key "
                << collective->collective_key << " participant " << p_idx
                << " num_participants " << collective->participants.size()
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "ncclReduceScatter",
              {{"buffer_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "reduce_scatter"}});
        });
        nccl_result = ncclReduceScatter(
            sendbuff, recvbuff, p->output->NumElements(), data_type,
            collective->reduction_op, nccl_comm, cu_stream);
        break;
      }
      case kAllToAll: {
        const char* sendbuff = p->input->tensor_data().data();
        char* recvbuff = const_cast<char*>(p->output->tensor_data().data());
        size_t count =
            p->input->NumElements() / collective->participants.size();
        size_t rank_offset = count * DataTypeSize(collective->data_type);
        VLOG(2) << "call Nccl All to All collective_key "
                << collective->collective_key << " participant " << p_idx
                << " num_participants " << collective->participants.size()
                << " sendbuff " << static_cast<const void*>(sendbuff)
                << " recvbuff " << static_cast<void*>(recvbuff) << " nccl_comm "
                << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        profiler::AnnotatedTraceMe traceme([&] {
          return profiler::TraceMeEncode(
              "ncclAllToAll",
              {{"buffer_size", ComputeBufferSize(p, collective->data_type)},
               {"collective_type", "all_to_all"}});
        });
        ncclGroupStart();
        for (int i = 0; i < collective->participants.size(); ++i) {
          ncclSend(sendbuff + i * rank_offset, count, data_type,
                   collective->participants[i]->global_rank, nccl_comm,
                   cu_stream);
          ncclRecv(recvbuff + i * rank_offset, count, data_type,
                   collective->participants[i]->global_rank, nccl_comm,
                   cu_stream);
        }
        nccl_result = ncclGroupEnd();
        break;
      }
    }
    auto done_callback = [collective, p_idx, nccl_result]() {
      VLOG(2) << "done Nccl kernel collective_key "
              << collective->collective_key << " participant " << p_idx
              << " ncclResult " << nccl_result;
      if (nccl_result == ncclSuccess) {
        collective->participants[p_idx]->done_callback(OkStatus());
      } else {
        collective->participants[p_idx]->done_callback(errors::Unknown(
            "Error invoking NCCL: ", ncclGetErrorString(nccl_result)));
      }
      collective->Unref();
    };
    p->event_mgr->ThenExecute(comm_stream, done_callback);
  }
}
void NcclManager::StartAbort(const Status& s) {
  absl::flat_hash_map<string, Collective*> collectives;
  std::vector<std::unique_ptr<Communicator>> communicators;
  {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      LOG(WARNING)
          << "NcclManager already aborted, ignoring subsequent StartAbort with "
          << s;
      return;
    }
    status_ = s;
    collectives.swap(collectives_);
    communicators.swap(communicators_);
  }
  VLOG(2) << "Aborted NcclManager " << this << " with " << collectives.size()
          << " collectives and " << communicators.size()
          << " comms with status " << s;
  for (const auto& item : collectives) {
    for (const std::unique_ptr<Participant>& p : item.second->participants) {
      p->done_callback(s);
    }
    item.second->Unref();
  }
  UnboundedWorkQueue queue(Env::Default(), "nccl_abort");
  int num_comms = 0;
  for (std::unique_ptr<Communicator>& communicator : communicators) {
    num_comms += communicator->members.size();
  }
  BlockingCounter pending(num_comms);
  for (std::unique_ptr<Communicator>& communicator : communicators) {
    for (CommunicatorMember& member : communicator->members) {
      queue.Schedule([&member, &pending]() {
        ncclCommAbort(member.nccl_comm);
        member.nccl_comm = nullptr;
        pending.DecrementCount();
      });
    }
  }
  pending.Wait();
}
void NcclManager::Reset() {
  mutex_lock l(mu_);
  status_ = Status();
  VLOG(2) << "Reset NcclManager " << this;
}
}  
#endif  