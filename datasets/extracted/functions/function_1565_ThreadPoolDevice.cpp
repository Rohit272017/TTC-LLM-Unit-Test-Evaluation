#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL) && defined(_OPENMP)
#ifndef DNNL_AARCH64_USE_ACL
#include "external/llvm_openmp/include/omp.h"
#define EIGEN_DONT_PARALLELIZE
#else
#include "omp.h"  
#endif
#endif  
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/core/util/util.h"
#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"
#include "tensorflow/core/platform/cpu_info.h"
#endif  
namespace tensorflow {
ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, locality)),
      allocator_(allocator),
      scoped_allocator_mgr_(new ScopedAllocatorMgr(name)) {
  auto s = NodeFileWriter::GetNodeFileWriterIfEnabled(name, env());
  if (!s.ok()) {
    LOG(ERROR) << s.status();
  } else {
    node_file_writer_ = *s;
    if (node_file_writer_) {
      LOG(INFO) << "Writing NodeDefs to file: "
                << node_file_writer_->filename();
    }
  }
#if defined(ENABLE_ONEDNN_OPENMP) && defined(INTEL_MKL)
  if (!IsMKLEnabled()) return;
#ifdef _OPENMP
  const char* user_omp_threads = getenv("OMP_NUM_THREADS");
  static absl::once_flag num_threads_setting_flag;
  if (user_omp_threads == nullptr) {
    const int mkl_intra_op = port::NumSchedulableCPUs();
    const int ht = port::NumHyperthreadsPerCore();
    absl::call_once(num_threads_setting_flag, omp_set_num_threads,
                    (mkl_intra_op + ht - 1) / ht);
  }
#ifndef DNNL_AARCH64_USE_ACL
  const char* user_kmp_blocktime = getenv("KMP_BLOCKTIME");
  static absl::once_flag blocktime_setting_flag;
  if (user_kmp_blocktime == nullptr) {
    absl::call_once(blocktime_setting_flag, kmp_set_blocktime, 1);
  }
#endif
#endif  
#endif  
}
ThreadPoolDevice::~ThreadPoolDevice() {}
Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}
Allocator* ThreadPoolDevice::GetScopedAllocator(AllocatorAttributes attr,
                                                int64_t step_id) {
  if (attr.scope_id > 0) {
    return scoped_allocator_mgr_->GetContainer(step_id)->GetInstance(
        attr.scope_id);
  }
  LOG(FATAL) << "Unexpected call to ThreadPoolDevice::GetScopedAllocator "
             << "attr.scope_id = " << attr.scope_id;
  return allocator_;
}
Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(allocator_, tensor_proto)) {
      *tensor = std::move(parsed);
      return absl::OkStatus();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 tensor_proto.DebugString());
}
void ThreadPoolDevice::CopyTensorInSameDevice(
    const Tensor* input_tensor, Tensor* output_tensor,
    const DeviceContext* device_context, StatusCallback done) {
  if (input_tensor->NumElements() != output_tensor->NumElements()) {
    done(errors::Internal(
        "CPU->CPU copy shape mismatch: input=", input_tensor->shape(),
        ", output=", output_tensor->shape()));
    return;
  }
  tensor::DeepCopy(*input_tensor, output_tensor);
  done(absl::OkStatus());
}
namespace {
const absl::flat_hash_set<std::string>* GetOpsToLogFromEnv() {
  auto* result = new absl::flat_hash_set<std::string>;
  const char* env = getenv("TF_CPU_DEBUG_OPS_TO_LOG");
  if (!env) {
    return result;
  }
  std::vector<absl::string_view> ops = absl::StrSplit(env, ',');
  LOG(INFO) << "Will log inputs & outputs from the following ops: ";
  for (absl::string_view op : ops) {
    result->insert(std::string(op));
    LOG(INFO) << "  |" << op << "|";
  }
  return result;
}
bool ShouldLogInputsAndOutputs(OpKernel* op_kernel) {
  static const absl::flat_hash_set<std::string>& ops_to_log =
      *GetOpsToLogFromEnv();
  static const bool is_empty = ops_to_log.empty();
  if (is_empty) {
    return false;
  }
  return ops_to_log.count(op_kernel->type_string());
}
}  
void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);
  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
  }
  op_kernel->Compute(context);
  if (context->status().ok() && node_file_writer_) {
    Status s = node_file_writer_->RecordNodeExecution(op_kernel, context);
    if (!s.ok()) {
      LOG(ERROR) << s;
      context->SetStatus(s);
    }
  }
  if (should_log_inputs_and_outputs) {
    LogOutputs(op_kernel, context);
  }
}
void ThreadPoolDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                    OpKernelContext* context,
                                    AsyncOpKernel::DoneCallback done) {
  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);
  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
    AsyncOpKernel::DoneCallback parent_done = done;
    done = [this, parent_done, op_kernel, context]() {
      LogOutputs(op_kernel, context);
      parent_done();
    };
  }
  op_kernel->ComputeAsync(context, done);
}
void ThreadPoolDevice::LogInputs(OpKernel* op_kernel,
                                 OpKernelContext* context) {
  LOG(INFO) << "Inputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_inputs(); i++) {
    if (!context->has_input(i)) {
      LOG(INFO) << "input # " << i << " is absent";
      continue;
    }
    LOG(INFO) << "input # " << i;
    LOG(INFO) << context->input(i).DebugString(-1);
  }
  LOG(INFO) << "";
}
void ThreadPoolDevice::LogOutputs(OpKernel* op_kernel,
                                  OpKernelContext* context) {
  if (!context->status().ok()) {
    LOG(INFO) << op_kernel->name()
              << " failed: " << context->status().message();
    return;
  }
  LOG(INFO) << "Outputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_outputs(); i++) {
    Tensor* output = context->mutable_output(i);
    if (output == nullptr) {
      LOG(INFO) << "output # " << i << " is null";
    } else {
      LOG(INFO) << "output # " << i;
      LOG(INFO) << output->DebugString(-1);
    }
  }
  LOG(INFO) << "";
}
#ifdef INTEL_MKL
namespace {
class MklCPUAllocatorFactory : public AllocatorFactory {
 public:
  bool NumaEnabled() override { return false; }
  Allocator* CreateAllocator() override { return new MklCPUAllocator; }
  virtual SubAllocator* CreateSubAllocator(int numa_node) {
    return new MklSubAllocator;
  }
};
REGISTER_MEM_ALLOCATOR("MklCPUAllocator",
                       ((IsMKLEnabled() || IsZenDnnEnabled()) ? 200 : 50),
                       MklCPUAllocatorFactory);
}  
#endif  
}  