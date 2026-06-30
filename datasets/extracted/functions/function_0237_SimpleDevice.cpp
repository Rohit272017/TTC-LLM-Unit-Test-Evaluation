#define EIGEN_USE_THREADS
#include "tensorflow/core/transforms/utils/eval_utils.h"
#include <cassert>
#include <utility>
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/public/version.h"
namespace mlir {
namespace tfg {
namespace util {
static constexpr int kThreads = 2;
SimpleDevice::SimpleDevice() : DeviceBase(tensorflow::Env::Default()) {
  eigen_worker_ = std::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "eval_utils", kThreads);
  eigen_worker_threads_.num_threads = kThreads;
  eigen_worker_threads_.workers = eigen_worker_.get();
  eigen_device_ = std::make_unique<Eigen::ThreadPoolDevice>(
      eigen_worker_threads_.workers->AsEigenThreadPool(),
      eigen_worker_threads_.num_threads);
  set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
  set_eigen_cpu_device(eigen_device_.get());
}
SimpleDevice::~SimpleDevice() {}
tensorflow::Allocator *SimpleDevice::GetAllocator(
    tensorflow::AllocatorAttributes attr) {
  return tensorflow::cpu_allocator();
}
tensorflow::Status SimpleDevice::MakeTensorFromProto(
    const tensorflow::TensorProto &tensor_proto,
    const tensorflow::AllocatorAttributes alloc_attrs,
    tensorflow::Tensor *tensor) {
  tensorflow::Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(tensorflow::cpu_allocator(), tensor_proto)) {
    return tensorflow::errors::InvalidArgument(
        "Cannot parse tensor from tensor_proto.");
  }
  *tensor = std::move(parsed);
  return absl::OkStatus();
}
LogicalResult EvaluateOperation(tensorflow::DeviceBase *cpu_device,
                                tensorflow::ResourceMgr *resource_mgr, TFOp op,
                                ArrayRef<ElementsAttr> operands,
                                SmallVectorImpl<TypedAttr> &results) {
  assert(cpu_device && "cpu device can't be null");
  assert(resource_mgr && "ResourceMgr can't be null");
  if (llvm::any_of(operands, [](Attribute operand) { return !operand; })) {
    VLOG(3) << "cannot be evaluated with null operands";
    return failure();
  }
  tensorflow::NodeDef node_def;
  if (!ConvertToNodeDef(&*op, &node_def, op.getDialect(), [&](Value value) {
         return GetValueName(value, op.getDialect());
       }).ok()) {
    VLOG(3) << "failed to convert operation to NodeDef";
    return failure();
  }
  absl::InlinedVector<tensorflow::Tensor, 4> input_tensors(operands.size());
  absl::InlinedVector<tensorflow::TensorValue, 4> input_tensor_values(
      operands.size());
  for (auto it : llvm::zip(operands, input_tensors, input_tensor_values)) {
    auto &[operand, input_tensor, input_tensor_value] = it;
    if (!ConvertToTensor(operand, &input_tensor).ok()) return failure();
    input_tensor_value.tensor = &input_tensor;
  }
  tensorflow::Status status;
  std::unique_ptr<tensorflow::OpKernel> op_kernel = tensorflow::CreateOpKernel(
      tensorflow::DEVICE_CPU, cpu_device, cpu_device->GetAllocator({}),
      node_def, TF_GRAPH_DEF_VERSION, &status);
  if (!status.ok()) {
    VLOG(3) << status.message();
    return failure();
  }
  tensorflow::OpKernelContext::Params params;
  params.device = cpu_device;
  params.frame_iter = tensorflow::FrameAndIter(0, 0);
  params.inputs = input_tensor_values;
  params.op_kernel = op_kernel.get();
  params.resource_manager = resource_mgr;
  absl::InlinedVector<tensorflow::AllocatorAttributes, 4> output_attrs(
      op_kernel->num_outputs());
  for (auto &attr : output_attrs) attr.set_on_host(true);
  params.output_attr_array = output_attrs.data();
  tensorflow::OpKernelContext op_context(&params);
  op_kernel->Compute(&op_context);
  if (!op_context.status().ok()) {
    VLOG(3) << op_context.status().message();
    return failure();
  }
  Builder builder(op->getContext());
  for (int i = 0; i < op_kernel->num_outputs(); ++i) {
    if (op_context.mutable_output(i) == nullptr) {
      results.push_back(nullptr);
      continue;
    }
    absl::StatusOr<ElementsAttr> attr_or =
        ConvertTensor(*(op_context.mutable_output(i)), builder);
    if (!attr_or.status().ok()) {
      VLOG(3) << attr_or.status().message();
      return failure();
    }
    results.push_back(attr_or.value());
  }
  return success();
}
}  
}  
}  