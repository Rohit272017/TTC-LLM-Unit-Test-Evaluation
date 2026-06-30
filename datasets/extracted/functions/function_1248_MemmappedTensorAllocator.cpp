#include "tensorflow/core/kernels/immutable_constant_op.h"
#include <unordered_set>
#include "tensorflow/core/framework/types.pb.h"
namespace tensorflow {
namespace {
class MemmappedTensorAllocator : public Allocator {
 public:
  MemmappedTensorAllocator() {}
  Status InitializeFromRegion(const string& name, Env* env) {
    const auto status =
        env->NewReadOnlyMemoryRegionFromFile(name, &memory_region_);
    if (!status.ok()) {
      return status;
    }
    return absl::OkStatus();
  }
  string Name() override { return "MemmappedTensorAllocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if ((reinterpret_cast<intptr_t>(memory_region_->data())) % alignment != 0) {
      allocation_status_ =
          errors::Internal("Readonly memory region has wrong alignment");
      return nullptr;
    }
    if (num_bytes > memory_region_->length()) {
      allocation_status_ = errors::Internal(
          "Readonly memory region has wrong length (", memory_region_->length(),
          ") when allocating ", num_bytes);
      return nullptr;
    }
    return const_cast<void*>(memory_region_->data());
  }
  void DeallocateRaw(void* ptr) override {
    if (ptr != memory_region_->data()) {
      LOG(ERROR)
          << "Deallocating not allocated region for readonly memory region";
    }
    if (delete_on_deallocate_) {
      delete this;
    }
  }
  const Status& allocation_status() const { return allocation_status_; }
  void set_delete_on_deallocate() { delete_on_deallocate_ = true; }
  bool AllocatesOpaqueHandle() const override { return true; }
 private:
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region_;
  Status allocation_status_;
  bool delete_on_deallocate_ = false;
  MemmappedTensorAllocator(const MemmappedTensorAllocator&) = delete;
  void operator=(const MemmappedTensorAllocator&) = delete;
};
}  
ImmutableConstantOp::ImmutableConstantOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context,
                 context->GetAttr(kMemoryRegionNameAttr, &region_name_));
  OP_REQUIRES_OK(context, context->GetAttr(kDTypeAttr, &dtype_));
  OP_REQUIRES(context, dtype_ != DT_RESOURCE && dtype_ != DT_VARIANT,
              errors::InvalidArgument(
                  "Resource and variant dtypes are invalid for this op."));
  OP_REQUIRES_OK(context, context->GetAttr(kShapeAttr, &shape_));
}
void ImmutableConstantOp::Compute(OpKernelContext* ctx) {
  std::unique_ptr<MemmappedTensorAllocator> allocator(
      new MemmappedTensorAllocator());
  OP_REQUIRES_OK(ctx,
                 allocator->InitializeFromRegion(region_name_, ctx->env()));
  OP_REQUIRES(ctx, dtype_ != DT_STRING,
              errors::Unimplemented("Sorry, DT_STRING is not currently "
                                    "supported for ImmutableConstOp."));
  ctx->set_output(0, Tensor(allocator.get(), dtype_, shape_));
  OP_REQUIRES_OK(ctx, allocator->allocation_status());
  allocator.release()->set_delete_on_deallocate();
}
ImmutableConstantOp::~ImmutableConstantOp() {}
constexpr char const* ImmutableConstantOp::kDTypeAttr;
constexpr char const* ImmutableConstantOp::kShapeAttr;
constexpr char const* ImmutableConstantOp::kMemoryRegionNameAttr;
REGISTER_KERNEL_BUILDER(Name("ImmutableConst").Device(DEVICE_CPU),
                        ImmutableConstantOp);
}  