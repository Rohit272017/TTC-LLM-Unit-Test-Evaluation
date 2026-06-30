#include "xla/service/cpu/ir_emitter2.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/cpu/shape_partition.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla::cpu {
namespace {
static llvm::StructType* Dim3StructTy(llvm::LLVMContext& ctx,
                                      std::string_view name) {
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}
static llvm::StructType* KernelThreadDimTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThreadDim");
}
static llvm::StructType* KernelThreadTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThread");
}
static llvm::StructType* KernelArgTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelArg", ptr, i64);
}
static llvm::StructType* KernelCallFrameTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}
static llvm::FunctionType* KernelFunctionTy(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 false);
}
}  
class IrEmitter2::ElementalIrEmitter : public xla::ElementalIrEmitter {
 public:
  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilder<>* b,
                     const HloModule* hlo_module, IrEmitter* nested_ir_emitter,
                     bool fast_min_max)
      : xla::ElementalIrEmitter(
            module, b,
            Options{true}),
        hlo_module_(hlo_module),
        nested_ir_emitter_(nested_ir_emitter),
        fast_min_max_(fast_min_max) {}
 protected:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override {
    return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
  }
  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override {
    return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
  }
  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override {
    return xla::cpu::EmitErf(module(), *b(), prim_type, value);
  }
  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    if (!hlo_module_ || !hlo_module_->has_schedule()) {
      return absl::InternalError(
          "HLO module must be scheduled to emit thread local computation.");
    }
    auto emit_computation = [&](const HloComputation* computation) {
      if (!nested_ir_emitter_->is_computation_emitted(*computation,
                                                      is_reducer)) {
        VLOG(2) << "Emit nested computation: " << computation->name();
        TF_RETURN_IF_ERROR(
            nested_ir_emitter_
                ->EmitComputation(
                    const_cast<HloComputation*>(computation), name, false,
                    hlo_module_->schedule()
                        .sequence(computation)
                        .instructions(),
                    is_reducer,
                    {llvm::Attribute::AlwaysInline})
                .status());
      }
      return absl::OkStatus();
    };
    for (HloComputation* embedded : callee.MakeEmbeddedComputationsList()) {
      if (embedded->IsFusionComputation()) continue;
      TF_RETURN_IF_ERROR(emit_computation(embedded));
    }
    TF_RETURN_IF_ERROR(emit_computation(&callee));
    VLOG(2) << "Emit thread local call to: " << callee.name();
    nested_ir_emitter_->b()->SetInsertPoint(b()->GetInsertPoint());
    auto values = nested_ir_emitter_->EmitThreadLocalCall(
        callee, parameters, name, is_reducer, false);
    return values;
  }
  bool fast_min_max() override { return fast_min_max_; }
 private:
  const HloModule* hlo_module_;
  IrEmitter* nested_ir_emitter_;
  bool fast_min_max_;
};
IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      call_frame_ty_(KernelCallFrameTy(module_->getContext())),
      thread_dims_ty_(KernelThreadDimTy(module_->getContext())),
      thread_ty_(KernelThreadTy(module_->getContext())),
      arg_ty_(KernelArgTy(module_->getContext())) {}
bool IrEmitter2::fast_min_max() const {
  return hlo_module_.config().debug_options().xla_cpu_enable_fast_min_max();
}
IrEmitter2::KernelInfo::KernelInfo(KernelPrototype prototype,
                                   const se::BlockDim& block_dims,
                                   const se::ThreadDim& thread_dims)
    : name(prototype.function->getName().str()),
      block_dims(block_dims),
      thread_dims(thread_dims),
      invariant_arguments(std::move(prototype.invariant_arguments)) {}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitElementalHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit elemental host kernel: " << instr->name();
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));
  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    operand_to_generator[operand] = [&, i](const llvm_ir::IrArray::Index& idx) {
      return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
    };
  }
  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  llvm_ir::ElementGenerator element_generator =
      elemental_emitter.MakeElementGenerator(instr, operand_to_generator);
  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, instr, kernel_prototype, element_generator));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), thread_dims));
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitPadHostKernel(
    const HloInstruction* pad) {
  VLOG(2) << "Emit Pad host kernel.";
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(pad));
  llvm_ir::IrArray operand_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray padvalue_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);
  auto builder_overwrite = nested_ir_emitter_->WithBuilder(b);
  nested_ir_emitter_->PushComputeFunction(
      &b, module_,
      0, kernel_prototype.function,
      nullptr, kernel_prototype.return_block);
  TF_RETURN_IF_ERROR(nested_ir_emitter_->HandlePad(
      const_cast<HloInstruction*>(pad), operand_array, padvalue_array,
      output_array));
  nested_ir_emitter_->PopComputeFunction();
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit fusion host kernel: " << fusion->name();
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kOutput) {
    return EmitDotFusionHostKernel(fusion);
  }
  if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return Internal("Unsupported loop fusion kind for instruction: %s",
                    fusion->ToString());
  }
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(fusion));
  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());
  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (int i = 0; i < fusion->operand_count(); i++) {
    fused_emitter.BindGenerator(
        *fusion->fused_parameter(i), [&, i](llvm_ir::IrArray::Index idx) {
          return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
        });
  }
  if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(
          const_cast<HloFusionInstruction*>(fusion),
          nested_ir_emitter_->assignment())) {
    TF_RETURN_IF_ERROR(llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
        const_cast<HloFusionInstruction*>(fusion), kernel_prototype.results[0],
        &fused_emitter, &b));
    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion->fused_expression_root()));
  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, fusion, kernel_prototype, element_generator));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), thread_dims));
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitReductionHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit reduction host kernel: " << instr->name();
  return EmitElementalHostKernel(instr);
}
static bool IsDotCodegenStrategy(DotImplementationStrategy strategy) {
  static std::array<DotImplementationStrategy, 3> kDotCodegenStrategies = {
      DotImplementationStrategy::kNaiveLlvmIr,
      DotImplementationStrategy::kTiledLlvmIrGemm,
      DotImplementationStrategy::kTiledLlvmIrGemv,
  };
  return absl::c_find(kDotCodegenStrategies, strategy) !=
         kDotCodegenStrategies.end();
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit dot host kernel: " << instr->name();
  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *instr,
      nested_ir_emitter_->target_machine_features());
  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));
  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());
  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];
  TF_RETURN_IF_ERROR(EmitDotOperation(
      *instr, target_array, lhs_array, rhs_array,
      nullptr, nullptr, &b,
      hlo_module_.config(), nested_ir_emitter_->target_machine_features(),
      false));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitConcatenateHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit concatenate host kernel: " << instr->name();
  auto fast_impl_reason = CanDoFastConcatenate(instr);
  if (fast_impl_reason.ok()) {
    VLOG(1) << "Emitting fast concatenate for " << instr->ToString() << ": "
            << fast_impl_reason.message();
    TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                        EmitKernelPrototype(instr));
    llvm::IRBuilder<> ir_builder(module_->getContext());
    ir_builder.SetInsertPoint(
        kernel_prototype.function->getEntryBlock().getTerminator());
    llvm_ir::IrArray output_array = kernel_prototype.results[0];
    TF_RETURN_IF_ERROR(::xla::cpu::EmitFastConcatenate(
        instr, kernel_prototype.arguments, output_array, module_, ir_builder));
    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }
  VLOG(1) << "Could not emit fast concatenate for " << instr->ToString() << ": "
          << fast_impl_reason.message();
  return EmitElementalHostKernel(instr);
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit dot fusion host kernel: " << fusion->name();
  const HloInstruction* add = fusion->fused_expression_root();
  if (add->opcode() != HloOpcode::kAdd) {
    return Internal("Dot fusion supports only `add` root instruction");
  }
  bool is_dot_operand0 = add->operand(0)->opcode() == HloOpcode::kDot;
  bool is_dot_operand1 = add->operand(1)->opcode() == HloOpcode::kDot;
  if (is_dot_operand0 == is_dot_operand1) {
    return Internal("Dot fusion root instruction must have single dot operand");
  }
  int64_t dot_op_index = is_dot_operand0 ? 0 : 1;
  int64_t addend_op_index = 1 - dot_op_index;
  const HloInstruction* dot = add->operand(dot_op_index);
  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *dot,
      nested_ir_emitter_->target_machine_features());
  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }
  int64_t dot_lhs_pnum = dot->operand(0)->parameter_number();
  int64_t dot_rhs_pnum = dot->operand(1)->parameter_number();
  int64_t addend_pnum = add->operand(addend_op_index)->parameter_number();
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(fusion));
  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());
  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[dot_lhs_pnum];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[dot_rhs_pnum];
  llvm_ir::IrArray addend_array = kernel_prototype.arguments[addend_pnum];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];
  TF_RETURN_IF_ERROR(EmitDotOperation(
      *dot, target_array, lhs_array, rhs_array, &addend_array,
      nullptr, &b, hlo_module_.config(),
      nested_ir_emitter_->target_machine_features(),
      false));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}
absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitSliceToDynamicHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit slice-to-dynamic host kernel: " << instr->name();
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));
  llvm::IRBuilder<> ir_builder(module_->getContext());
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());
  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  auto guard = nested_ir_emitter_->WithBuilder(ir_builder);
  TF_RETURN_IF_ERROR(nested_ir_emitter_->EmitSliceToDynamic(
      instr, kernel_prototype.arguments, output_array));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}
absl::StatusOr<IrEmitter2::KernelInfo>
IrEmitter2::EmitSelectAndScatterHostKernel(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                      EmitKernelPrototype(instr));
  llvm_ir::IrArray operand_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray source_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  TF_RETURN_IF_ERROR(nested_ir_emitter_->HandleSelectAndScatter(
      const_cast<HloInstruction*>(instr), operand_array, source_array,
      output_array));
  return kernels_.emplace_back(
      KernelInfo(std::move(kernel_prototype), se::BlockDim(), se::ThreadDim()));
}
absl::StatusOr<IrEmitter2::KernelInfo>
IrEmitter2::EmitDynamicUpdateSliceHostKernel(const HloInstruction* instr) {
  if (llvm_ir::CanUpdateDynamicSliceInPlace(const_cast<HloInstruction*>(instr),
                                            nested_ir_emitter_->assignment())) {
    VLOG(2) << "Emit in-place dynamic-update-slice kernel: " << instr->name();
    TF_ASSIGN_OR_RETURN(KernelPrototype kernel_prototype,
                        EmitKernelPrototype(instr));
    llvm::IRBuilder<> b(module_->getContext());
    b.SetInsertPoint(
        kernel_prototype.function->getEntryBlock().getTerminator());
    TF_RETURN_IF_ERROR(llvm_ir::EmitDynamicUpdateSliceInPlace(
        kernel_prototype.arguments, kernel_prototype.results.front(),
        llvm_ir::IrName(instr, "in_place"), &b));
    return kernels_.emplace_back(KernelInfo(std::move(kernel_prototype),
                                            se::BlockDim(), se::ThreadDim()));
  }
  return EmitElementalHostKernel(instr);
}
absl::StatusOr<IrEmitter2::ComparatorInfo> IrEmitter2::EmitSortComparator(
    const HloInstruction* instr) {
  HloComputation* comparator = instr->to_apply();
  auto info = absl::c_find_if(comparators_, [&](const ComparatorInfo& info) {
    return info.name == comparator->name();
  });
  if (info != comparators_.end()) return *info;
  auto schedule = comparator->MakeInstructionPostOrder();
  TF_ASSIGN_OR_RETURN(llvm::Function * comparator_function,
                      nested_ir_emitter_->EmitComputation(
                          comparator, comparator->name(),
                          true, schedule,
                          false));
  comparator_function->setUWTableKind(llvm::UWTableKind::Default);
  return comparators_.emplace_back(
      ComparatorInfo{comparator_function->getName().str()});
}
absl::StatusOr<BufferAllocation::Slice> IrEmitter2::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return nested_ir_emitter_->assignment().GetUniqueSlice(instruction, index);
}
absl::StatusOr<std::vector<IrEmitter2::KernelParameter>>
IrEmitter2::GetKernelArgumentsParameters(const HloInstruction* instruction) {
  std::vector<KernelParameter> arguments;
  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          GetAllocationSlice(operand, indexed.index));
      arguments.push_back(KernelParameter{indexed.shape, slice});
    }
  }
  return arguments;
}
absl::StatusOr<std::vector<IrEmitter2::KernelParameter>>
IrEmitter2::GetKernelResultsParameters(const HloInstruction* instruction) {
  std::vector<KernelParameter> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        GetAllocationSlice(instruction, indexed.index));
    results.push_back(KernelParameter{indexed.shape, slice});
  }
  return results;
}
absl::Status IrEmitter2::VerifyKernelParameters(
    absl::Span<const KernelParameter> arguments,
    absl::Span<const KernelParameter> results) {
  for (size_t i = 0; i < arguments.size(); ++i) {
    for (size_t j = i + 1; j < arguments.size(); ++j) {
      const KernelParameter& a = arguments[i];
      const KernelParameter& b = arguments[j];
      if (a.slice != b.slice && a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel arguments must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = i + 1; j < results.size(); ++j) {
      const KernelParameter& a = results[i];
      const KernelParameter& b = results[j];
      if (a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel results must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < arguments.size(); ++j) {
      const KernelParameter& result = results[i];
      const KernelParameter& argument = arguments[j];
      if (result.slice.OverlapsWith(argument.slice) &&
          result.slice != argument.slice) {
        return Internal(
            "Kernel results must not partially overlap with arguments: result "
            "#%d (%s) overlaps with argument #%d (%s)",
            i, result.slice.ToString(), j, argument.slice.ToString());
        break;
      }
    }
  }
  return absl::OkStatus();
}
IrEmitter2::KernelThreadDims IrEmitter2::EmitKernelThreadDims(
    llvm::IRBuilder<>& b, llvm::Value* call_frame) {
  auto* td_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 0, "tdims_gep");
  auto* tdims = b.CreateLoad(b.getPtrTy(), td_gep, "tdims");
  auto* x_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 0, "tdim_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 1, "tdim_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 2, "tdim_z_gep");
  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tdim_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tdim_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tdim_z")};
}
IrEmitter2::KernelThread IrEmitter2::EmitKernelThread(llvm::IRBuilder<>& b,
                                                      llvm::Value* call_frame) {
  auto* t_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 1, "tid_gep");
  auto* tids = b.CreateLoad(b.getPtrTy(), t_gep, "tids");
  auto* x_gep = b.CreateStructGEP(thread_ty_, tids, 0, "tid_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_ty_, tids, 1, "tid_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_ty_, tids, 2, "tid_z_gep");
  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tid_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tid_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tid_z")};
}
llvm_ir::IrArray IrEmitter2::EmitKernelArgument(llvm::IRBuilder<>& b,
                                                llvm::Value* call_frame,
                                                int64_t index,
                                                const Shape& shape) {
  llvm::Type* ptr = llvm::PointerType::get(b.getContext(), 0);
  std::string name = absl::StrCat("arg", index);
  auto* args_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 3, "args_gep");
  auto* args = b.CreateLoad(ptr, args_gep, "args");
  auto* data_gep = b.CreateConstGEP2_32(arg_ty_, args, index, 0, name + "_gep");
  auto* data = b.CreateLoad(ptr, data_gep, name);
  llvm_ir::SetAlignmentMetadataForLoad(data, cpu_function_runtime::MinAlign());
  IrEmitter::AttachDereferenceableMetadataForLoad(data, ByteSizeOf(shape));
  AttachInvariantLoadMetadataForLoad(data);
  return llvm_ir::IrArray(data, llvm_ir::ShapeToIrType(shape, module_), shape);
}
absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    std::string_view name, absl::Span<const KernelParameter> arguments,
    absl::Span<const KernelParameter> results) {
  VLOG(3) << "Emit kernel prototype: " << name
          << ", #arguments=" << arguments.size()
          << ", #results=" << results.size();
  for (const KernelParameter& argument : arguments) {
    VLOG(3) << "  argument: " << argument.shape.ToString(true) << " in "
            << argument.slice.ToString();
  }
  for (const KernelParameter& result : results) {
    VLOG(3) << "  result: " << result.shape.ToString(true) << " in "
            << result.slice.ToString();
  }
  TF_RETURN_IF_ERROR(VerifyKernelParameters(arguments, results));
  llvm::LLVMContext& ctx = module_->getContext();
  llvm::MDBuilder mb(ctx);
  llvm::IRBuilder<> b(ctx);
  llvm::MDNode* domain = mb.createAliasScopeDomain(
      absl::StrFormat("XLA host kernel %s AA domain", name));
  absl::btree_map<BufferAllocation::Slice, llvm::MDNode*> alias_scopes;
  for (const KernelParameter& result : results) {
    if (result.slice.allocation()->is_parameter_aliased_with_output()) {
      continue;
    }
    alias_scopes[result.slice] = mb.createAliasScope(
        absl::StrFormat("result slice: %s", result.slice.ToString()), domain);
  }
  auto get_alias_scope = [&](BufferAllocation::Slice slice) -> llvm::MDNode* {
    auto it = alias_scopes.find(slice);
    return it == alias_scopes.end() ? nullptr
                                    : llvm::MDNode::get(ctx, it->second);
  };
  auto get_noalias = [&](BufferAllocation::Slice slice) -> llvm::MDNode* {
    llvm::SmallVector<llvm::Metadata*> scopes;
    for (const auto& [alias_slice, alias_scope] : alias_scopes) {
      if (!slice.OverlapsWith(alias_slice)) {
        scopes.push_back(alias_scope);
      }
    }
    return scopes.empty() ? nullptr : llvm::MDNode::get(ctx, scopes);
  };
  absl::flat_hash_set<BufferAllocation::Slice> result_slices;
  result_slices.reserve(results.size());
  for (const KernelParameter& result : results) {
    result_slices.insert(result.slice);
  }
  llvm::Function* function = llvm::Function::Create(
      KernelFunctionTy(ctx), llvm::GlobalValue::ExternalLinkage, name, module_);
  function->setCallingConv(llvm::CallingConv::C);
  function->setUWTableKind(llvm::UWTableKind::Default);
  const DebugOptions& debug_options = hlo_module_.config().debug_options();
  function->addFnAttr(
      "prefer-vector-width",
      absl::StrCat(debug_options.xla_cpu_prefer_vector_width()));
  function->addFnAttr("frame-pointer", "all");
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));
  llvm::Value* call_frame = function->getArg(0);
  KernelThreadDims kernel_thread_dims = EmitKernelThreadDims(b, call_frame);
  KernelThread kernel_thread = EmitKernelThread(b, call_frame);
  int64_t idx = 0;
  absl::flat_hash_set<int64_t> invariant_arguments;
  std::vector<llvm_ir::IrArray> ir_arguments;
  for (int64_t i = 0; i < arguments.size(); ++i) {
    const KernelParameter& argument = arguments[i];
    auto ir_argument = EmitKernelArgument(b, call_frame, idx++, argument.shape);
    if (auto* noalias = get_noalias(argument.slice)) {
      ir_argument.AddNoaliasMetadata(noalias);
    }
    if (!result_slices.contains(argument.slice)) {
      ir_argument.MarkInvariantOverWholeProgram(&ctx);
      invariant_arguments.insert(i);
    }
    ir_arguments.push_back(std::move(ir_argument));
  }
  std::vector<llvm_ir::IrArray> ir_results;
  for (const KernelParameter& result : results) {
    auto ir_result = EmitKernelArgument(b, call_frame, idx++, result.shape);
    if (auto* noalias = get_noalias(result.slice)) {
      ir_result.AddNoaliasMetadata(noalias);
    }
    if (auto* alias_scope = get_alias_scope(result.slice)) {
      ir_result.AddAliasScopeMetadata(alias_scope);
    }
    ir_results.push_back(std::move(ir_result));
  }
  llvm::BasicBlock* return_block =
      llvm::BasicBlock::Create(ctx, "return", function);
  b.CreateBr(return_block);
  b.SetInsertPoint(return_block);
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));
  return KernelPrototype{function,
                         return_block,
                         kernel_thread_dims,
                         kernel_thread,
                         std::move(ir_arguments),
                         std::move(ir_results),
                         std::move(invariant_arguments)};
}
absl::StatusOr<IrEmitter2::KernelPrototype> IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> arguments,
                      GetKernelArgumentsParameters(instr));
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> results,
                      GetKernelResultsParameters(instr));
  return EmitKernelPrototype(instr->name(), std::move(arguments),
                             std::move(results));
}
std::optional<IrEmitter2::ParallelConfig> IrEmitter2::GetParallelConfig(
    const HloInstruction* instr) {
  auto backend_config = instr->backend_config<BackendConfig>();
  if (!backend_config.ok() ||
      backend_config->outer_dimension_partitions().empty()) {
    return std::nullopt;
  }
  ParallelConfig config;
  config.outer_dimension_partitions.assign(
      backend_config->outer_dimension_partitions().begin(),
      backend_config->outer_dimension_partitions().end());
  return config;
}
absl::Status IrEmitter2::CanDoFastConcatenate(
    const HloInstruction* concatenate) const {
  if (!concatenate->parent()
           ->root_instruction()
           ->template backend_config<BackendConfig>()
           ->outer_dimension_partitions()
           .empty()) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        "Cannot generate memcpy-based concat for the parallel CPU backend");
  }
  const Shape& output_shape = concatenate->shape();
  for (auto* op : concatenate->operands()) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Operand has mismatching layouts");
    }
  }
  return absl::OkStatus();
};
IrEmitter2::ParallelPartitionBounds IrEmitter2::EmitParallelPartitionBounds(
    llvm::IRBuilder<>& b, const KernelPrototype& kernel_prototype,
    const ParallelConfig& parallel_config, const Shape& shape,
    std::string_view name) {
  ShapePartitionIterator it(shape, parallel_config.outer_dimension_partitions);
  size_t num_parallel_dimensions =
      parallel_config.outer_dimension_partitions.size();
  llvm::ArrayType* dim_bounds_ty = llvm::ArrayType::get(b.getInt64Ty(), 2);
  llvm::ArrayType* partition_bounds_ty =
      llvm::ArrayType::get(dim_bounds_ty, num_parallel_dimensions);
  llvm::ArrayType* parallel_bounds_ty =
      llvm::ArrayType::get(partition_bounds_ty, it.GetTotalPartitionCount());
  std::vector<llvm::Constant*> partition_bounds;
  for (int64_t i = 0; i < it.GetTotalPartitionCount(); ++i) {
    std::vector<llvm::Constant*> dim_counts;
    for (auto [lower, size] : it.GetPartition(i)) {
      dim_counts.push_back(llvm::ConstantArray::get(
          dim_bounds_ty, {b.getInt64(lower), b.getInt64(lower + size)}));
    }
    partition_bounds.push_back(
        llvm::ConstantArray::get(partition_bounds_ty, dim_counts));
  }
  llvm::Constant* parallel_bounds =
      llvm::ConstantArray::get(parallel_bounds_ty, partition_bounds);
  llvm::Module* module = b.GetInsertBlock()->getParent()->getParent();
  llvm::GlobalVariable* parallel_bounds_global = new llvm::GlobalVariable(
      *module,
      parallel_bounds_ty,
      true,
      llvm::GlobalValue::PrivateLinkage,
      parallel_bounds,
      absl::StrCat(name, "_parallel_bounds"));
  ParallelPartitionBounds bounds;
  for (size_t i = 0; i < num_parallel_dimensions; ++i) {
    llvm::Value* partition = kernel_prototype.thread.x;
    llvm::Value* parallel_dim = b.getInt32(i);
    llvm::Value* lower_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(0)},
        absl::StrCat("lo_dim_", i, "_gep"));
    llvm::Value* upper_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(1)},
        absl::StrCat("up_dim_", i, "_gep"));
    bounds.emplace_back(
        b.CreateLoad(b.getInt64Ty(), lower_gep, absl::StrCat("lo_dim_", i)),
        b.CreateLoad(b.getInt64Ty(), upper_gep, absl::StrCat("up_dim_", i)));
  }
  return bounds;
}
absl::StatusOr<se::ThreadDim> IrEmitter2::EmitElementalLoops(
    llvm::IRBuilder<>& b, const HloInstruction* instr,
    const KernelPrototype& kernel_prototype,
    const llvm_ir::ElementGenerator& element_generator) {
  bool multiple_results = kernel_prototype.results.size() > 1;
  bool support_multiple_results = instr->opcode() == HloOpcode::kFusion ||
                                  instr->opcode() == HloOpcode::kReduce ||
                                  instr->opcode() == HloOpcode::kReduceWindow;
  auto parallel_config = GetParallelConfig(instr);
  bool has_parallel_config = parallel_config.has_value();
  if (multiple_results && !support_multiple_results) {
    return Internal(
        "Multi-output host kernels are not supported for %s instruction",
        HloOpcodeString(instr->opcode()));
  }
  if (multiple_results) {
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, kernel_prototype.results, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim();
  }
  const llvm_ir::IrArray& result = kernel_prototype.results.front();
  if (has_parallel_config) {
    ParallelPartitionBounds parallel_bounds = EmitParallelPartitionBounds(
        b, kernel_prototype, *parallel_config, instr->shape(), instr->name());
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, result, &parallel_bounds, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim(ShapePartitionAssigner::GetTotalPartitionCount(
        parallel_config->outer_dimension_partitions));
  }
  TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, result, &b)
                         .EmitLoop(llvm_ir::IrName(instr)));
  return se::ThreadDim();
}
int64_t IrEmitter2::ByteSizeOf(const Shape& shape) const {
  return llvm_ir::ByteSizeOf(shape, module_->getDataLayout());
}
void IrEmitter2::AttachInvariantLoadMetadataForLoad(
    llvm::LoadInst* instr) const {
  nested_ir_emitter_->AttachInvariantLoadMetadataForLoad(instr,
                                                         hlo_module_.config());
}
}  