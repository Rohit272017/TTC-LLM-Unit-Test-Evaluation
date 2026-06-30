#include "xla/service/llvm_ir/alias_analysis.h"
#include <map>
#include "absl/container/flat_hash_set.h"
#include "llvm/IR/MDBuilder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_value.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
namespace xla {
namespace llvm_ir {
static const BufferAllocation* kParameterAllocation = new BufferAllocation(
    -1, 0, LogicalBuffer::Color(0));
void AliasAnalysis::AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                                    llvm_ir::IrArray* array,
                                                    const ShapeIndex& index) {
  BufferAllocation::Slice buffer_slice;
  if (hlo.opcode() == HloOpcode::kParameter &&
      hlo.parent() == module_.entry_computation()) {
    buffer_slice = BufferAllocation::Slice(kParameterAllocation, 0, 0);
  } else {
    auto unique_slice = assignment_.GetUniqueSlice(&hlo, index);
    if (!unique_slice.ok()) {
      return;
    }
    buffer_slice = unique_slice.value();
  }
  if (module_.config().debug_options().xla_llvm_enable_alias_scope_metadata()) {
    llvm::MDNode*& alias_scope_md = alias_scope_metadata_[buffer_slice];
    if (alias_scope_md == nullptr) {
      alias_scope_md =
          GetAliasScopeMetadataForBuffer(buffer_slice, GetAliasDomain());
    }
    if (alias_scope_md != nullptr) {
      array->AddAliasScopeMetadata(alias_scope_md);
    }
  }
  if (module_.config().debug_options().xla_llvm_enable_noalias_metadata()) {
    llvm::MDNode*& noalias_md = noalias_metadata_[{buffer_slice, &hlo}];
    if (noalias_md == nullptr) {
      noalias_md = GetNoaliasMetadataForBuffer(buffer_slice, GetAliasDomain(),
                                               assignment_, hlo);
    }
    if (noalias_md != nullptr) {
      array->AddNoaliasMetadata(noalias_md);
    }
  }
  if (module_.config()
          .debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    if (hlo.opcode() == HloOpcode::kParameter &&
        hlo.parent() == module_.entry_computation()) {
      array->MarkInvariantOverWholeProgram(context_);
    }
  }
}
llvm::MDNode* AliasAnalysis::GetAliasDomain() {
  llvm::MDBuilder metadata_builder(*context_);
  if (alias_domain_ == nullptr) {
    alias_domain_ =
        metadata_builder.createAliasScopeDomain("XLA global AA domain");
  }
  return alias_domain_;
}
llvm::MDNode* AliasAnalysis::GetAliasScopeMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain) {
  if (buffer_slice.allocation() == kParameterAllocation) {
    return nullptr;
  }
  llvm::MDBuilder metadata_builder(domain->getContext());
  llvm::MDNode* scope = metadata_builder.createAliasScope(
      "buffer: " + buffer_slice.ToString(), domain);
  llvm::MDNode* scope_list = llvm::MDNode::get(domain->getContext(), scope);
  return scope_list;
}
llvm::MDNode* AliasAnalysis::GetNoaliasMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain,
    const BufferAssignment& assignment, const HloInstruction& hlo) {
  std::vector<const HloValue*> worklist;
  absl::flat_hash_set<const HloInstruction*> added_to_worklist;
  auto add_buffers_to_worklist =
      [&](const HloInstruction* instruction) {
        if (instruction->opcode() == HloOpcode::kParameter) {
          return;
        }
        if (added_to_worklist.contains(instruction)) {
          return;
        }
        added_to_worklist.insert(instruction);
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&](const Shape& , const ShapeIndex& index) {
              for (const HloValue* buffer :
                   assignment.GetSourceBuffers(instruction, index)) {
                if (assignment.HasAllocation(*buffer)) {
                  worklist.push_back(buffer);
                }
              }
            });
      };
  for (HloInstruction* user : hlo.users()) {
    add_buffers_to_worklist(user);
    for (HloInstruction* operand : user->operands()) {
      add_buffers_to_worklist(operand);
    }
  }
  add_buffers_to_worklist(&hlo);
  for (HloInstruction* operand : hlo.operands()) {
    add_buffers_to_worklist(operand);
  }
  std::set<BufferAllocation::Slice> buffers;
  for (const HloValue* buffer : worklist) {
    const BufferAllocation::Slice noalias_slice =
        assignment.GetAssignedAllocation(*buffer).GetSlice(*buffer);
    if (!buffer_slice.OverlapsWith(noalias_slice)) {
      buffers.insert(noalias_slice);
      constexpr int kMaxNoAliasSetSize = 500;
      if (buffers.size() >= kMaxNoAliasSetSize) {
        break;
      }
    }
  }
  if (buffers.empty()) {
    return nullptr;
  }
  llvm::MDBuilder metadata_builder(domain->getContext());
  std::vector<llvm::Metadata*> scopes;
  for (const BufferAllocation::Slice noalias_slice : buffers) {
    llvm::MDNode* scope = metadata_builder.createAliasScope(
        "buffer: " + noalias_slice.ToString(), domain);
    scopes.push_back(scope);
  }
  llvm::MDNode* noalias_list =
      llvm::MDNode::get(domain->getContext(), AsArrayRef(scopes));
  return noalias_list;
}
}  
}  