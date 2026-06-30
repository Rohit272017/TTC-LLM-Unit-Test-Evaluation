#include "xla/service/gpu/transforms/command_buffer_scheduling.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla::gpu {
using CommandBuffer = CommandBufferScheduling::CommandBuffer;
using CommandBufferConfig = CommandBufferScheduling::CommandBufferConfig;
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config);
static bool IsConstant(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kConstant;
}
static bool IsParameter(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kParameter;
}
static bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
};
static bool IsAsyncStartCommand(const HloInstruction* hlo,
                                const CommandBufferConfig& config) {
  if (hlo->opcode() == HloOpcode::kAllReduceStart ||
      hlo->opcode() == HloOpcode::kAllGatherStart) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }
  if (hlo->opcode() == HloOpcode::kAsyncStart) {
    if (IsCublasGemm(*hlo->async_wrapped_instruction())) {
      return config.enabled_commands.contains(DebugOptions::CUBLAS);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kFusion) {
      return config.enabled_commands.contains(DebugOptions::FUSION);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kReduceScatter ||
        hlo->async_wrapped_opcode() == HloOpcode::kAllToAll) {
      return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
    }
  }
  if (hlo->opcode() == HloOpcode::kReduceScatter ||
      hlo->opcode() == HloOpcode::kAllToAll) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }
  return false;
}
static bool IsAsyncDoneCommand(const HloInstruction* hlo,
                               const CommandBufferConfig& config) {
  if (hlo->opcode() == HloOpcode::kAllReduceDone ||
      hlo->opcode() == HloOpcode::kAllGatherDone) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }
  if (hlo->opcode() == HloOpcode::kAsyncDone) {
    if (IsCublasGemm(*hlo->async_wrapped_instruction())) {
      return config.enabled_commands.contains(DebugOptions::CUBLAS);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kFusion) {
      return config.enabled_commands.contains(DebugOptions::FUSION);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kReduceScatter ||
        hlo->async_wrapped_opcode() == HloOpcode::kAllToAll) {
      return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
    }
  }
  return false;
}
static HloInstruction* FindAsyncDoneCommand(const HloInstruction* start) {
  if (start->opcode() == HloOpcode::kAllReduceStart ||
      start->opcode() == HloOpcode::kAllGatherStart) {
    CHECK(start->users().size() == 1);  
    return start->users().front();
  } else if (start->opcode() == HloOpcode::kAsyncStart) {
    return start->async_chain_done();
  }
  return nullptr;
}
template <HloOpcode op>
static bool IsCommand(const HloInstruction*, const CommandBufferConfig&);
template <>
bool IsCommand<HloOpcode::kWhile>(const HloInstruction* hlo,
                                  const CommandBufferConfig& config) {
  return config.enabled_commands.contains(DebugOptions::CONDITIONALS) &&
         IsCommand(hlo->while_body(), config) &&
         IsCommand(hlo->while_condition(), config);
}
template <>
bool IsCommand<HloOpcode::kConditional>(const HloInstruction* hlo,
                                        const CommandBufferConfig& config) {
  return config.enabled_commands.contains(DebugOptions::CONDITIONALS) &&
         absl::c_all_of(hlo->branch_computations(),
                        [&](const HloComputation* comp) {
                          return IsCommand(comp, config);
                        });
}
static bool IsCommand(const HloCustomCallInstruction* hlo,
                      const CommandBufferConfig& config) {
  if (config.enabled_commands.contains(DebugOptions::CUBLAS) &&
      IsLegacyCublasMatmul(*hlo)) {
    return true;
  }
  if (config.enabled_commands.contains(DebugOptions::CUBLASLT) &&
      (IsCublasLtMatmul(*hlo) || IsCublasLtMatmulF8(*hlo))) {
    return true;
  }
  if (config.enabled_commands.contains(DebugOptions::CUDNN) &&
      IsCustomCallTofMHA(*hlo)) {
    VLOG(3) << "Recording FusedMHA, target " << hlo->custom_call_target()
            << " into command buffer.";
    return true;
  }
  if (!config.enabled_commands.contains(DebugOptions::CUSTOM_CALL)) {
    return false;
  }
  if (config.enabled_legacy_custom_call_targets.contains(
          hlo->custom_call_target())) {
    VLOG(3) << "Recording legacy custom call target "
            << hlo->custom_call_target() << " into command buffer.";
    return true;
  }
  auto registration = ffi::FindHandler(hlo->custom_call_target(), "gpu");
  return registration.ok()
             ? ffi::IsCommandBufferCompatible(registration->traits)
             : false;
}
static bool IsCommand(const HloInstruction* hlo,
                      const CommandBufferConfig& config) {
  if (auto* fusion = DynCast<HloFusionInstruction>(hlo)) {
    auto gpu_config = fusion->backend_config<GpuBackendConfig>();
    const FusionBackendConfig& backend_config =
        gpu_config->fusion_backend_config();
    if (backend_config.kind() == kCuDnnFusionKind) {
      return config.enabled_commands.contains(DebugOptions::CUDNN);
    }
    const auto& custom_config = backend_config.custom_fusion_config();
    if (custom_config.name() == "address_computation") {
      auto fusion_analysis =
          HloFusionAnalysis::Create(*hlo, config.device_description);
      const HloFusionAdaptor& adaptor = fusion_analysis.fusion();
      auto hero_adaptor =
          HloBfsFindIf(adaptor.GetRoots(), adaptor, [](auto node) {
            return node.opcode() == HloOpcode::kCustomCall ||
                   node.opcode() == HloOpcode::kReduceScatter;
          });
      const HloInstruction* hero = &hero_adaptor->instruction();
      return IsCommand(hero, config) || IsAsyncStartCommand(hero, config);
    }
    if (custom_config.name() == "dynamic_address_computation") {
      return false;
    }
    return config.enabled_commands.contains(DebugOptions::FUSION);
  }
  if (auto* sort = DynCast<HloSortInstruction>(hlo))
    return config.enabled_commands.contains(DebugOptions::FUSION);
  if (hlo->opcode() == HloOpcode::kPartitionId ||
      hlo->opcode() == HloOpcode::kReplicaId) {
    return config.enabled_commands.contains(DebugOptions::FUSION);
  }
  if (auto* custom_call = DynCast<HloCustomCallInstruction>(hlo))
    return IsCommand(custom_call, config);
  if (hlo->opcode() == HloOpcode::kWhile)
    return IsCommand<HloOpcode::kWhile>(hlo, config);
  if (hlo->opcode() == HloOpcode::kConditional)
    return IsCommand<HloOpcode::kConditional>(hlo, config);
  return false;
}
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config) {
  return absl::c_all_of(
      computation->instructions(), [&](const HloInstruction* inst) {
        return IsNoOp(inst) || IsConstant(inst) || IsParameter(inst) ||
               IsCommand(inst, config) || IsAsyncStartCommand(inst, config) ||
               IsAsyncDoneCommand(inst, config);
      });
}
static void RemoveTrailingNoOps(HloInstructionSequence& seq) {
  std::vector<HloInstruction*> instructions = seq.instructions();
  for (int i = instructions.size() - 1; i >= 0; i--) {
    if (HloInstruction* inst = instructions[i]; IsNoOp(inst)) {
      seq.remove_instruction(inst);
    } else {
      break;
    }
  }
}
std::vector<HloInstructionSequence>
CommandBufferScheduling::CollectCommandBufferSequences(
    const HloInstructionSequence schedule, const CommandBufferConfig& config,
    int32_t min_num_commands) {
  std::vector<HloInstructionSequence> sequences;
  HloInstructionSequence current_seq;
  int64_t num_commands_in_current_seq = 0;
  auto collect_current_seq = [&]() {
    if (num_commands_in_current_seq >= std::max(1, min_num_commands)) {
      RemoveTrailingNoOps(current_seq);
      sequences.push_back(std::move(current_seq));
    }
    current_seq = HloInstructionSequence();
    num_commands_in_current_seq = 0;
  };
  auto& instructions = schedule.instructions();
  auto collect_async_region = [&](const HloInstruction* start) {
    auto get_index = [&](const HloInstruction* inst) -> size_t {
      auto it = std::find(instructions.begin(), instructions.end(), inst);
      return std::distance(instructions.begin(), it);
    };
    HloInstructionSequence seq;
    size_t done_index = get_index(FindAsyncDoneCommand(start));
    for (size_t i = get_index(start); i <= done_index; i++) {
      HloInstruction* inst = instructions.at(i);
      if (IsAsyncStartCommand(inst, config)) {
        const HloInstruction* done = FindAsyncDoneCommand(inst);
        done_index = std::max(done_index, get_index(done));
      }
      seq.push_back(inst);
    }
    return seq;
  };
  auto check_async_region = [&](const HloInstructionSequence& seq) {
    if (!absl::c_all_of(seq.instructions(), [&](HloInstruction* inst) {
          return IsNoOp(inst) || IsCommand(inst, config) ||
                 IsAsyncStartCommand(inst, config) ||
                 IsAsyncDoneCommand(inst, config);
        })) {
      return false;
    }
    absl::flat_hash_set<HloInstruction*> done_instructions;
    for (const HloInstruction* inst : seq.instructions()) {
      if (IsAsyncStartCommand(inst, config)) {
        done_instructions.insert(FindAsyncDoneCommand(inst));
      }
      if (IsAsyncDoneCommand(inst, config)) {
        if (!done_instructions.contains(inst)) {
          return false;
        }
      }
    }
    return true;
  };
  for (size_t i = 0; i < instructions.size(); i++) {
    HloInstruction* inst = instructions.at(i);
    if (IsNoOp(inst) && num_commands_in_current_seq) {
      current_seq.push_back(inst);
      continue;
    }
    if (IsCommand(inst, config)) {
      num_commands_in_current_seq++;
      current_seq.push_back(inst);
      continue;
    }
    if (IsAsyncStartCommand(inst, config)) {
      HloInstructionSequence seq = collect_async_region(inst);
      if (check_async_region(seq)) {
        num_commands_in_current_seq += seq.instructions().size();
        for (HloInstruction* inst : seq.instructions()) {
          current_seq.push_back(inst);
        }
        i += seq.instructions().size() - 1;
        continue;
      }
    }
    collect_current_seq();
  }
  collect_current_seq();
  return sequences;
}
absl::StatusOr<bool> CommandBufferScheduling::MoveParametersAndConstantsToFront(
    HloComputation* computation) {
  HloInstructionSequence new_sequence;
  HloSchedule& schedule = computation->parent()->schedule();
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(computation);
  for (HloInstruction* inst : sequence.instructions()) {
    if (IsParameter(inst) || IsConstant(inst)) {
      new_sequence.push_back(inst);
      for (HloInstruction* control_predecessor : inst->control_predecessors()) {
        for (HloInstruction* user : inst->users()) {
          TF_RETURN_IF_ERROR(control_predecessor->AddControlDependencyTo(user));
        }
      }
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
    }
  }
  for (HloInstruction* inst : sequence.instructions()) {
    if (!IsParameter(inst) && !IsConstant(inst)) {
      new_sequence.push_back(inst);
    }
  }
  schedule.set_sequence(computation, new_sequence);
  for (auto [old_i, new_i] :
       llvm::zip(sequence.instructions(), new_sequence.instructions())) {
    if (old_i != new_i) return true;
  }
  return false;
}
absl::StatusOr<CommandBuffer> CommandBufferScheduling::PrepareCommandBuffer(
    const HloInstructionSequence& seq, HloModule* module) {
  auto builder = HloComputation::Builder("command_buffer");
  absl::Span<HloInstruction* const> instructions =
      absl::MakeSpan(seq.instructions());
  absl::flat_hash_set<HloInstruction*> in_command_buffer(instructions.begin(),
                                                         instructions.end());
  absl::flat_hash_map<HloInstruction*, HloParameterInstruction*> parameters;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> inst_mapping;
  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      if (auto it = inst_mapping.find(operand); it != inst_mapping.end())
        operands.push_back(it->second);
    }
    return operands;
  };
  for (HloInstruction* inst : instructions) {
    for (HloInstruction* operand : inst->operands()) {
      if (parameters.contains(operand)) continue;
      if (in_command_buffer.contains(operand)) continue;
      int64_t parameter_id = parameters.size();
      auto* parameter = Cast<HloParameterInstruction>(
          builder.AddInstruction(HloInstruction::CreateParameter(
              parameter_id, operand->shape(), "p")));
      parameter->UniquifyName(module);
      parameter->UniquifyId(module);
      inst_mapping[operand] = parameters[operand] = parameter;
    }
  }
  for (HloInstruction* inst : seq.instructions()) {
    HloCloneContext ctx(inst->GetModule());
    for (HloComputation* called_computation : inst->called_computations()) {
      if (called_computation->IsAsyncComputation()) {
        called_computation->RemoveAsyncStart();
      }
      ctx.MapComputation(called_computation, called_computation);
    }
    inst_mapping[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), mapped_operands(inst), &ctx));
    inst_mapping[inst]->UniquifyId(module);
  }
  std::vector<HloInstruction*> arguments(parameters.size());
  for (auto& [argument, parameter] : parameters) {
    arguments[parameter->parameter_number()] = argument;
  }
  std::vector<HloInstruction*> results;
  std::vector<HloInstruction*> returned;
  auto has_external_users = [&](HloInstruction* inst) {
    return inst->IsRoot() || absl::c_any_of(inst->users(), [&](auto* user) {
             return !in_command_buffer.contains(user);
           });
  };
  for (HloInstruction* inst : instructions) {
    if (has_external_users(inst)) {
      results.push_back(inst);
      returned.push_back(inst_mapping[inst]);
    }
  }
  if (returned.size() > 1) {
    HloInstruction* inst =
        builder.AddInstruction(HloInstruction::CreateTuple(returned));
    inst->UniquifyName(module);
    inst->UniquifyId(module);
  }
  std::unique_ptr<HloComputation> comp = builder.Build();
  comp->UniquifyName(module);
  comp->SetUniqueId(comp->root_instruction()->unique_id());
  return CommandBuffer{std::move(arguments), std::move(results),
                       std::move(comp), std::move(inst_mapping)};
}
absl::StatusOr<HloComputation*> CommandBufferScheduling::RewriteCommandBuffer(
    HloComputation* parent, const HloInstructionSequence& seq,
    CommandBuffer command_buffer) {
  if (command_buffer.results.empty())
    return absl::InternalError("command buffer results must not be empty");
  Shape cmd_buffer_result_shape;
  bool has_single_result = command_buffer.results.size() == 1;
  if (has_single_result) {
    cmd_buffer_result_shape = command_buffer.results[0]->shape();
  } else {
    absl::InlinedVector<Shape, 4> shapes;
    shapes.reserve(command_buffer.results.size());
    for (auto* res : command_buffer.results) shapes.push_back(res->shape());
    cmd_buffer_result_shape = ShapeUtil::MakeTupleShape(shapes);
  }
  HloComputation* computation =
      parent->parent()->AddComputation(std::move(command_buffer.computation),
                                       false);
  HloInstruction* call = parent->AddInstruction(HloInstruction::CreateCall(
      cmd_buffer_result_shape, command_buffer.arguments, computation));
  if (has_single_result) {
    TF_RETURN_IF_ERROR(command_buffer.results[0]->ReplaceAllUsesWith(call));
  } else {
    for (int i = 0; i < command_buffer.results.size(); i++) {
      TF_RETURN_IF_ERROR(
          command_buffer.results[i]->ReplaceAllUsesWith(parent->AddInstruction(
              HloInstruction::CreateGetTupleElement(call, i))));
    }
  }
  HloSchedule& schedule = parent->parent()->schedule();
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(parent);
  sequence.replace_instruction(seq.instructions().back(), call);
  HloInstructionSequence cmd_buffer_schedule;
  for (auto* argument : command_buffer.arguments) {
    cmd_buffer_schedule.push_back(command_buffer.inst_mapping[argument]);
  }
  for (auto* inst : seq.instructions()) {
    cmd_buffer_schedule.push_back(command_buffer.inst_mapping[inst]);
  }
  if (!has_single_result) {
    cmd_buffer_schedule.push_back(computation->root_instruction());
  }
  schedule.set_sequence(computation, cmd_buffer_schedule);
  auto& inst_mapping = command_buffer.inst_mapping;
  for (HloInstruction* inst : seq.instructions()) {
    HloInstruction* cmd_inst = inst_mapping[inst];
    for (HloInstruction* predecessor : inst->control_predecessors()) {
      if (auto it = inst_mapping.find(predecessor); it != inst_mapping.end()) {
        HloInstruction* cmd_predecessor = it->second;
        if (IsParameter(cmd_predecessor)) {
          TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(call));
        } else {
          TF_RETURN_IF_ERROR(cmd_predecessor->AddControlDependencyTo(cmd_inst));
        }
      } else {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(call));
      }
    }
    for (HloInstruction* successor : inst->control_successors()) {
      if (auto it = inst_mapping.find(successor); it != inst_mapping.end()) {
        HloInstruction* cmd_successor = it->second;
        TF_RETURN_IF_ERROR(cmd_inst->AddControlDependencyTo(cmd_successor));
      } else {
        TF_RETURN_IF_ERROR(call->AddControlDependencyTo(successor));
      }
    }
    TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
  }
  for (int32_t i = seq.instructions().size() - 1; i >= 0; i--) {
    TF_RETURN_IF_ERROR(parent->RemoveInstruction(seq.instructions()[i]));
  }
  return computation;
}
CommandBufferScheduling::CommandBufferScheduling(
    const se::DeviceDescription& device_description)
    : device_description_(device_description) {}
absl::StatusOr<bool> CommandBufferScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) return Internal("module is not scheduled");
  const DebugOptions& debug_options = module->config().debug_options();
  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }
  absl::flat_hash_set<std::string> legacy_custom_call_targets;
  for (const auto& target :
       debug_options.legacy_command_buffer_custom_call_targets()) {
    legacy_custom_call_targets.insert(target);
  }
  CommandBufferConfig config{std::move(commands),
                             std::move(legacy_custom_call_targets),
                             device_description_};
  static constexpr auto kRequireConditionals = {DebugOptions::CONDITIONALS};
  static constexpr auto kRequireTracing = {
      DebugOptions::CUBLAS, DebugOptions::CUBLASLT, DebugOptions::CUDNN,
      DebugOptions::CUSTOM_CALL, DebugOptions::COLLECTIVES};
  auto erase = [&](absl::Span<const DebugOptions::CommandBufferCmdType> cmds) {
    for (auto cmd : cmds) {
      if (config.enabled_commands.erase(cmd)) {
        VLOG(1) << "Removed command buffer support for "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " as it's not supported with gpu toolkit version "
                << device_description_.runtime_version()
                << " and driver version "
                << device_description_.driver_version()
                << ". This might negatively impact peformance. To enable "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " support in command buffers use cuda-compat package: "
#if defined(PLATFORM_GOOGLE)
                << "set CUDA_COMPAT_LOAD=1 env variable.";
#else
                << "https:
#endif
      }
    }
  };
  auto erase_cuda = [&](const se::CudaComputeCapability& cuda_comp) {
    if (std::min(device_description_.runtime_version(),
                 device_description_.driver_version()) <
        se::SemanticVersion{12, 3, 0}) {
      erase(kRequireTracing);  
      erase(kRequireConditionals);  
    }
  };
  auto erase_rocm = [&](const se::RocmComputeCapability& rocm_comp) {
    erase(kRequireConditionals);  
  };
  std::visit(VariantVisitor{erase_cuda, erase_rocm},
             device_description_.gpu_compute_capability());
  auto order = module->MakeComputationPostOrder();
  std::reverse(order.begin(), order.end());
  absl::flat_hash_set<HloComputation*> processed_command_buffers;
  auto changed = false;
  for (HloComputation* comp : order) {
    if (comp->IsFusionComputation() || comp->IsAsyncComputation() ||
        comp->IsCustomCallComputation())
      continue;
    if (processed_command_buffers.contains(comp)) continue;
    TF_ASSIGN_OR_RETURN(bool changed_, MoveParametersAndConstantsToFront(comp));
    changed |= changed_;
    std::vector<HloInstructionSequence> sequences =
        CollectCommandBufferSequences(
            module->schedule().sequence(comp), config,
            debug_options.xla_gpu_graph_min_graph_size());
    for (const HloInstructionSequence& seq : sequences) {
      TF_ASSIGN_OR_RETURN(CommandBuffer command_buffer,
                          PrepareCommandBuffer(seq, comp->parent()));
      TF_ASSIGN_OR_RETURN(
          HloComputation * command_buffer_computation,
          RewriteCommandBuffer(comp, seq, std::move(command_buffer)));
      changed = true;
      for (HloComputation* called :
           command_buffer_computation->MakeEmbeddedComputationsList()) {
        processed_command_buffers.insert(called);
      }
    }
  }
  TF_RETURN_IF_ERROR(module->schedule().Update());
  return changed;
}
}  