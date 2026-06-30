#include "xla/tools/hlo_extractor.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/status.h"
namespace xla {
namespace {
class ExtractionVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ExtractionVisitor(
      const HloInstruction* root_instruction,
      absl::flat_hash_set<const HloInstruction*>* boundary,
      ExtractSelector extract_selector,
      ReplaceTypeSelector replace_type_selector)
      : root_instruction_(root_instruction),
        old_module_(root_instruction->GetModule()),
        module_(std::make_unique<HloModule>(
            "extracted", config_,
            std::make_unique<CompilationEnvironments>(
                old_module_->comp_envs()))),
        clone_context_(module_.get()),
        boundary_(boundary),
        extract_selector_(extract_selector),
        replace_type_selector_(replace_type_selector) {
    for (auto computation : old_module_->computations()) {
      old_computations_to_builders_.insert(
          {computation,
           std::make_unique<HloComputation::Builder>(computation->name())});
    }
    for (auto computation : old_module_->computations()) {
      parameter_numbers_[computation] = 0;
    }
  }
  absl::Status HandleParameter(const HloInstruction* parameter) override {
    return ReplaceWithParameter(parameter);
  }
  absl::Status DefaultAction(const HloInstruction* hlo) override {
    if ((boundary_ != nullptr && boundary_->contains(hlo) > 0) ||
        (extract_selector_ != nullptr && !extract_selector_(hlo))) {
      if (replace_type_selector_ != nullptr) {
        switch (replace_type_selector_(hlo)) {
          case ReplaceType::kReplaceConst:
            return ReplaceWithConstant(hlo);
          case ReplaceType::kReplaceParam:
            CHECK(hlo->parent() == root_instruction_->parent())
                << "Replacing instructions at non-entry computation with "
                   "parameters is not supported.";
            return ReplaceWithParameter(hlo);
          case ReplaceType::kReplaceZeroBroadcast:
            return ReplaceWithConstantBroadcast(
                hlo, ReplaceType::kReplaceZeroBroadcast);
          case ReplaceType::kReplaceRandomBroadcast:
            return ReplaceWithConstantBroadcast(
                hlo, ReplaceType::kReplaceRandomBroadcast);
          default:
            QCHECK(false) << "Unsupported replacement type";
        }
      }
      return ReplaceWithParameter(hlo);
    }
    std::vector<HloInstruction*> new_operands;
    for (auto operand : hlo->operands()) {
      new_operands.push_back(clone_context_.GetInstruction(operand));
    }
    auto instruction =
        hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context_);
    auto it = old_computations_to_builders_.find(hlo->parent());
    CHECK(it != old_computations_to_builders_.end());
    auto builder = it->second.get();
    builder->AddInstruction(std::move(instruction));
    if (hlo->IsRoot() && hlo != root_instruction_) {
      CHECK(clone_context_.FindComputation(hlo->parent()) == nullptr);
      auto new_computation = module_->AddEmbeddedComputation(builder->Build());
      clone_context_.MapComputation(hlo->parent(), new_computation);
    }
    return absl::OkStatus();
  }
  absl::Status FinishVisit(const HloInstruction* ) override {
    auto new_entry_computation = module_->AddEntryComputation(
        old_computations_to_builders_.at(root_instruction_->parent())->Build());
    clone_context_.MapComputation(root_instruction_->parent(),
                                  new_entry_computation);
    for (auto computation : old_module_->MakeComputationPostOrder()) {
      for (auto old_instruction : computation->MakeInstructionPostOrder()) {
        if (auto new_instruction =
                clone_context_.FindInstruction(old_instruction)) {
          new_instruction->SetAndSanitizeName(old_instruction->name());
        }
      }
    }
    for (HloInstruction* instruction : extra_created_instructions_) {
      module_->SetAndUniquifyInstrName(instruction, instruction->name());
    }
    return absl::OkStatus();
  }
  HloModule* module() { return module_.get(); }
  std::unique_ptr<HloModule> ConsumeModule() { return std::move(module_); }
 private:
  absl::Status ReplaceWithConstant(const HloInstruction* hlo) {
    absl::StatusOr<Literal> literal_status = MakeFakeLiteral(hlo->shape());
    TF_CHECK_OK(literal_status.status());
    auto new_const =
        HloInstruction::CreateConstant(std::move(literal_status.value()));
    clone_context_.MapInstruction(hlo, new_const.get());
    auto it = old_computations_to_builders_.find(hlo->parent());
    CHECK(it != old_computations_to_builders_.end());
    auto builder = it->second.get();
    builder->AddInstruction(std::move(new_const));
    return absl::OkStatus();
  }
  absl::Status ReplaceWithParameter(const HloInstruction* hlo) {
    CHECK(parameter_numbers_.contains(hlo->parent()));
    auto new_parameter = HloInstruction::CreateParameter(
        parameter_numbers_.at(hlo->parent())++, hlo->shape(), hlo->name());
    clone_context_.MapInstruction(hlo, new_parameter.get());
    CHECK(old_computations_to_builders_.contains(hlo->parent()));
    auto builder = old_computations_to_builders_[hlo->parent()].get();
    builder->AddInstruction(std::move(new_parameter));
    return absl::OkStatus();
  }
  HloInstruction* ReplaceWithConstantBroadcastHelper(
      const Shape& shape, HloComputation::Builder* builder,
      ReplaceType replace_type) {
    if (shape.IsTuple()) {
      std::vector<HloInstruction*> tuple_operands;
      for (const auto& subshape : shape.tuple_shapes()) {
        tuple_operands.push_back(ReplaceWithConstantBroadcastHelper(
            subshape, builder, replace_type));
      }
      auto zero_tuple =
          builder->AddInstruction(HloInstruction::CreateTuple(tuple_operands));
      extra_created_instructions_.push_back(zero_tuple);
      return zero_tuple;
    } else {
      Shape constant_shape = ShapeUtil::MakeShape(shape.element_type(), {});
      HloInstruction* constant_instruction;
      CHECK(replace_type == ReplaceType::kReplaceZeroBroadcast ||
            replace_type == ReplaceType::kReplaceRandomBroadcast);
      if (replace_type == ReplaceType::kReplaceZeroBroadcast) {
        constant_instruction =
            builder->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(constant_shape.element_type())));
      } else {
        absl::StatusOr<Literal> literal_status =
            MakeFakeLiteral(constant_shape);
        TF_CHECK_OK(literal_status.status());
        constant_instruction = builder->AddInstruction(
            HloInstruction::CreateConstant(std::move(literal_status.value())));
      }
      extra_created_instructions_.push_back(constant_instruction);
      auto broadcast_constant_instruction = builder->AddInstruction(
          HloInstruction::CreateBroadcast(shape, constant_instruction, {}));
      extra_created_instructions_.push_back(broadcast_constant_instruction);
      return broadcast_constant_instruction;
    }
  }
  absl::Status ReplaceWithConstantBroadcast(const HloInstruction* hlo,
                                            ReplaceType replace_type) {
    CHECK(replace_type == ReplaceType::kReplaceZeroBroadcast ||
          replace_type == ReplaceType::kReplaceRandomBroadcast);
    CHECK(old_computations_to_builders_.contains(hlo->parent()));
    auto builder = old_computations_to_builders_[hlo->parent()].get();
    HloInstruction* zero_broadcast =
        ReplaceWithConstantBroadcastHelper(hlo->shape(), builder, replace_type);
    clone_context_.MapInstruction(hlo, zero_broadcast);
    return absl::OkStatus();
  }
  const HloInstruction* root_instruction_;
  HloModule* old_module_;
  HloModuleConfig config_;
  std::unique_ptr<HloModule> module_;
  HloCloneContext clone_context_;
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloComputation::Builder>>
      old_computations_to_builders_;
  absl::flat_hash_map<const HloComputation*, int> parameter_numbers_;
  absl::flat_hash_set<const HloInstruction*>* boundary_;
  ExtractSelector extract_selector_;
  ReplaceTypeSelector replace_type_selector_;
  std::vector<HloInstruction*> extra_created_instructions_;
};
void ComputeBoundary(const HloInstruction* root, int64_t limit,
                     absl::flat_hash_set<const HloInstruction*>* boundary) {
  std::deque<const HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, int64_t> visited;
  worklist.push_back(root);
  visited.emplace(root, 0);
  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    int64_t hops = visited[hlo];
    if (hops > limit) {
      boundary->insert(hlo);
      continue;
    }
    for (const HloInstruction* operand : hlo->operands()) {
      if (visited.count(operand)) {
        continue;
      }
      worklist.push_back(operand);
      visited.emplace(operand, hops + 1);
    }
  }
}
}  
std::unique_ptr<HloModule> ExtractModule(
    const HloInstruction* instruction, int64_t height,
    ExtractSelector extract_selector, ReplaceTypeSelector replace_type_selector,
    bool cross_computation) {
  QCHECK(height == -1 || !cross_computation)
      << "Boundary cannnot be calculated across the computations.";
  absl::flat_hash_set<const HloInstruction*> boundary;
  if (height != -1) {
    ComputeBoundary(instruction, height, &boundary);
  }
  ExtractionVisitor visitor(instruction, &boundary, extract_selector,
                            replace_type_selector);
  TF_CHECK_OK(instruction->Accept(&visitor, true,
                                  false,
                                  cross_computation));
  ExtractionVisitor cleanup_visitor(
      visitor.module()->entry_computation()->root_instruction(),
      nullptr,
      nullptr,
      nullptr);
  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor, true,
      false,
      false));
  HloVerifier verifier(false,
                       true);
  TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  return cleanup_visitor.ConsumeModule();
}
}  