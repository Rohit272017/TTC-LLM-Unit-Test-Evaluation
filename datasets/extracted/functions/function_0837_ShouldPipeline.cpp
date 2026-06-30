#include "xla/service/gpu/gpu_p2p_pipeliner.h"
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/hlo_parser.h"
#include "xla/util.h"
namespace xla {
namespace gpu {
namespace {
bool ShouldPipeline(const HloInstruction* instr) {
  if (!HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(instr)) {
    return false;
  }
  auto it = instr->frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it == instr->frontend_attributes().map().end()) {
    return false;
  }
  auto allowed_predecessor = [&]() {
    return instr->opcode() == HloOpcode::kRecvDone &&
           instr->control_predecessors().size() == 1 &&
           instr->control_predecessors()[0]->opcode() == HloOpcode::kSend;
  };
  if (!instr->control_successors().empty() ||
      (!instr->control_predecessors().empty() && !allowed_predecessor())) {
    return false;
  }
  bool is_pipelined =
      (instr->user_count() == 1 && instr->parent() != nullptr &&
       instr->users()[0] == instr->parent()->root_instruction());
  return !is_pipelined;
}
bool ShouldAllowLoopVariantParameterInChain(const HloInstruction* instr) {
  CHECK(instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->operand(0)->opcode() == HloOpcode::kParameter);
  return true;
}
absl::Status PostprocessP2PImpl(
    HloInstruction* instr,
    std::function<std::string(std::vector<ReplicaGroup>&)> transformer) {
  if (!HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(instr)) {
    return Internal("Expected SendDone/RecvDone as the pipelined collective");
  }
  instr = instr->mutable_operand(0);
  if (!HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend>(instr)) {
    return Internal("Expected Send/Recv as the SendDone/RecvDone operand");
  }
  auto validation_it =
      instr->frontend_attributes().map().find(kSendRecvValidationAttr);
  if (validation_it == instr->frontend_attributes().map().end() ||
      validation_it->second == "invalid") {
    return absl::OkStatus();
  }
  auto statusor_bounds = ParseReplicaGroupsOnly(validation_it->second);
  if (!statusor_bounds.ok()) {
    return statusor_bounds.status();
  }
  std::string validation_attr = transformer(statusor_bounds.value());
  xla::FrontendAttributes attributes = instr->frontend_attributes();
  (*attributes.mutable_map())[kSendRecvValidationAttr] = validation_attr;
  instr->set_frontend_attributes(attributes);
  return absl::OkStatus();
}
absl::Status PostprocessPeeledP2P(HloInstruction* instr) {
  auto transform_bounds = [&](std::vector<ReplicaGroup>& replica_groups) {
    std::vector<std::pair<int64_t, int64_t>> bounds;
    bounds.reserve(replica_groups.size());
    bool all_invalid = true;
    for (const auto& replica_group : replica_groups) {
      int64_t lower_bound = replica_group.replica_ids(0);
      int64_t upper_bound = replica_group.replica_ids(1);
      if (lower_bound <= 0 && upper_bound >= 0) {
        all_invalid = false;
        bounds.push_back({0, 0});
      } else {
        bounds.push_back({1, 0});
      }
    }
    std::string validation_attr;
    if (all_invalid) {
      validation_attr = "invalid";
    } else {
      validation_attr = "{" +
                        absl::StrJoin(bounds, ",",
                                      absl::PairFormatter(
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, "{", value);
                                          },
                                          ",",
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, value, "}");
                                          })) +
                        "}";
    }
    return validation_attr;
  };
  return PostprocessP2PImpl(instr, transform_bounds);
};
absl::Status PostprocessRotatedP2P(HloInstruction* instr) {
  auto transform_bounds = [&](std::vector<ReplicaGroup>& replica_groups) {
    std::vector<std::pair<int64_t, int64_t>> bounds;
    bounds.reserve(replica_groups.size());
    bool all_invalid = true;
    for (const auto& replica_group : replica_groups) {
      int64_t lower_bound = replica_group.replica_ids(0);
      int64_t upper_bound = replica_group.replica_ids(1);
      if (lower_bound <= upper_bound) {
        if (lower_bound >= 1) {
          --lower_bound;
        }
        if (upper_bound >= 1) {
          --upper_bound;
        }
        if (lower_bound <= upper_bound) {
          all_invalid = false;
          bounds.push_back({lower_bound, upper_bound});
        } else {
          bounds.push_back({1, 0});
        }
      } else {
        bounds.push_back({lower_bound, upper_bound});
      }
    }
    std::string validation_attr;
    if (all_invalid) {
      validation_attr = "invalid";
    } else {
      validation_attr = "{" +
                        absl::StrJoin(bounds, ",",
                                      absl::PairFormatter(
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, "{", value);
                                          },
                                          ",",
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, value, "}");
                                          })) +
                        "}";
    }
    return validation_attr;
  };
  return PostprocessP2PImpl(instr, transform_bounds);
}
}  
void AddP2PPipeliner(HloPassPipeline& pipeline) {
  CollectivePipeliner::Config config{
      0,
      INT64_MAX,
      true,
      false,
      true,
      CollectivePipeliner::PipeliningDirection::kBackward,
      ShouldPipeline,
      HloPredicateTrue,
      HloPredicateTrue,
      ShouldAllowLoopVariantParameterInChain,
      true,
       PostprocessPeeledP2P,
       PostprocessRotatedP2P};
  pipeline.AddPass<CollectivePipeliner>(config);
}
}  
}  