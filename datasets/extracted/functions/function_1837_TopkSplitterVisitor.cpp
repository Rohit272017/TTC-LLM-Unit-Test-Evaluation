#include "xla/service/gpu/transforms/topk_splitter.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
constexpr size_t kRequiredAlignment = 1024;
constexpr size_t kMaximumBatchSize = 1024;
class TopkSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit TopkSplitterVisitor(size_t split_threshold)
      : split_threshold_(split_threshold) {}
  absl::Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return absl::OkStatus();
    }
    HloComputation* comp = inst->parent();
    Shape data_shape = topk->operand(0)->shape();
    bool has_batch = data_shape.dimensions_size() == 2;
    if (has_batch && data_shape.dimensions(0) != 1) {
      return absl::OkStatus();
    }
    size_t n = data_shape.dimensions(has_batch ? 1 : 0);
    int64_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
    if (k > sqrt(n)) {
      return absl::OkStatus();
    }
    if (n % kRequiredAlignment != 0) {
      return absl::OkStatus();
    }
    if (n < split_threshold_) return absl::OkStatus();
    int new_batch =
        std::min(absl::bit_floor(n / split_threshold_), kMaximumBatchSize);
    int new_n = n / new_batch;
    Shape split_input_shape =
        ShapeUtil::MakeShape(data_shape.element_type(), {new_batch, new_n});
    TF_ASSIGN_OR_RETURN(
        HloInstruction * reshaped,
        MakeReshapeHlo(split_input_shape, topk->mutable_operand(0)));
    Shape batch_topk_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(data_shape.element_type(), {new_batch, k}),
         ShapeUtil::MakeShape(S32, {new_batch, k})});
    HloInstruction* batch_topk =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            batch_topk_shape, {reshaped}, topk->to_apply(), "TopK",
            ""));
    TF_ASSIGN_OR_RETURN(HloInstruction * indices,
                        MakeGetTupleElementHlo(batch_topk, 1));
    TF_ASSIGN_OR_RETURN(HloInstruction * values,
                        MakeGetTupleElementHlo(batch_topk, 0));
    Shape iota_shape = ShapeUtil::MakeShape(S32, {new_batch});
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fix,
        MakeBinaryHlo(
            HloOpcode::kMultiply, MakeIotaHlo(comp, iota_shape, 0),
            MakeBroadcastHlo(MakeR0ConstantHlo<int32_t>(comp, new_n),
                             {}, iota_shape)));
    TF_ASSIGN_OR_RETURN(
        indices, MakeBinaryHlo(HloOpcode::kAdd, indices,
                               MakeBroadcastHlo(fix, {0}, indices->shape())));
    Shape linear_index_shape = ShapeUtil::MakeShape(S32, {k * new_batch});
    Shape linear_shape = ShapeUtil::ChangeElementType(
        linear_index_shape, data_shape.element_type());
    Shape linear_sort_shape =
        ShapeUtil::MakeTupleShape({linear_shape, linear_index_shape});
    HloInstruction* aggregated_sort =
        comp->AddInstruction(HloInstruction::CreateSort(
            linear_sort_shape, 0,
            {*MakeReshapeHlo(linear_shape, values),
             *MakeReshapeHlo(linear_index_shape, indices)},
            topk->to_apply(), true));
    auto slice_tuple = [&](HloInstruction* sort, const size_t index) {
      return *MakeReshapeHlo(
          topk->shape().tuple_shapes(index),
          *MakeSliceHlo(*MakeGetTupleElementHlo(sort, index), {0}, {k}, {1}));
    };
    return ReplaceInstruction(topk,
                              comp->AddInstruction(HloInstruction::CreateTuple({
                                  slice_tuple(aggregated_sort, 0),
                                  slice_tuple(aggregated_sort, 1),
                              })));
  }
 private:
  size_t split_threshold_;
};
}  
absl::StatusOr<bool> TopKSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TopkSplitterVisitor(split_threshold_)
      .RunOnModule(module, execution_threads);
}
}  
}  