#include "xla/service/gpu/transforms/reduction_splitter.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
class ReductionSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionSplitterVisitor(bool ignore_small_dims)
      : ignore_small_dims_(ignore_small_dims) {}
  absl::Status HandleReduce(HloInstruction *reduce) override {
    VLOG(4) << "Input: " << reduce->ToString();
    if (IsReductionFromOrToContiguousDimensions(*reduce)) {
      VLOG(4) << "Reduction with contiguous dimensions. Return.";
      return absl::OkStatus();
    }
    if (reduce->dimensions().size() < 2) {
      return absl::OkStatus();
    }
    if (!reduce->shape().IsArray()) {
      return absl::OkStatus();
    }
    HloInstruction *operand = reduce->mutable_operand(0);
    const Shape &shape = operand->shape();
    CHECK(shape == LayoutUtil::GetWithDefaultLayout(shape))
        << "Default layout should be enforced on reduction operand";
    for (int64_t i = 0; i < reduce->dimensions().size(); ++i) {
      for (int64_t j = i + 1; j < reduce->dimensions().size(); ++j) {
        CHECK(abs(reduce->dimensions(i) - reduce->dimensions(j)) > 1)
            << "Reduction dimensions must not be consecutive";
      }
    }
    int64_t max_shape_dim = 0;
    int64_t max_reduce_dim = 0;
    const auto &input_shape = reduce->operand(0)->shape();
    for (int64_t i = 0; i < reduce->dimensions().size(); ++i) {
      if (input_shape.dimensions(reduce->dimensions(i)) > max_shape_dim) {
        max_reduce_dim = reduce->dimensions(i);
        max_shape_dim = input_shape.dimensions(max_reduce_dim);
      }
    }
    if (ignore_small_dims_ && max_shape_dim <= 8) {
      return absl::OkStatus();
    }
    VLOG(3) << "Splitting reduction " << reduce->name() << " at dimension "
            << max_reduce_dim;
    std::vector<int64_t> pre_reduce_dims;
    pre_reduce_dims.push_back(max_reduce_dim);
    std::vector<int64_t> pre_reduce_shape_dims(input_shape.dimensions().begin(),
                                               input_shape.dimensions().end());
    pre_reduce_shape_dims.erase(pre_reduce_shape_dims.begin() + max_reduce_dim);
    Shape pre_reduce_shape = ShapeUtil::MakeShape(
        reduce->shape().element_type(), pre_reduce_shape_dims);
    std::unique_ptr<HloInstruction> pre_reduce = HloInstruction::CreateReduce(
        pre_reduce_shape, reduce->mutable_operand(0),
        reduce->mutable_operand(1), pre_reduce_dims, reduce->to_apply());
    pre_reduce->set_metadata(reduce->metadata());
    std::vector<int64_t> final_reduce_dims(reduce->dimensions().begin(),
                                           reduce->dimensions().end());
    final_reduce_dims.erase(
        std::remove(final_reduce_dims.begin(), final_reduce_dims.end(),
                    max_reduce_dim),
        final_reduce_dims.end());
    for (int64_t i = 0; i < final_reduce_dims.size(); ++i) {
      if (final_reduce_dims[i] > max_reduce_dim) {
        final_reduce_dims[i]--;
      }
    }
    std::unique_ptr<HloInstruction> final_reduce = HloInstruction::CreateReduce(
        reduce->shape(),
        reduce->parent()->AddInstruction(std::move(pre_reduce)),
        reduce->mutable_operand(1), final_reduce_dims, reduce->to_apply());
    return ReplaceWithNewInstruction(reduce, std::move(final_reduce));
  }
 private:
  bool ignore_small_dims_;
};
absl::StatusOr<bool> ReductionSplitter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionSplitterVisitor(ignore_small_dims_)
                          .RunOnModule(module, execution_threads));
  return changed;
}
}  
}  