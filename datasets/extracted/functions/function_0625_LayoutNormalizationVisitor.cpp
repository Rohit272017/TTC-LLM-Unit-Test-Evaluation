#include "xla/service/layout_normalization.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
class LayoutNormalizationVisitor : public DfsHloRewriteVisitor {
 public:
  explicit LayoutNormalizationVisitor(
      const CustomCallTransformer& custom_call_transformer = nullptr)
      : custom_call_transformer_(custom_call_transformer) {}
  absl::Status HandleConstant(HloInstruction* hlo) override {
    Literal& literal = *Cast<HloConstantInstruction>(hlo)->mutable_literal();
    if (literal.shape().IsTuple()) {
      return absl::OkStatus();
    }
    const Shape& shape = hlo->shape();
    Shape normalized_shape = Normalize(shape);
    *literal.mutable_shape_do_not_use() = normalized_shape;
    literal.mutable_shape_do_not_use()
        ->mutable_layout()
        ->set_element_size_in_bits(0);
    HloInstruction* bc_to_orig = MakeBitcastHlo(hlo, shape);
    *hlo->mutable_shape() = normalized_shape;
    TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWithDifferentShape(bc_to_orig));
    MarkAsChanged();
    return absl::OkStatus();
  }
  absl::Status HandleSlice(HloInstruction* hlo) override {
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& s = hlo->shape();
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));
    Shape normalized = Normalize(operand_shape);
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return Permute(input, layout_as_permutation);
    };
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_slice,
                        MakeSliceHlo(normalized_input,
                                     normalize_slice_attr(hlo->slice_starts()),
                                     normalize_slice_attr(hlo->slice_limits()),
                                     normalize_slice_attr(hlo->slice_strides()),
                                     &hlo->metadata()));
    *normalized_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    SetVisited(*normalized_slice);
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized_slice, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status DefaultAction(HloInstruction* hlo) override {
    if (!hlo->user_count()) {
      return absl::OkStatus();
    }
    auto users = hlo->users();
    auto shape = hlo->shape();
    if (shape.IsTuple() || shape.IsToken()) {
      return absl::OkStatus();
    }
    auto normalized_shape = Normalize(shape);
    auto bc_to_normalized = MakeBitcastHlo(hlo, normalized_shape);
    SetVisited(*bc_to_normalized);
    auto bc_to_orig = MakeBitcastHlo(bc_to_normalized, shape);
    TF_RETURN_IF_ERROR(hlo->ReplaceUsesWith(users, bc_to_orig));
    MarkAsChanged();
    return absl::OkStatus();
  }
  absl::Status HandleConcatenate(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    int64_t orig_concat_dim = hlo->dimensions(0);
    std::vector<HloInstruction*> normalized_inputs;
    for (HloInstruction* operand : hlo->mutable_operands()) {
      TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
      normalized_inputs.push_back(normalized_input);
    }
    auto normalized_shape = Normalize(s);
    auto layout_as_permutation = ToTransposeDimensions(s.layout());
    int64_t normalized_concat_dim =
        InversePermutation(layout_as_permutation)[orig_concat_dim];
    auto normalized_concat =
        hlo->AddInstruction(HloInstruction::CreateConcatenate(
            normalized_shape, normalized_inputs, normalized_concat_dim));
    SetVisited(*normalized_concat);
    auto bc_to_orig = MakeBitcastHlo(normalized_concat, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleReduceWindow(HloInstruction* hlo) override {
    if (hlo->shape().IsTuple()) {
      return absl::OkStatus();
    }
    HloInstruction* operand = hlo->mutable_operand(0);
    TF_RET_CHECK(hlo->shape().layout() == operand->shape().layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    std::vector<WindowDimension> window_dimensions;
    for (const WindowDimension& d : hlo->window().dimensions()) {
      window_dimensions.push_back(d);
    }
    window_dimensions = Permute(window_dimensions, layout_as_permutation);
    Window new_window;
    for (const WindowDimension& d : window_dimensions) {
      *new_window.add_dimensions() = d;
    }
    TF_ASSIGN_OR_RETURN(
        HloInstruction * rw,
        MakeReduceWindowHlo(normalized_input, hlo->mutable_operand(1),
                            new_window, hlo->called_computations()[0],
                            &hlo->metadata()));
    SetVisited(*rw);
    HloInstruction* bc_to_orig = MakeBitcastHlo(rw, hlo->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleBroadcast(HloInstruction* hlo) override {
    VLOG(3) << "Input broadcast: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(operand->shape().layout());
    std::vector<int64_t> orig_output_layout_as_permutation =
        ToTransposeDimensions(s.layout());
    std::vector<int64_t> br_dimensions;
    if (!hlo->dimensions().empty()) {
      br_dimensions.reserve(hlo->dimensions().size());
      auto inverse_perm = InversePermutation(orig_output_layout_as_permutation);
      for (int64_t dim :
           ComposePermutations(hlo->dimensions(), layout_as_permutation)) {
        br_dimensions.push_back(inverse_perm[dim]);
      }
    }
    auto normalized_broadcast = MakeBroadcastHlo(
        normalized_input, br_dimensions, normalized_shape, &hlo->metadata());
    SetVisited(*normalized_broadcast);
    VLOG(3) << "Generated broadcast: " << normalized_broadcast->ToString();
    auto bc_to_orig = MakeBitcastHlo(normalized_broadcast, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleIota(HloInstruction* hlo) override {
    VLOG(3) << "Input iota: " << hlo->ToString();
    auto s = hlo->shape();
    auto normalized_shape = Normalize(s);
    std::vector<int64_t> orig_output_layout_as_permutation =
        ToTransposeDimensions(s.layout());
    int64_t new_iota_dimension = InversePermutation(
        orig_output_layout_as_permutation)[hlo->dimensions()[0]];
    auto normalized_iota = hlo->AddInstruction(
        HloInstruction::CreateIota(normalized_shape, new_iota_dimension));
    SetVisited(*normalized_iota);
    VLOG(3) << "Generated iota: " << normalized_iota->ToString();
    auto bc_to_orig = MakeBitcastHlo(normalized_iota, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleBitcastConvert(HloInstruction* hlo) override {
    if (hlo->shape().rank() == hlo->operand(0)->shape().rank()) {
      return HandleElementwiseUnary(hlo);
    }
    return DefaultAction(hlo);
  }
  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_shape = operand->shape();
    TF_RET_CHECK(
        Layout::Equal().IgnoreElementSize()(s.layout(), operand_shape.layout()))
        << "Unexpected non-layout preserving elementwise unary: "
        << hlo->ToString();
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));
    PrimitiveType to_element_type = s.element_type();
    HloInstruction* new_unary;
    if (hlo->opcode() == HloOpcode::kConvert) {
      new_unary =
          MakeConvertToHlo(normalized_input, to_element_type, &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kReducePrecision) {
      new_unary =
          MakeReducePrecisionHlo(normalized_input, hlo->exponent_bits(),
                                 hlo->mantissa_bits(), &hlo->metadata());
    } else if (hlo->opcode() == HloOpcode::kBitcastConvert) {
      new_unary = MakeBitcastConvertToHlo(normalized_input, to_element_type,
                                          &hlo->metadata());
    } else {
      TF_ASSIGN_OR_RETURN(
          new_unary,
          MakeUnaryHlo(hlo->opcode(), normalized_input, &hlo->metadata()));
    }
    if (normalized_input != new_unary) {
      SetVisited(*new_unary);
    }
    auto bc_to_orig = MakeBitcastHlo(new_unary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto a = hlo->mutable_operand(0);
    auto b = hlo->mutable_operand(1);
    auto layout_equal = Layout::Equal();
    if (hlo->opcode() == HloOpcode::kCompare) {
      layout_equal.IgnoreElementSize();
    }
    TF_RET_CHECK(layout_equal(a->shape().layout(), s.layout()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(a));
    TF_ASSIGN_OR_RETURN(auto b0, GetNormalizedInput(b));
    HloInstruction* new_binary;
    if (hlo->opcode() == HloOpcode::kCompare) {
      TF_ASSIGN_OR_RETURN(new_binary,
                          MakeCompareHlo(hlo->comparison_direction(), a0, b0,
                                         &hlo->metadata()));
    } else {
      TF_ASSIGN_OR_RETURN(
          new_binary, MakeBinaryHlo(hlo->opcode(), a0, b0, &hlo->metadata()));
    }
    SetVisited(*new_binary);
    auto bc_to_orig = MakeBitcastHlo(new_binary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleReshape(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_RET_CHECK(ShapeUtil::ReshapeIsBitcast(s, operand->shape()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_reshape_s = Normalize(s);
    TF_ASSIGN_OR_RETURN(auto new_reshape,
                        MakeReshapeHlo(normalized_reshape_s, a0));
    SetVisited(*new_reshape);
    auto bc_to_orig = MakeBitcastHlo(new_reshape, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleScatter(HloInstruction* hlo) override {
    auto* scatter = Cast<HloScatterInstruction>(hlo);
    std::vector<HloInstruction*> normalized_operands;
    normalized_operands.reserve(scatter->scatter_operand_count());
    Shape operand_shape = scatter->scatter_operands().front()->shape();
    for (HloInstruction* operand : scatter->scatter_operands()) {
      if (operand->shape().layout() != operand_shape.layout()) {
        return FailedPrecondition(
            "All scatter operands must have the same layout");
      }
      TF_ASSIGN_OR_RETURN(auto normalized_operand, GetNormalizedInput(operand));
      normalized_operands.push_back(normalized_operand);
    }
    std::vector<HloInstruction*> normalized_updates;
    normalized_updates.reserve(scatter->scatter_operand_count());
    Shape update_shape = scatter->scatter_updates().front()->shape();
    for (HloInstruction* operand : scatter->scatter_updates()) {
      if (operand->shape().layout() != update_shape.layout()) {
        return FailedPrecondition(
            "All scatter updates must have the same layout");
      }
      TF_ASSIGN_OR_RETURN(auto normalized_update, GetNormalizedInput(operand));
      normalized_updates.push_back(normalized_update);
    }
    const auto& dims = scatter->scatter_dimension_numbers();
    if (scatter->scatter_updates().front()->shape().rank() -
            dims.update_window_dims_size() >
        1) {
      return FailedPrecondition(
          "There should be just a single scatter dimension. Make sure to run "
          "ScatterSimplifier before LayoutNormalization");
    }
    TF_ASSIGN_OR_RETURN(auto normalized_indices,
                        GetNormalizedInput(scatter->scatter_indices()));
    auto indices_permutation = InversePermutation(
        ToTransposeDimensions(scatter->scatter_indices()->shape().layout()));
    auto layout_permutation =
        ToTransposeDimensions(scatter->scatter_operands()[0]->shape().layout());
    auto operand_permutation = InversePermutation(layout_permutation);
    auto update_permutation = InversePermutation(
        ToTransposeDimensions(scatter->scatter_updates()[0]->shape().layout()));
    ScatterDimensionNumbers normalized_dims;
    normalized_dims.set_index_vector_dim(
        indices_permutation[dims.index_vector_dim()]);
    for (int64_t dim : dims.scatter_dims_to_operand_dims()) {
      normalized_dims.add_scatter_dims_to_operand_dims(
          operand_permutation[dim]);
    }
    std::vector<int64_t> normalized_update_window_dims;
    normalized_update_window_dims.reserve(dims.update_window_dims_size());
    for (int64_t dim : dims.update_window_dims()) {
      normalized_update_window_dims.push_back(update_permutation[dim]);
    }
    std::vector<int64_t> window_dimensions(operand_permutation.size());
    for (int64_t i = 0, j = 0, k = 0; i < window_dimensions.size(); ++i) {
      if (j < dims.inserted_window_dims_size() &&
          dims.inserted_window_dims(j) == i) {
        window_dimensions[i] = -1;
        ++j;
      } else {
        window_dimensions[i] = normalized_update_window_dims[k];
        ++k;
      }
    }
    std::vector<int64_t> permuted_window_dimensions =
        ComposePermutations(window_dimensions, layout_permutation);
    for (int64_t i = 0; i < permuted_window_dimensions.size(); ++i) {
      if (permuted_window_dimensions[i] == -1) {
        normalized_dims.add_inserted_window_dims(i);
      } else {
        normalized_dims.add_update_window_dims(permuted_window_dimensions[i]);
      }
    }
    auto normalized_shape = normalized_operands.front()->shape();
    if (scatter->shape().IsTuple()) {
      std::vector<Shape> tuple_shapes;
      tuple_shapes.reserve(normalized_operands.size());
      for (HloInstruction* operand : normalized_operands) {
        tuple_shapes.push_back(operand->shape());
      }
      normalized_shape = ShapeUtil::MakeTupleShape(tuple_shapes);
    }
    auto normalized_scatter = hlo->AddInstruction(HloInstruction::CreateScatter(
        normalized_shape, normalized_operands, normalized_indices,
        normalized_updates, scatter->to_apply(), normalized_dims,
        scatter->indices_are_sorted(), scatter->unique_indices()));
    SetVisited(*normalized_scatter);
    auto bc_to_orig = MakeBitcastHlo(normalized_scatter, scatter->shape());
    TF_RETURN_IF_ERROR(ReplaceInstruction(scatter, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleTranspose(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_s = operand->shape();
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_shape = Normalize(s);
    VLOG(3) << "Input transpose: " << hlo->ToString();
    if (!ShapeUtil::TransposeIsBitcast(s, operand_s, hlo->dimensions())) {
      auto l0_perm =
          InversePermutation(ToTransposeDimensions(operand_s.layout()));
      auto l_perm = ToTransposeDimensions(s.layout());
      auto t = ComposePermutations(l0_perm, hlo->dimensions());
      auto dimensions = ComposePermutations(t, l_perm);
      auto normalized_transpose = hlo->AddInstruction(
          HloInstruction::CreateTranspose(normalized_shape, a0, dimensions));
      SetVisited(*normalized_transpose);
      VLOG(3) << "Generated normalized physical transpose: "
              << normalized_transpose->ToString();
      auto bc_to_orig = MakeBitcastHlo(normalized_transpose, s);
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    } else {
      auto bc_to_orig = MakeBitcastHlo(a0, s, &hlo->metadata());
      TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    }
    return absl::OkStatus();
  }
  absl::Status HandleCopy(HloInstruction* hlo) override {
    VLOG(3) << "Processing copy: " << hlo->ToString();
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto s_normalized = Normalize(s);
    auto l0_perm =
        InversePermutation(ToTransposeDimensions(operand->shape().layout()));
    auto l_perm = ToTransposeDimensions(s.layout());
    auto dimensions = ComposePermutations(l0_perm, l_perm);
    auto t = hlo->AddInstruction(
        HloInstruction::CreateTranspose(s_normalized, a0, dimensions));
    SetVisited(*t);
    auto bc_to_orig = MakeBitcastHlo(t, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleReverse(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    std::vector<int64_t> new_dimensions;
    new_dimensions.reserve(hlo->dimensions().size());
    auto inverse_perm = InversePermutation(layout_as_permutation);
    for (int64_t dim : hlo->dimensions()) {
      new_dimensions.push_back(inverse_perm[dim]);
    }
    absl::c_sort(new_dimensions);
    auto normalized_reverse = hlo->AddInstruction(
        HloInstruction::CreateReverse(a0->shape(), a0, new_dimensions));
    SetVisited(*normalized_reverse);
    auto bc_to_orig = MakeBitcastHlo(normalized_reverse, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandlePad(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto padded_by = hlo->mutable_operand(1);
    auto padded_config = hlo->padding_config();
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));
    auto s_normalized = Normalize(s);
    auto layout_as_permutation = ToTransposeDimensions(s.layout());
    PaddingConfig new_padding;
    new_padding.mutable_dimensions()->Reserve(s_normalized.dimensions_size());
    for (int dim = 0; dim < s_normalized.dimensions_size(); dim++) {
      new_padding.add_dimensions();
    }
    auto inverse_perm = InversePermutation(layout_as_permutation);
    for (int dim = 0; dim < s.dimensions_size(); dim++) {
      int tr_dim = static_cast<int>(inverse_perm[dim]);
      *new_padding.mutable_dimensions(tr_dim) = padded_config.dimensions(dim);
    }
    auto padded_normalized = hlo->AddInstruction(HloInstruction::CreatePad(
        s_normalized, normalized_input, padded_by, new_padding));
    SetVisited(*padded_normalized);
    auto bc_to_orig = MakeBitcastHlo(padded_normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleCustomCall(HloInstruction* hlo) override {
    if (custom_call_transformer_) {
      TF_ASSIGN_OR_RETURN(
          std::optional<HloInstruction*> transformed_custom_call,
          custom_call_transformer_(Cast<HloCustomCallInstruction>(hlo)));
      if (transformed_custom_call) {
        SetVisited(*(*transformed_custom_call)->operand(0));
        TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, *transformed_custom_call));
        return absl::OkStatus();
      }
    }
    return DefaultAction(hlo);
  }
  absl::Status HandleSelect(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_input,
                        GetNormalizedInput(operand));
    Shape normalized = Normalize(operand_shape);
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    std::vector<HloInstruction*> new_start_indices =
        GetNewStartIdxs(hlo, 1, layout_as_permutation);
    auto normalize_slice_attr = [&](absl::Span<int64_t const> input) {
      return Permute(input, layout_as_permutation);
    };
    TF_ASSIGN_OR_RETURN(
        HloInstruction * normalized_dynamic_slice,
        MakeDynamicSliceHlo(normalized_input, new_start_indices,
                            normalize_slice_attr(hlo->dynamic_slice_sizes()),
                            &hlo->metadata()));
    *normalized_dynamic_slice->mutable_shape()->mutable_layout() =
        normalized_input->shape().layout();
    SetVisited(*normalized_dynamic_slice);
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized_dynamic_slice, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleDynamicUpdateSlice(HloInstruction* hlo) override {
    const Shape& s = hlo->shape();
    HloInstruction* operand = hlo->mutable_operand(0);
    HloInstruction* update = hlo->mutable_operand(1);
    const Shape& operand_shape = operand->shape();
    TF_RET_CHECK(s.layout() == operand_shape.layout());
    std::vector<int64_t> layout_as_permutation =
        ToTransposeDimensions(hlo->shape().layout());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                        GetNormalizedInput(operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * new_update,
                        GetNormalizedInput(update));
    std::vector<HloInstruction*> new_start_indices =
        GetNewStartIdxs(hlo, 2, layout_as_permutation);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dus,
        MakeDynamicUpdateSliceHlo(new_operand, new_update, new_start_indices,
                                  &hlo->metadata()));
    *new_dus->mutable_shape()->mutable_layout() = new_operand->shape().layout();
    SetVisited(*new_dus);
    HloInstruction* bc_to_orig = MakeBitcastHlo(new_dus, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  absl::Status HandleClamp(HloInstruction* hlo) override {
    return HandleTernary(hlo);
  }
 private:
  absl::Status HandleTernary(HloInstruction* hlo) {
    Shape s = hlo->shape();
    HloOpcode opcode = hlo->opcode();
    TF_RET_CHECK(opcode == HloOpcode::kClamp || opcode == HloOpcode::kSelect);
    HloInstruction* arg0 = hlo->mutable_operand(0);
    HloInstruction* arg1 = hlo->mutable_operand(1);
    HloInstruction* arg2 = hlo->mutable_operand(2);
    if (opcode == HloOpcode::kClamp) {
      TF_RET_CHECK(arg1->shape().layout() == s.layout());
    } else if (opcode == HloOpcode::kSelect) {
      TF_RET_CHECK(arg1->shape().layout() == s.layout());
      TF_RET_CHECK(arg2->shape().layout() == s.layout());
    } else {
      TF_RET_CHECK(false);
    }
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg0,
                        GetNormalizedInput(arg0));
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg1,
                        GetNormalizedInput(arg1));
    TF_ASSIGN_OR_RETURN(HloInstruction * normalized_arg2,
                        GetNormalizedInput(arg2));
    TF_ASSIGN_OR_RETURN(Shape new_shape, ShapeInference::InferTernaryOpShape(
                                             opcode, normalized_arg0,
                                             normalized_arg1, normalized_arg2));
    HloInstruction* normalized = hlo->parent()->AddInstruction(
        HloInstruction::CreateTernary(new_shape, opcode, normalized_arg0,
                                      normalized_arg1, normalized_arg2));
    hlo->SetupDerivedInstruction(normalized);
    SetVisited(*normalized);
    HloInstruction* bc_to_orig = MakeBitcastHlo(normalized, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return absl::OkStatus();
  }
  std::vector<HloInstruction*> GetNewStartIdxs(
      HloInstruction* hlo, int param_offset,
      const std::vector<int64_t> layout_as_permutation) {
    std::vector<HloInstruction*> start_indices;
    for (int i = param_offset; i < hlo->operand_count(); i++) {
      start_indices.push_back(hlo->mutable_operand(i));
    }
    std::vector<HloInstruction*> permuted_start_indices =
        Permute(start_indices, layout_as_permutation);
    return permuted_start_indices;
  }
  std::vector<int64_t> ToTransposeDimensions(const Layout& l) {
    std::vector<int64_t> out(l.minor_to_major().begin(),
                             l.minor_to_major().end());
    absl::c_reverse(out);
    return out;
  }
  absl::StatusOr<HloInstruction*> GetNormalizedInput(HloInstruction* hlo) {
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kBitcast)
        << "Unexpected HLO input: " << hlo->ToString();
    auto input = hlo->mutable_operand(0);
    auto input_shape = input->shape();
    TF_RET_CHECK(Layout::Equal().IgnoreElementSize()(
        input_shape.layout(),
        LayoutUtil::GetDefaultLayoutForShape(input_shape)));
    return input;
  }
  Shape Normalize(const Shape& s) {
    return ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s);
  }
  CustomCallTransformer custom_call_transformer_;
};
}  
absl::StatusOr<bool> LayoutNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return LayoutNormalizationVisitor{custom_call_transformer_}.RunOnModule(
      module, execution_threads);
}
}  