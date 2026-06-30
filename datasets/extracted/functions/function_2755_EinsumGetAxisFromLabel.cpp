#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/gradients/grad_helper.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
namespace tensorflow {
namespace ops {
namespace {
constexpr absl::string_view kEllipsis = "...";
absl::optional<int> EinsumGetAxisFromLabel(absl::string_view subscripts,
                                           char label) {
  std::vector<absl::string_view> splits = absl::StrSplit(subscripts, kEllipsis);
  auto index = splits[0].find(label);
  if (index != splits[0].npos) {
    return index;
  }
  if (splits.size() < 2) {
    return absl::nullopt;
  }
  index = splits[1].find(label);
  if (index != splits[1].npos) {
    return index - splits[1].length();
  }
  return absl::nullopt;
}
std::tuple<int, absl::optional<int>> EinsumGetBcastSubshape(
    absl::string_view subscripts) {
  int start = subscripts.find(kEllipsis);
  if (start == subscripts.npos) {
    return std::make_tuple(0, 0);
  }
  int remaining = subscripts.length() - (start + kEllipsis.length());
  absl::optional<int> end;
  if (remaining > 0) {
    end = -remaining;
  } else {
    end = absl::nullopt;
  }
  return std::make_tuple(start, end);
}
Output Slice1dHelper(const Scope& scope, Output tensor, int start,
                     absl::optional<int> end) {
  if (end.has_value() && *end > 0) {
    return Slice(scope, tensor, Const(scope, start, TensorShape({1})),
                 Const(scope, *end - start, TensorShape({1})));
  } else {
    return Slice(scope, tensor, Const(scope, start, TensorShape({1})),
                 Add(scope, Shape(scope, tensor), end.value_or(0) - start));
  }
}
std::tuple<std::string, Output, Output> EinsumGetReducedSubscripts(
    const Scope& scope, const absl::btree_set<char>& reduced_label_set,
    Output input_shape, absl::string_view subscripts) {
  const std::string reduced_subs =
      std::string(reduced_label_set.begin(), reduced_label_set.end());
  std::vector<int> reduced_axes;
  reduced_axes.reserve(reduced_subs.size());
  for (const char s : reduced_subs) {
    auto axis = EinsumGetAxisFromLabel(subscripts, s);
    if (!axis.has_value()) {
      scope.UpdateStatus(errors::Internal(
          absl::StrCat("Missing axis", absl::string_view(&s, 1))));
    } else {
      reduced_axes.push_back(*axis);
    }
  }
  std::vector<Output> reduced_dims_inputs;
  reduced_dims_inputs.reserve(reduced_axes.size());
  for (const int i : reduced_axes) {
    if (i < 0) {
      reduced_dims_inputs.push_back(
          Gather(scope, input_shape, Add(scope, Size(scope, input_shape), i)));
    } else {
      reduced_dims_inputs.push_back(Gather(scope, input_shape, i));
    }
  }
  const Output reduced_dims = Stack(scope, reduced_dims_inputs);
  Tensor reduced_axes_tensor(
      DataType::DT_INT32, TensorShape({static_cast<int>(reduced_axes.size())}));
  std::copy_n(reduced_axes.begin(), reduced_axes.size(),
              reduced_axes_tensor.flat<int>().data());
  return std::make_tuple(reduced_subs, reduced_dims,
                         Const(scope, reduced_axes_tensor));
}
Output EinsumGradReducedHelper(const Scope& scope, const Output& output_grad,
                               absl::string_view output_subs,
                               absl::string_view input_subs,
                               const Output& input_shape,
                               const absl::btree_set<char>& reduced_label_set) {
  std::string reduced_subs;
  Output reduced_dims, reduced_axes;
  std::tie(reduced_subs, reduced_dims, reduced_axes) =
      EinsumGetReducedSubscripts(scope, reduced_label_set, input_shape,
                                 input_subs);
  const int distinct_input_labels =
      absl::flat_hash_set<char>(input_subs.begin(), input_subs.end()).size();
  const int distinct_output_labels =
      absl::flat_hash_set<char>(output_subs.begin(), output_subs.end()).size();
  const bool has_repeated_labels =
      (distinct_input_labels + distinct_output_labels) <
      input_subs.length() + output_subs.length();
  std::string input_subs_without_reduced_labels;
  for (const char s : input_subs) {
    if (!absl::c_linear_search(reduced_label_set, s)) {
      input_subs_without_reduced_labels.push_back(s);
    }
  }
  if (!has_repeated_labels &&
      input_subs_without_reduced_labels == output_subs) {
    auto reduced_shape = ReducedShapeHelper(scope, input_shape, reduced_axes);
    return BroadcastTo(scope, Reshape(scope, output_grad, reduced_shape),
                       input_shape);
  }
  Output output_grad_shape = Shape(scope, output_grad);
  auto grad_shape_with_reduced_labels =
      Concat(scope, {reduced_dims, output_grad_shape}, 0);
  auto reduced_shape = Concat(
      scope,
      {Const(scope, 1, TensorShape{static_cast<int>(reduced_label_set.size())}),
       output_grad_shape},
      0);
  Output broadcasted_grad =
      BroadcastTo(scope, Reshape(scope, output_grad, reduced_shape),
                  grad_shape_with_reduced_labels);
  return Einsum(scope, {broadcasted_grad},
                absl::StrCat(reduced_subs, output_subs, "->", input_subs));
}
Output EinsumGradWrt(const Scope& scope, Output output_grad,
                     Output other_operand, Output input_shape,
                     absl::string_view input_subs, absl::string_view other_subs,
                     absl::string_view output_subs) {
  absl::btree_set<char> reduced_label_set(input_subs.begin(), input_subs.end());
  for (const char x : output_subs) {
    reduced_label_set.erase(x);
  }
  for (const char x : other_subs) {
    reduced_label_set.erase(x);
  }
  reduced_label_set.erase('.');
  std::string left_subs;
  for (const char s : input_subs) {
    if (!reduced_label_set.contains(s)) {
      left_subs.push_back(s);
    }
  }
  Output grad_reduced =
      Einsum(scope, {output_grad, other_operand},
             absl::StrCat(output_subs, ",", other_subs, "->", left_subs));
  if (reduced_label_set.empty()) {
    return grad_reduced;
  }
  return EinsumGradReducedHelper(scope, grad_reduced, left_subs, input_subs,
                                 input_shape, reduced_label_set);
}
Status EinsumGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Expect 1 grad input.");
  }
  const Output& grad = grad_inputs[0];
  std::string equation;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "equation", &equation));
  std::vector<absl::string_view> equation_split =
      absl::StrSplit(equation, "->");
  if (equation_split.size() != 2) {
    return errors::InvalidArgument("Equation must contain a single ->");
  }
  const absl::string_view input_subs = equation_split[0];
  const absl::string_view output_subs = equation_split[1];
  if (op.num_inputs() == 1) {
    auto input_shape = Shape(scope, op.input(0));
    absl::btree_set<char> reduced_label_set(input_subs.begin(),
                                            input_subs.end());
    for (const char x : output_subs) {
      reduced_label_set.erase(x);
    }
    reduced_label_set.erase('.');
    if (reduced_label_set.empty()) {
      grad_outputs->push_back(Einsum(
          scope, grad_inputs, absl::StrCat(output_subs, "->", input_subs)));
      return scope.status();
    }
    grad_outputs->push_back(EinsumGradReducedHelper(
        scope, grad, output_subs, input_subs, input_shape, reduced_label_set));
    return scope.status();
  }
  std::vector<absl::string_view> subs = absl::StrSplit(input_subs, ',');
  if (subs.size() != 2) {
    return errors::InvalidArgument("Only 2 inputs are supported");
  }
  std::string x_subs(subs[0]);
  std::string y_subs(subs[1]);
  if (absl::StrContains(output_subs, kEllipsis)) {
    if (!absl::StrContains(x_subs, kEllipsis)) {
      absl::StrAppend(&x_subs, kEllipsis);
    }
    if (!absl::StrContains(y_subs, kEllipsis)) {
      absl::StrAppend(&y_subs, kEllipsis);
    }
  }
  tensorflow::Output x = op.input(0);
  tensorflow::Output y = op.input(1);
  if (DataTypeIsComplex(grad.type())) {
    x = Conj(scope, x);
    y = Conj(scope, y);
  }
  const auto x_shape = Shape(scope, x);
  const auto y_shape = Shape(scope, y);
  Output grad_x =
      EinsumGradWrt(scope, grad, y, x_shape, x_subs, y_subs, output_subs);
  Output grad_y =
      EinsumGradWrt(scope, grad, x, y_shape, y_subs, x_subs, output_subs);
  if (!absl::StrContains(output_subs, kEllipsis)) {
    grad_outputs->push_back(grad_x);
    grad_outputs->push_back(grad_y);
    return scope.status();
  }
  int bx_start, by_start;
  absl::optional<int> bx_end, by_end;
  std::tie(bx_start, bx_end) = EinsumGetBcastSubshape(x_subs);
  std::tie(by_start, by_end) = EinsumGetBcastSubshape(y_subs);
  auto args = internal::BroadcastGradientArgs(
      scope, Slice1dHelper(scope, x_shape, bx_start, bx_end),
      Slice1dHelper(scope, y_shape, by_start, by_end));
  grad_x = Reshape(
      scope, ReduceSum(scope, grad_x, Add(scope, bx_start, args.r0)), x_shape);
  grad_y = Reshape(
      scope, ReduceSum(scope, grad_y, Add(scope, by_start, args.r1)), y_shape);
  grad_outputs->push_back(grad_x);
  grad_outputs->push_back(grad_y);
  return scope.status();
}
REGISTER_GRADIENT_OP("Einsum", EinsumGrad);
}  
}  
}  