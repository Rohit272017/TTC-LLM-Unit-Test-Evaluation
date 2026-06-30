#include "xla/service/gpu/model/symbolic_tile.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_map_serialization.h"
namespace xla {
namespace gpu {
namespace {
using ::llvm::SmallVector;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineDimExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::AffineSymbolExpr;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::mlir::MLIRContext;
using Constraint = ConstraintExpression::Constraint;
using ConjointConstraints = ConstraintExpression::ConjointConstraints;
AffineMap SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
    AffineMap affine_map, AffineExpr value, int num_range_vars) {
  CHECK_LE(num_range_vars, affine_map.getNumSymbols());
  MLIRContext* mlir_context = affine_map.getContext();
  int64_t num_dims = affine_map.getNumDims();
  int64_t num_symbols = affine_map.getNumSymbols();
  llvm::DenseMap<AffineExpr, AffineExpr> indices;
  for (int64_t i = 0; i < num_dims; ++i) {
    indices[getAffineDimExpr(i, mlir_context)] = value;
  }
  for (int64_t i = 0; i < num_range_vars; ++i) {
    indices[getAffineSymbolExpr(i, mlir_context)] = value;
  }
  return simplifyAffineMap(affine_map.replace(indices, num_dims, num_symbols));
}
struct SizeAndStrideExpression {
  AffineExpr size;
  AffineExpr stride;
  ConstraintExpression constraints;
  SizeAndStrideExpression(
      AffineExpr size, AffineExpr stride,
      ConstraintExpression constraints = ConstraintExpression())
      : size(std::move(size)),
        stride(std::move(stride)),
        constraints(std::move(constraints)) {}
};
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromMod(
    AffineExpr lhs, AffineExpr modulus) {
  CHECK(modulus.getKind() == AffineExprKind::Constant);
  if (auto tile_size_expr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
    AffineExpr size = tile_size_expr -
                      mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                  tile_size_expr - 1, modulus) *
                          modulus;
    Interval zero_interval{0, 0};
    ConstraintExpression constraints;
    constraints.And(
        {{tile_size_expr % modulus, zero_interval}});
    constraints.Or(
        {{modulus % tile_size_expr, zero_interval}});
    return SizeAndStrideExpression(
        size, getAffineConstantExpr(1, lhs.getContext()),
        std::move(constraints));
  }
  return std::nullopt;
}
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromFloorDiv(
    AffineExpr num, AffineExpr den) {
  if (den.getKind() != AffineExprKind::Constant) {
    return std::nullopt;
  }
  if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(num)) {
    AffineExpr size = mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                  dim_expr + (den - 1), den);
    return SizeAndStrideExpression(
        size, getAffineConstantExpr(1, num.getContext()));
  }
  return std::nullopt;
}
void DestructureSummationImpl(AffineExpr expr,
                              std::vector<AffineExpr>& summands) {
  switch (expr.getKind()) {
    case AffineExprKind::Add: {
      const auto add = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
      DestructureSummationImpl(add.getLHS(), summands);
      DestructureSummationImpl(add.getRHS(), summands);
      break;
    }
    default:
      summands.push_back(expr);
      break;
  }
}
std::vector<AffineExpr> DestructureSummation(AffineExpr expr) {
  std::vector<AffineExpr> summands;
  DestructureSummationImpl(expr, summands);
  return summands;
}
std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals);
std::optional<std::vector<SizeAndStrideExpression>>
ExtractSizesAndStridesFromMultivariateSummation(
    AffineExpr summation, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals) {
  std::vector<AffineExpr> summands = DestructureSummation(summation);
  std::vector<SizeAndStrideExpression> sizes_and_strides;
  sizes_and_strides.reserve(summands.size());
  for (AffineExpr summand : summands) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(summand, dimension_intervals, symbol_intervals);
    if (!maybe_size_and_stride.has_value()) {
      VLOG(1) << "Couldn't extract size and stride from " << ToString(summand);
      return std::nullopt;
    }
    sizes_and_strides.push_back(*maybe_size_and_stride);
  }
  return sizes_and_strides;
}
AffineExpr CombineSizes(
    absl::Span<SizeAndStrideExpression const> sizes_and_strides) {
  CHECK(!sizes_and_strides.empty());
  AffineExpr product =
      getAffineConstantExpr(1, sizes_and_strides[0].size.getContext());
  for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    product = product * size_and_stride.size;
  }
  return product;
}
AffineExpr IfNeqOne(AffineExpr eq_param, AffineExpr true_expr,
                    AffineExpr false_expr,
                    int64_t eq_param_inclusive_upper_bound) {
  AffineExpr b = getAffineConstantExpr(eq_param_inclusive_upper_bound,
                                       eq_param.getContext());
  AffineExpr condition = mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                     b + 1 - eq_param, b);
  return condition * false_expr + (1 - condition) * true_expr;
}
void SortByStride(std::vector<SizeAndStrideExpression>& sizes_and_strides,
                  bool reverse = false) {
  absl::c_sort(sizes_and_strides, [&](const SizeAndStrideExpression& sas1,
                                      const SizeAndStrideExpression& sas2) {
    int64_t stride1 = llvm::cast<AffineConstantExpr>(sas1.stride).getValue();
    int64_t stride2 = llvm::cast<AffineConstantExpr>(sas2.stride).getValue();
    if (reverse) {
      return stride1 > stride2;
    }
    return stride1 < stride2;
  });
}
std::optional<int64_t> TryGetSizeExpressionRangeSize(
    AffineExpr size, absl::Span<Interval const> dimension_intervals) {
  if (size.getKind() == AffineExprKind::Constant) {
    return llvm::cast<AffineConstantExpr>(size).getValue();
  }
  CHECK(size.getKind() == AffineExprKind::DimId);
  auto dim_position = llvm::dyn_cast<AffineDimExpr>(size).getPosition();
  const Interval& interval = dimension_intervals.at(dim_position);
  if (interval.lower != 0) {
    VLOG(1) << "Attempted to combine strides but got dimension "
            << ToString(size) << " with lower bound " << interval.lower
            << " != 0";
    return std::nullopt;
  }
  return interval.upper + 1;
};
std::optional<AffineExpr> CombineStrides(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  CHECK(!sizes_and_strides.empty());
  for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    if (size_and_stride.stride.getKind() != AffineExprKind::Constant) {
      VLOG(1) << "Attempted to combine non-constant stride: "
              << ToString(size_and_stride.stride);
      return std::nullopt;
    }
    if (size_and_stride.size.getKind() != AffineExprKind::Constant &&
        size_and_stride.size.getKind() != AffineExprKind::DimId) {
      VLOG(1) << "Attempted to combine strides but got non-constant, "
                 "non-dimension size "
              << ToString(size_and_stride.size);
      return std::nullopt;
    }
  }
  SortByStride(sizes_and_strides);
  for (auto [dim_id, size_and_stride] : llvm::enumerate(sizes_and_strides)) {
    int64_t stride =
        llvm::cast<AffineConstantExpr>(size_and_stride.stride).getValue();
    if (dim_id > 0) {
      const SizeAndStrideExpression& previous_size_and_stride =
          sizes_and_strides[dim_id - 1];
      std::optional<int64_t> previous_size_expression_range_size =
          TryGetSizeExpressionRangeSize(previous_size_and_stride.size,
                                        dimension_intervals);
      if (!previous_size_expression_range_size.has_value()) {
        return std::nullopt;
      }
      int64_t previous_stride =
          llvm::cast<AffineConstantExpr>(previous_size_and_stride.stride)
              .getValue();
      if (*previous_size_expression_range_size * previous_stride != stride) {
        VLOG(1) << "Attempted to combine strides but stride did not grow "
                << "exactly as expected: got "
                << *previous_size_expression_range_size << " * "
                << previous_stride << " != " << stride;
        return std::nullopt;
      }
    }
  }
  MLIRContext* ctx = sizes_and_strides[0].stride.getContext();
  AffineExpr nested_if = getAffineConstantExpr(0, ctx);
  for (auto size_and_stride_it = sizes_and_strides.rbegin();
       size_and_stride_it != sizes_and_strides.rend(); ++size_and_stride_it) {
    AffineExpr size = size_and_stride_it->size;
    AffineExpr stride = size_and_stride_it->stride;
    std::optional<int64_t> size_expression_range_size =
        TryGetSizeExpressionRangeSize(size, dimension_intervals);
    if (!size_expression_range_size.has_value()) {
      return std::nullopt;
    }
    nested_if = IfNeqOne(size, stride, nested_if, *size_expression_range_size);
  }
  return nested_if;
}
std::optional<ConjointConstraints>
TryConstructSingleConjointConstraintForDestructuredSummation(
    absl::Span<SizeAndStrideExpression const> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals, int64_t partial_dim_index,
    int64_t num_full_dims) {
  CHECK_LE(partial_dim_index + num_full_dims, sizes_and_strides.size());
  ConjointConstraints constraints;
  Interval one = Interval{1, 1};
  int64_t running_size_index = 0;
  while (running_size_index < partial_dim_index) {
    constraints.push_back(
        Constraint{sizes_and_strides[running_size_index].size, one});
    ++running_size_index;
  }
  ++running_size_index;
  while (running_size_index <= partial_dim_index + num_full_dims) {
    AffineExpr size_expr = sizes_and_strides[running_size_index].size;
    std::optional<int64_t> max_size =
        TryGetSizeExpressionRangeSize(size_expr, dimension_intervals);
    if (!max_size.has_value()) {
      return std::nullopt;
    }
    constraints.push_back(Constraint{
        size_expr, Interval{*max_size, *max_size}});
    ++running_size_index;
  }
  while (running_size_index < sizes_and_strides.size()) {
    constraints.push_back(
        Constraint{sizes_and_strides[running_size_index].size, one});
    ++running_size_index;
  }
  return constraints;
}
ConstraintExpression ConstructConstraintExpressionForDestructuredSummation(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  SortByStride(sizes_and_strides, true);
  ConstraintExpression result;
  int64_t num_components = sizes_and_strides.size();
  for (int64_t partial_dim_index = 0; partial_dim_index < num_components;
       ++partial_dim_index) {
    for (int64_t num_full_dims = 0;
         num_full_dims < num_components - partial_dim_index; ++num_full_dims) {
      std::optional<ConjointConstraints> single_conjoint_constraint =
          TryConstructSingleConjointConstraintForDestructuredSummation(
              sizes_and_strides, dimension_intervals, partial_dim_index,
              num_full_dims);
      if (!single_conjoint_constraint.has_value()) {
        continue;
      }
      result.Or(std::move(*single_conjoint_constraint));
    }
  }
  if (result.IsAlwaysSatisfied()) {
    return ConstraintExpression::GetUnsatisfiableConstraintExpression();
  }
  return result;
}
std::optional<SizeAndStrideExpression> CombineSizesAndStrides(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  CHECK(!sizes_and_strides.empty());
  if (VLOG_IS_ON(1)) {
    for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
      LOG(INFO) << "CombineSizesAndStrides:";
      LOG(INFO) << "size: " << ToString(size_and_stride.size)
                << " stride: " << ToString(size_and_stride.stride);
    }
  }
  ConstraintExpression constraints;
  for (SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    constraints = ConstraintExpression::And(
        std::move(constraints), std::move(size_and_stride.constraints));
  }
  AffineExpr size = CombineSizes(sizes_and_strides);
  std::optional<AffineExpr> stride =
      CombineStrides(sizes_and_strides, dimension_intervals);
  if (!stride.has_value()) {
    return std::nullopt;
  }
  constraints = ConstraintExpression::And(
      std::move(constraints),
      ConstructConstraintExpressionForDestructuredSummation(
          std::move(sizes_and_strides), dimension_intervals));
  return SizeAndStrideExpression(size, *stride, std::move(constraints));
}
std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals) {
  MLIRContext* ctx = strided_indexing.getContext();
  switch (strided_indexing.getKind()) {
    case AffineExprKind::DimId:
      return SizeAndStrideExpression(strided_indexing,
                                     getAffineConstantExpr(1, ctx));
    case mlir::AffineExprKind::Mul: {
      const auto mul = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      AffineExpr lhs = mul.getLHS();
      std::optional<SizeAndStrideExpression> maybe_size_and_stride =
          ExtractSizeAndStride(lhs, dimension_intervals, symbol_intervals);
      if (!maybe_size_and_stride.has_value()) {
        return std::nullopt;
      }
      return SizeAndStrideExpression(
          maybe_size_and_stride->size,
          maybe_size_and_stride->stride * mul.getRHS());
    }
    case mlir::AffineExprKind::Mod: {
      auto mod = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromMod(mod.getLHS(), mod.getRHS());
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto floor_div = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromFloorDiv(floor_div.getLHS(),
                                              floor_div.getRHS());
    }
    case mlir::AffineExprKind::Constant:
      return SizeAndStrideExpression(getAffineConstantExpr(1, ctx),
                                     getAffineConstantExpr(0, ctx));
    case mlir::AffineExprKind::SymbolId: {
      auto symbol = llvm::cast<AffineSymbolExpr>(strided_indexing);
      const Interval& symbol_interval = symbol_intervals[symbol.getPosition()];
      if (symbol_interval.lower != 0) {
        return std::nullopt;
      }
      return SizeAndStrideExpression(
          getAffineConstantExpr(symbol_interval.upper + 1, ctx),
          getAffineConstantExpr(1, ctx));
    }
    case mlir::AffineExprKind::Add: {
      std::optional<std::vector<SizeAndStrideExpression>>
          maybe_sizes_and_strides =
              ExtractSizesAndStridesFromMultivariateSummation(
                  strided_indexing, dimension_intervals, symbol_intervals);
      if (!maybe_sizes_and_strides.has_value()) {
        return std::nullopt;
      }
      return CombineSizesAndStrides(std::move(*maybe_sizes_and_strides),
                                    dimension_intervals);
    }
    case mlir::AffineExprKind::CeilDiv:
      break;
  };
  LOG(FATAL) << "unreachable";
}
AffineExpr SimplifyAffineExpr(const AffineExpr& expr,
                              const IndexingMap& reference) {
  AffineMap tmp_affine_map =
      AffineMap::get(reference.GetDimVars().size(),
                     reference.GetSymbolCount(),
                     {expr},
                     reference.GetMLIRContext());
  IndexingMap tmp_indexing_map(
      std::move(tmp_affine_map),
      reference.GetDimVars(),
      reference.GetRangeVars(),
      reference.GetRTVars(),
      reference.GetConstraints());
  tmp_indexing_map.Simplify();
  CHECK_EQ(tmp_indexing_map.GetAffineMap().getResults().size(), 1);
  return tmp_indexing_map.GetAffineMap().getResults().back();
}
std::optional<ConjointConstraints> TryIntersectConjointConstraints(
    ConjointConstraints conjunction_1,
    const ConjointConstraints& conjunction_2) {
  if (conjunction_1.empty()) {
    return conjunction_2;
  }
  if (conjunction_2.empty()) {
    return std::move(conjunction_1);
  }
  ConjointConstraints result = std::move(conjunction_1);
  for (const auto& constraint : conjunction_2) {
    Constraint* result_it =
        llvm::find_if(result, [&](const Constraint& result_constraint) {
          return result_constraint.expr == constraint.expr;
        });
    const auto& [expr, interval] = constraint;
    if (result_it != result.end()) {
      auto& [result_expr, result_interval] = *result_it;
      result_interval = result_interval.Intersect(interval);
      if (!result_interval.IsFeasible()) {
        VLOG(1) << "Got two incompatible intervals for expression "
                << ToString(expr);
        return std::nullopt;
      }
    } else {
      result.push_back(Constraint{expr, interval});
    }
  }
  return result;
}
}  
 ConstraintExpression ConstraintExpression::And(
    ConstraintExpression first, ConstraintExpression second) {
  if (!first.is_satisfiable_ || !second.is_satisfiable_) {
    return ConstraintExpression::GetUnsatisfiableConstraintExpression();
  }
  if (first.IsAlwaysSatisfied()) {
    return second;
  }
  if (second.IsAlwaysSatisfied()) {
    return first;
  }
  ConstraintExpression result;
  for (ConjointConstraints& conjunction_1 :
       first.disjoint_conjoint_constraints_) {
    for (ConjointConstraints& conjunction_2 :
         second.disjoint_conjoint_constraints_) {
      std::optional<ConjointConstraints> maybe_conjunction =
          TryIntersectConjointConstraints(conjunction_1, conjunction_2);
      if (maybe_conjunction.has_value()) {
        result.disjoint_conjoint_constraints_.push_back(
            std::move(*maybe_conjunction));
      }
    }
  }
  result.is_satisfiable_ = !result.disjoint_conjoint_constraints_.empty();
  return result;
}
 ConstraintExpression ConstraintExpression::Or(
    ConstraintExpression first, ConstraintExpression second) {
  if (!first.is_satisfiable_) {
    return second;
  }
  if (!second.is_satisfiable_) {
    return first;
  }
  absl::c_copy(second.disjoint_conjoint_constraints_,
               std::back_inserter(first.disjoint_conjoint_constraints_));
  return first;
}
void ConstraintExpression::Or(ConjointConstraints conjunction) {
  if (conjunction.empty()) {
    return;
  }
  disjoint_conjoint_constraints_.push_back(std::move(conjunction));
  is_satisfiable_ = true;
}
void ConstraintExpression::And(ConjointConstraints conjunction) {
  if (!is_satisfiable_ || conjunction.empty()) {
    return;
  }
  if (disjoint_conjoint_constraints_.empty()) {
    disjoint_conjoint_constraints_.push_back(std::move(conjunction));
    return;
  }
  llvm::SmallVector<ConjointConstraints, 2> new_constraints;
  new_constraints.reserve(disjoint_conjoint_constraints_.size());
  for (ConjointConstraints& conjunction_2 : disjoint_conjoint_constraints_) {
    std::optional<ConjointConstraints> maybe_result =
        TryIntersectConjointConstraints(std::move(conjunction_2), conjunction);
    if (maybe_result.has_value()) {
      new_constraints.push_back(std::move(*maybe_result));
    }
  }
  is_satisfiable_ = !new_constraints.empty();
  disjoint_conjoint_constraints_ = std::move(new_constraints);
}
bool ConstraintExpression::IsSatisfiedBy(
    absl::Span<const int64_t> parameters) const {
  if (IsAlwaysSatisfied()) {
    return true;
  }
  if (!is_satisfiable_) {
    return false;
  }
  bool constraints_are_satisfied = false;
  for (const ConstraintExpression::ConjointConstraints& conjunction :
       disjoint_conjoint_constraints_) {
    bool conjunction_is_satisfied = true;
    for (const auto& [constrained_expr, interval] : conjunction) {
      int64_t constrained_value =
          EvaluateAffineExpr(constrained_expr, parameters);
      if (constrained_value < interval.lower ||
          constrained_value > interval.upper) {
        conjunction_is_satisfied = false;
        break;
      }
    }
    constraints_are_satisfied |= conjunction_is_satisfied;
  }
  return constraints_are_satisfied;
}
std::string ConstraintExpression::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}
void ConstraintExpression::Print(std::ostream& out) const {
  if (IsAlwaysSatisfied()) {
    out << "always satisfied";
  } else if (is_satisfiable()) {
    std::vector<std::string> conjunction_strings;
    conjunction_strings.reserve(disjoint_conjoint_constraints_.size());
    for (const auto& disjunction : disjoint_conjoint_constraints_) {
      std::vector<std::string> constraint_strings;
      constraint_strings.reserve(disjunction.size());
      for (const auto& [expr, interval] : disjunction) {
        constraint_strings.push_back(absl::StrCat(xla::gpu::ToString(expr),
                                                  " in ", interval.ToString()));
      }
      std::sort(constraint_strings.begin(), constraint_strings.end());
      conjunction_strings.push_back(absl::StrJoin(constraint_strings, " && "));
    }
    std::sort(conjunction_strings.begin(), conjunction_strings.end());
    out << absl::StrJoin(conjunction_strings, " || ");
  } else {
    out << "unsatisfiable";
  }
  out << "\n";
}
namespace {
bool IsConstraintAlwaysSatisfied(mlir::AffineExpr expr, Interval interval) {
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return interval.Contains(constant.getValue());
  }
  return false;
}
bool IsConstraintUnsatisfiable(mlir::AffineExpr expr, Interval interval) {
  if (!interval.IsFeasible()) {
    return true;
  }
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return !interval.Contains(constant.getValue());
  }
  return false;
}
struct Unsatisfiable {};
struct AlwaysSatisfied {};
std::variant<Unsatisfiable, AlwaysSatisfied, ConjointConstraints>
SimplifyConjointConstraints(const ConjointConstraints& conjunction) {
  ConjointConstraints result;
  for (const auto& [expr, interval] : conjunction) {
    if (IsConstraintAlwaysSatisfied(expr, interval)) {
      continue;
    }
    if (IsConstraintUnsatisfiable(expr, interval)) {
      return Unsatisfiable{};
    }
    result.push_back(Constraint{expr, interval});
  }
  if (result.empty()) {
    return AlwaysSatisfied{};
  }
  auto comp = [](const Constraint& a, const Constraint& b) {
    if (a.expr != b.expr) {
      CHECK_EQ(a.expr.getContext(), b.expr.getContext())
          << "AffineExpr should be from the same MLIRContext.";
      return a.expr.getImpl() < b.expr.getImpl();
    }
    if (a.interval.lower != b.interval.lower) {
      return a.interval.lower < b.interval.lower;
    }
    return a.interval.upper < b.interval.upper;
  };
  std::sort(result.begin(), result.end(), comp);
  return result;
}
ConstraintExpression SimplifyConstraintExpression(
    const ConstraintExpression constraint_expression) {
  if (!constraint_expression.is_satisfiable() ||
      constraint_expression.IsAlwaysSatisfied()) {
    return constraint_expression;
  }
  SmallVector<ConjointConstraints, 2> simplified_disjoint_conjoint_constraints;
  for (const auto& conjunction :
       constraint_expression.DisjointConjointConstraints()) {
    auto simplified_conjunction = SimplifyConjointConstraints(conjunction);
    if (std::holds_alternative<Unsatisfiable>(simplified_conjunction)) {
      continue;
    }
    if (std::holds_alternative<AlwaysSatisfied>(simplified_conjunction)) {
      return ConstraintExpression();
    }
    simplified_disjoint_conjoint_constraints.push_back(
        std::get<ConjointConstraints>(std::move(simplified_conjunction)));
  }
  absl::flat_hash_set<ConjointConstraints> unique_conjunctions;
  for (const auto& conjunction : simplified_disjoint_conjoint_constraints) {
    if (unique_conjunctions.contains(conjunction)) {
      continue;
    }
    unique_conjunctions.insert(std::move(conjunction));
  }
  auto result = ConstraintExpression::GetUnsatisfiableConstraintExpression();
  for (auto& conjoint_constraints : unique_conjunctions) {
    result.Or(std::move(conjoint_constraints));
  }
  return result;
}
}  
void ConstraintExpression::Simplify() {
  *this = SimplifyConstraintExpression(std::move(*this));
}
 std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    IndexingMap indexing_map) {
  VLOG(1) << "SymbolicTile::FromIndexingMap: " << indexing_map;
  bool did_simplify = indexing_map.Simplify();
  VLOG(1) << "did_simplify: " << did_simplify;
  if (indexing_map.GetConstraintsCount() != 0) {
    VLOG(1) << "Deriving symbolic tile from indexing map with pre-existing "
            << "constraints might produce spurious constraints. Bailing out. "
            << indexing_map;
    return std::nullopt;
  }
  AffineMap input_affine_map = indexing_map.GetAffineMap();
  MLIRContext* mlir_context = input_affine_map.getContext();
  std::vector<AffineExpr> offset_expressions =
      SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
          input_affine_map, getAffineConstantExpr(0, mlir_context),
          indexing_map.GetRangeVarsCount())
          .getResults();
  for (AffineExpr& expr : offset_expressions) {
    expr = SimplifyAffineExpr(expr, indexing_map);
  }
  ConstraintExpression constraints;
  std::vector<AffineExpr> size_expressions;
  std::vector<AffineExpr> stride_expressions;
  size_expressions.reserve(offset_expressions.size());
  stride_expressions.reserve(offset_expressions.size());
  for (auto [composite_indexing, offset] :
       llvm::zip(input_affine_map.getResults(), offset_expressions)) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(SimplifyAffineExpr(composite_indexing - offset,
                                                indexing_map),
                             indexing_map.GetDimensionBounds(),
                             indexing_map.GetSymbolBounds());
    if (!maybe_size_and_stride.has_value()) {
      VLOG(1) << "No size and stride extracted";
      return std::nullopt;
    }
    size_expressions.push_back(maybe_size_and_stride->size);
    stride_expressions.push_back(maybe_size_and_stride->stride);
    constraints = ConstraintExpression::And(
        std::move(constraints), std::move(maybe_size_and_stride->constraints));
  }
  for (auto [offset, size, stride] :
       llvm::zip(offset_expressions, size_expressions, stride_expressions)) {
    auto constant = llvm::dyn_cast<AffineConstantExpr>(stride);
    if (constant && constant.getValue() < 0) {
      offset = offset + size * stride - stride;
      stride = -stride;
    } else if (!constant) {
      VLOG(1) << "Unexpected non-constant stride expression: "
              << xla::gpu::ToString(stride);
    }
  }
  std::vector<IndexingMap::Variable> tile_sizes = indexing_map.GetDimVars();
  for (IndexingMap::Variable& tile_size : tile_sizes) {
    tile_size.bounds.lower += 1;
    tile_size.bounds.upper += 1;
  }
  std::vector<AffineExpr> results;
  absl::c_move(std::move(offset_expressions), std::back_inserter(results));
  absl::c_move(std::move(size_expressions), std::back_inserter(results));
  absl::c_move(std::move(stride_expressions), std::back_inserter(results));
  AffineMap tile_affine_map =
      AffineMap::get(tile_sizes.size(),
                     indexing_map.GetSymbolCount(),
                     results,
                     indexing_map.GetMLIRContext());
  IndexingMap tile_map(
      std::move(tile_affine_map),
      std::move(tile_sizes),
      indexing_map.GetRangeVars(),
      indexing_map.GetRTVars());
  tile_map.RemoveUnusedSymbols();
  CHECK_EQ(tile_map.GetRangeVarsCount(), 0);
  VLOG(1) << "tile_map: " << tile_map;
  constraints.Simplify();
  return SymbolicTile(std::move(tile_map), std::move(constraints));
}
std::string SymbolicTile::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}
void SymbolicTile::Print(std::ostream& out) const {
  out << "Symbolic tile with \n";
  out << "\toffset_map: " << offset_map();
  out << "\n\tsize_map: " << size_map();
  out << "\n\tstride_map: " << stride_map();
  const std::vector<IndexingMap::Variable>& rt_vars = tile_map_.GetRTVars();
  if (!rt_vars.empty()) {
    out << "\n\trt_vars: ";
    for (const auto& [index, rt_var] : llvm::enumerate(rt_vars)) {
      out << 's' << index << " in " << rt_var.bounds << ", ";
    }
  }
  if (!constraints_.IsAlwaysSatisfied()) {
    out << "\n\tconstraints: ";
    constraints_.Print(out);
  }
}
namespace {
constexpr int kNumComponentsPerTiledDimension = 3;
}  
AffineMap SymbolicTile::offset_map() const {
  int64_t num_results = tile_map_.GetAffineMap().getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;
  return tile_map_.GetAffineMap().getSliceMap(0, component_size);
}
AffineMap SymbolicTile::size_map() const {
  AffineMap affine_map = tile_map_.GetAffineMap();
  int64_t num_results = affine_map.getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;
  return AffineMap::get(
      affine_map.getNumDims(),
      affine_map.getNumSymbols() - tile_map_.GetRTVarsCount(),
      affine_map.getResults().slice(component_size, component_size),
      affine_map.getContext());
}
AffineMap SymbolicTile::stride_map() const {
  AffineMap affine_map = tile_map_.GetAffineMap();
  int64_t num_results = affine_map.getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;
  return AffineMap::get(
      affine_map.getNumDims(),
      affine_map.getNumSymbols() - tile_map_.GetRTVarsCount(),
      affine_map.getResults().slice(2 * component_size, component_size),
      affine_map.getContext());
}
}  
}  