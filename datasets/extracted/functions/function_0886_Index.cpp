#include "xla/service/llvm_ir/ir_array.h"
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace xla {
namespace llvm_ir {
IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      llvm::Value* linear, const Shape& shape,
                      llvm::Type* index_type)
    : Index(multidim, shape, index_type) {
  CHECK_NE(linear, nullptr);
  linear_ = linear;
}
void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 llvm::IRBuilder<>* b) const {
  int64_t divisor = 1;
  const Layout& layout = shape.layout();
  for (int64_t i = 0; i < layout.minor_to_major_size(); ++i) {
    int64_t dimension = layout.minor_to_major(i);
    int64_t size_of_current_dimension = shape.dimensions(dimension);
    auto* quot = b->CreateUDiv(linear, GetConstantWithIndexType(divisor));
    if (i < layout.minor_to_major_size() - 1) {
      (*multidim)[dimension] = b->CreateURem(
          quot, GetConstantWithIndexType(size_of_current_dimension));
    } else {
      (*multidim)[dimension] = quot;
    }
    divisor *= size_of_current_dimension;
  }
}
void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 absl::Span<llvm::Value*> dynamic_dims,
                                 llvm::IRBuilder<>* b) const {
  CHECK_EQ(shape.dimensions_size(), dynamic_dims.size());
  CHECK_EQ(multidim_.size(), shape.rank());
  llvm::Value* divisor = GetConstantWithIndexType(1);
  const Layout& layout = shape.layout();
  for (int64_t i = 0; i < layout.minor_to_major_size(); ++i) {
    int64_t dimension = layout.minor_to_major(i);
    auto* quot = b->CreateUDiv(linear, divisor, "quot");
    if (i < layout.minor_to_major_size() - 1) {
      llvm::Value* casted_dynamic_dim =
          b->CreateIntCast(dynamic_dims[dimension], quot->getType(),
                           true);
      (*multidim)[dimension] =
          b->CreateURem(quot, casted_dynamic_dim, "dim_value");
      divisor = b->CreateMul(divisor, casted_dynamic_dim, "divisor");
    } else {
      (*multidim)[dimension] = quot;
    }
  }
}
IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, b);
}
IrArray::Index::Index(llvm::Value* linear,
                      absl::Span<llvm::Value* const> multidim,
                      const Shape& shape, llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK_EQ(multidim.size(), shape.rank());
  for (auto dim : multidim) {
    if (dim) {
      CHECK_EQ(dim->getType(), index_type_);
    }
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, b);
  for (int i = 0; i < multidim.size(); ++i) {
    if (multidim[i] != nullptr) {
      multidim_[i] = multidim[i];
    }
  }
}
IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      absl::Span<llvm::Value*> dynamic_dims,
                      llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, dynamic_dims, b);
}
IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      absl::Span<int64_t const> dimensions,
                      llvm::Type* index_type)
    : Index(multidim, ShapeUtil::MakeShape( PRED, dimensions),
            index_type) {}
IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      const Shape& shape, llvm::Type* index_type)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(nullptr),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()),
      index_type_(index_type) {
  CHECK_NE(index_type_, nullptr);
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  for (const auto* dim : multidim) {
    CHECK_NE(dim, nullptr);
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}
IrArray::IrArray(llvm::Value* base_ptr, llvm::Type* pointee_type, Shape shape)
    : base_ptr_(base_ptr),
      pointee_type_(pointee_type),
      shape_(std::move(shape)) {
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(base_ptr_->getType()->isPointerTy());
  int depth = 0;
  element_type_ = pointee_type;
  while (llvm::ArrayType* array_type =
             llvm::dyn_cast<llvm::ArrayType>(element_type_)) {
    element_type_ = array_type->getElementType();
    ++depth;
  }
  if (!shape_.IsArray() || ShapeUtil::IsScalar(shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, shape_.rank()) << shape.ShortDebugString();
  }
}
bool IrArray::Index::LinearValidOnShape(const Shape& a) const {
  auto b = ShapeUtil::MakeShape(a.element_type(), dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ShapeUtil::ElementsIn(a) == ShapeUtil::ElementsIn(b) &&
         ShapeUtil::ReshapeIsBitcast(a, b);
}
IrArray::Index IrArray::Index::SourceIndexOfReshape(
    const Shape& output_shape, const Shape& input_shape,
    llvm::IRBuilder<>* builder) const {
  CHECK_EQ(multidim_.size(), output_shape.rank());
  std::vector<llvm::Value*> source_multidim_index(
      input_shape.rank(), llvm::UndefValue::get(index_type_));
  if (std::optional<ShapeUtil::ShapeEqualityDescriptor> trivial_reshape =
          ShapeUtil::InsertedOrDeleted1SizedDimensions(input_shape,
                                                       output_shape)) {
    for (int64_t i = 0, j = 0, k = 0, l = 0; i < source_multidim_index.size();
         ++i) {
      if (j == trivial_reshape->deleted_dimensions.size() ||
          trivial_reshape->deleted_dimensions[j] > i) {
        while (l < trivial_reshape->inserted_dimensions.size() &&
               trivial_reshape->inserted_dimensions[l] == k) {
          ++k;
          ++l;
        }
        source_multidim_index[i] = multidim_[k];
        ++k;
      } else {
        source_multidim_index[i] = GetConstantWithIndexType(0);
        ++j;
      }
    }
  } else {
    const auto common_factors =
        CommonFactors(input_shape.dimensions(), output_shape.dimensions());
    for (ssize_t k = common_factors.size() - 2; k >= 0; --k) {
      absl::Span<int64_t const> dimensions = output_shape.dimensions().subspan(
          common_factors[k].second,
          common_factors[k + 1].second - common_factors[k].second);
      llvm::Value* logical_linear_index =
          Index(absl::Span<llvm::Value* const>(multidim_).subspan(
                    common_factors[k].second,
                    common_factors[k + 1].second - common_factors[k].second),
                dimensions, index_type_)
              .Linearize(dimensions, builder);
      for (int64_t i = common_factors[k + 1].first - 1;
           i >= common_factors[k].first; --i) {
        llvm::Value* divisor =
            GetConstantWithIndexType(input_shape.dimensions(i));
        if (input_shape.dimensions(i) == 1) {
          source_multidim_index[i] = GetConstantWithIndexType(0);
        } else if (i == common_factors[k].first) {
          source_multidim_index[i] = logical_linear_index;
        } else {
          source_multidim_index[i] =
              builder->CreateURem(logical_linear_index, divisor);
        }
        logical_linear_index =
            builder->CreateUDiv(logical_linear_index, divisor);
      }
    }
  }
  if (linear() != nullptr && LayoutUtil::HasLayout(input_shape) &&
      LayoutUtil::HasLayout(output_shape) &&
      ShapeUtil::ReshapeIsBitcast(input_shape, output_shape)) {
    return Index(source_multidim_index, linear(), input_shape, index_type_);
  }
  return Index(source_multidim_index, input_shape, index_type_);
}
IrArray::Index IrArray::Index::SourceIndexOfSlice(
    const Shape& operand_shape, absl::Span<const int64_t> starts,
    absl::Span<const int64_t> strides, llvm::IRBuilder<>* builder) const {
  std::vector<llvm::Value*> source_multi_index(multidim_.size());
  for (int i = 0; i < multidim_.size(); ++i) {
    int64_t stride = strides[i];
    if (stride != 1) {
      source_multi_index[i] = builder->CreateAdd(
          builder->CreateMul(multidim_[i], GetConstantWithIndexType(stride)),
          GetConstantWithIndexType(starts[i]));
    } else {
      source_multi_index[i] =
          builder->CreateAdd(multidim_[i], GetConstantWithIndexType(starts[i]));
    }
  }
  return Index(source_multi_index, operand_shape, index_type_);
}
IrArray::Index IrArray::Index::SourceIndexOfTranspose(
    const Shape& shape, const Shape& operand_shape,
    absl::Span<const int64_t> dimension_mapping) const {
  std::vector<llvm::Value*> operand_multidim_index =
      PermuteInverse(multidim(), dimension_mapping);
  if (linear() != nullptr && LayoutUtil::HasLayout(operand_shape) &&
      LayoutUtil::HasLayout(shape) &&
      ShapeUtil::TransposeIsBitcast(operand_shape, shape, dimension_mapping)) {
    return Index(operand_multidim_index, linear(), operand_shape, index_type_);
  }
  return Index(operand_multidim_index, operand_shape, index_type_);
}
IrArray::Index IrArray::Index::SourceIndexOfBitcast(
    const Shape& shape, const Shape& operand_shape,
    llvm::IRBuilder<>* builder) const {
  CHECK(LayoutUtil::HasLayout(shape) && LayoutUtil::HasLayout(operand_shape));
  const ShapeUtil::BitcastDecomposition decomposition =
      ShapeUtil::DecomposeBitcast(operand_shape, shape);
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposition)) {
    return SourceIndexOfReshape(shape, operand_shape, builder);
  }
  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          decomposition)) {
    const auto& decomposition_transpose =
        std::get<ShapeUtil::BitcastDecompositionTranspose>(decomposition);
    return SourceIndexOfTranspose(shape, operand_shape,
                                  decomposition_transpose.transpose_dims);
  }
  CHECK(std::holds_alternative<ShapeUtil::BitcastDecompositionTrt>(
      decomposition));
  const auto& decomposition_trt =
      std::get<ShapeUtil::BitcastDecompositionTrt>(decomposition);
  Index index = *this;
  if (!decomposition_trt.IsTranspose2Identity()) {
    index = index.SourceIndexOfTranspose(shape, decomposition_trt.reshape_shape,
                                         decomposition_trt.transpose2_dims);
  }
  index =
      index.SourceIndexOfReshape(decomposition_trt.reshape_shape,
                                 decomposition_trt.transpose1_shape, builder);
  if (!decomposition_trt.IsTranspose1Identity()) {
    index = index.SourceIndexOfTranspose(decomposition_trt.transpose1_shape,
                                         operand_shape,
                                         decomposition_trt.transpose1_dims);
  }
  return index;
}
IrArray::Index IrArray::Index::SourceIndexOfBitcast(
    const Shape& operand_shape, llvm::IRBuilder<>* builder) const {
  auto shape = ShapeUtil::MakeShape(F32, dims_);
  *shape.mutable_layout() = layout_;
  return SourceIndexOfBitcast(shape, operand_shape, builder);
}
IrArray::Index IrArray::Index::SourceIndexOfBroadcast(
    const Shape& shape, const Shape& operand_shape,
    absl::Span<const int64_t> dimension_mapping,
    llvm::IRBuilder<>* builder) const {
  int64_t rank = operand_shape.rank();
  std::vector<llvm::Value*> source_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    source_index[i] = multidim_[dimension_mapping[i]];
  }
  if (linear_ == nullptr || !LayoutUtil::HasLayout(operand_shape) ||
      !LayoutUtil::HasLayout(shape) || rank == 1) {
    return Index(source_index, operand_shape, index_type_);
  }
  std::vector<int64_t> logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(shape.layout());
  int64_t output_rank = shape.rank();
  int64_t min_broadcasted_dimension = output_rank;
  int64_t max_broadcasted_dimension = -1;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t physical_dim = logical_to_physical[dimension_mapping[i]];
    min_broadcasted_dimension =
        std::min(min_broadcasted_dimension, physical_dim);
    max_broadcasted_dimension =
        std::max(max_broadcasted_dimension, physical_dim);
  }
  bool contiguous_broadcast_dimensions =
      max_broadcasted_dimension - min_broadcasted_dimension == rank - 1;
  if (!contiguous_broadcast_dimensions) {
    return Index(source_index, operand_shape, index_type_);
  }
  std::vector<int64_t> operand_logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(operand_shape.layout());
  for (int64_t i = 0; i < rank; ++i) {
    if (operand_logical_to_physical[i] !=
        logical_to_physical[dimension_mapping[i]] - min_broadcasted_dimension) {
      return Index(source_index, operand_shape, index_type_);
    }
  }
  llvm::Value* linear = linear_;
  int64_t divisor = 1;
  for (int64_t i = max_broadcasted_dimension + 1; i < output_rank; ++i) {
    divisor *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  if (divisor > 1) {
    linear = builder->CreateUDiv(linear, GetConstantWithIndexType(divisor));
  }
  if (min_broadcasted_dimension > 0) {
    int64_t mod = 1;
    for (int64_t i = min_broadcasted_dimension; i <= max_broadcasted_dimension;
         ++i) {
      mod *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
    }
    linear = builder->CreateURem(linear, GetConstantWithIndexType(mod));
  }
  return Index(source_index, linear, operand_shape, index_type_);
}
llvm::Value* IrArray::Index::Linearize(absl::Span<const int64_t> dimensions,
                                       llvm::IRBuilder<>* builder) const {
  CHECK_EQ(size(), dimensions.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  int64_t multiplier = 1;
  for (ssize_t i = 0; i < size(); ++i) {
    int64_t dimension = layout_.minor_to_major(i);
    llvm::Value* addend = builder->CreateMul(
        (*this)[dimension], GetConstantWithIndexType(multiplier), "",
        true, true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              true, true);
    multiplier *= dimensions[dimension];
  }
  return logical_linear_index;
}
llvm::Value* IrArray::Index::Linearize(
    const std::vector<llvm::Value*>& dynamic_dims,
    llvm::IRBuilder<>* builder) const {
  CHECK_EQ(size(), dynamic_dims.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  llvm::Value* multiplier = GetConstantWithIndexType(1);
  for (ssize_t i = 0; i < size(); ++i) {
    int64_t dimension = layout_.minor_to_major(i);
    llvm::Value* addend = builder->CreateMul((*this)[dimension], multiplier, "",
                                             true, true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              true, true);
    if (i < size() - 1) {
      multiplier = builder->CreateMul(multiplier, dynamic_dims[dimension],
                                      "multiplier");
    }
  }
  return logical_linear_index;
}
llvm::Value* IrArray::EmitArrayElementAddress(const IrArray::Index& index,
                                              llvm::IRBuilder<>* b,
                                              absl::string_view name,
                                              bool use_linear_index,
                                              llvm::Value** bit_offset) const {
  if (ShapeUtil::IsScalar(shape_)) {
    if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
      CHECK_NE(bit_offset, nullptr);
      *bit_offset =
          b->getInt8(8 - primitive_util::BitWidth(shape_.element_type()));
    }
    return base_ptr_;
  }
  CHECK_EQ(index.size(), shape_.rank());
  CHECK(index.ShapeIsCompatible(shape_))
      << "Shape " << index.AsShapeWithType(shape_.element_type()).ToString(true)
      << " is not compatible with " << shape_.ToString(true);
  if (use_linear_index && index.LinearValidOnShape(shape_)) {
    return EmitLinearArrayElementAddress(index, b, name, bit_offset);
  }
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    IrArray::Index linear_index = index;
    if (!index.LinearValidOnShape(shape_)) {
      std::vector<int64_t> dimensions;
      dimensions.reserve(shape_.rank());
      for (int64_t i = 0; i < shape_.rank(); ++i) {
        dimensions.push_back(shape_.dimensions(i));
      }
      llvm::Value* linearized = index.Linearize(dimensions, b);
      linear_index = IrArray::Index(linearized, shape_, b);
    }
    return EmitLinearArrayElementAddress(linear_index, b, name, bit_offset);
  }
  std::vector<llvm::Value*> actual_index;
  for (int64_t i = 0; i < index.size(); ++i) {
    auto dim = shape_.dimensions(i);
    actual_index.push_back(
        dim == 1 ? llvm::ConstantInt::get(index[i]->getType(), 0) : index[i]);
  }
  CHECK_GT(index.size(), 0);
  std::vector<llvm::Value*> gep_indices(
      1, llvm::ConstantInt::get(index[0]->getType(), 0));
  for (int64_t i = 0; i < shape_.rank(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return b->CreateInBoundsGEP(pointee_type_, base_ptr_, gep_indices,
                              llvm_ir::AsStringRef(name));
}
llvm::Value* IrArray::EmitLinearArrayElementAddress(
    const IrArray::Index& index, llvm::IRBuilder<>* b, absl::string_view name,
    llvm::Value** bit_offset) const {
  CHECK(index.LinearValidOnShape(shape_));
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* type = PrimitiveTypeToIrType(shape_.element_type(), module);
  if (!primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    auto linear_index = llvm::dyn_cast<llvm::BinaryOperator>(index.linear());
    if (linear_index && (linear_index->getOpcode() == llvm::Instruction::Add)) {
      llvm::Value* index_operand_0 = linear_index->getOperand(0);
      llvm::Value* index_operand_1 = linear_index->getOperand(1);
      llvm::Value* ptr_address =
          b->CreateGEP(type, base_ptr_, index_operand_0, "");
      return b->CreateInBoundsGEP(type, ptr_address, index_operand_1,
                                  llvm_ir::AsStringRef(name));
    } else {
      return b->CreateInBoundsGEP(type, base_ptr_, index.linear(),
                                  llvm_ir::AsStringRef(name));
    }
  }
  llvm::Type* index_type = index.linear()->getType();
  auto bit_width = primitive_util::BitWidth(shape_.element_type());
  llvm::Value* elements_per_byte =
      llvm::ConstantInt::get(index_type, 8 / bit_width);
  llvm::Value* remainder = b->CreateURem(index.linear(), elements_per_byte);
  llvm::Value* byte_offset = b->CreateUDiv(index.linear(), elements_per_byte);
  CHECK_NE(bit_offset, nullptr);
  *bit_offset = b->CreateIntCast(
      b->CreateSub(llvm::ConstantInt::get(index_type, 8 - bit_width),
                   b->CreateMul(remainder,
                                llvm::ConstantInt::get(index_type, bit_width))),
      b->getInt8Ty(), false);
  return b->CreateInBoundsGEP(b->getInt8Ty(), base_ptr_, byte_offset,
                              llvm_ir::AsStringRef(name));
}
void IrArray::AnnotateLoadStoreInstructionWithMetadata(
    llvm::Instruction* instruction) const {
  CHECK(llvm::isa<llvm::LoadInst>(instruction) ||
        llvm::isa<llvm::StoreInst>(instruction));
  CHECK(!llvm::isa<llvm::StoreInst>(instruction) || !is_invariant_)
      << "Trying to create a store to an invariant IRArray.";
  for (const auto& kind_md_pair : metadata_) {
    instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
  }
}
llvm::Value* IrArray::EmitReadArrayElement(const Index& index,
                                           llvm::IRBuilder<>* b,
                                           absl::string_view name,
                                           bool use_linear_index) const {
  llvm::Value* bit_offset = nullptr;
  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, name, use_linear_index, &bit_offset);
  llvm::Type* load_type =
      primitive_util::IsSubByteNonPredType(shape_.element_type())
          ? b->getInt8Ty()
          : element_type_;
  llvm::LoadInst* load =
      b->CreateLoad(load_type, element_address, llvm_ir::AsStringRef(name));
  AnnotateLoadStoreInstructionWithMetadata(load);
  llvm::Value* elem = load;
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    llvm::Value* shifted = b->CreateLShr(load, bit_offset);
    elem = b->CreateTrunc(
        shifted, b->getIntNTy(primitive_util::BitWidth(shape_.element_type())));
  }
  return elem;
}
void IrArray::EmitWriteArrayElement(const Index& index, llvm::Value* value,
                                    llvm::IRBuilder<>* b,
                                    bool use_linear_index) const {
  llvm::Value* bit_offset = nullptr;
  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, "", use_linear_index, &bit_offset);
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    llvm::LoadInst* load = b->CreateLoad(b->getInt8Ty(), element_address);
    AnnotateLoadStoreInstructionWithMetadata(load);
    value = b->CreateIntCast(value, b->getInt8Ty(),
                             false);
    value = b->CreateShl(value, bit_offset);
    auto bit_width = primitive_util::BitWidth(shape_.element_type());
    llvm::Value* mask = b->getInt8(~LsbMask<uint8_t>(bit_width));
    mask = b->CreateIntrinsic(b->getInt8Ty(), llvm::Intrinsic::fshl,
                              {mask, mask, bit_offset});
    llvm::Value* masked_load = b->CreateAnd(load, mask);
    value = b->CreateOr(masked_load, value);
  }
  llvm::StoreInst* store = b->CreateStore(value, element_address);
  AnnotateLoadStoreInstructionWithMetadata(store);
}
IrArray IrArray::CastToShape(const Shape& new_shape,
                             llvm::IRBuilder<>* b) const {
  if (shape_ == new_shape) return *this;
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(new_shape, module);
  IrArray new_irarray(base_ptr_, new_ir_type, new_shape);
  new_irarray.metadata_ = metadata_;
  return new_irarray;
}
bool IrArray::Index::ShapeIsCompatible(const Shape& a, const Shape& b) {
  const auto get_strides = [](const Shape& shape) {
    int rank = shape.dimensions().size();
    int64_t stride = 1;
    std::vector<int64_t> strides;
    for (int i = 0; i < rank; i++) {
      auto dim = shape.dimensions(shape.layout().minor_to_major(i));
      if (dim != 1) {
        stride *= dim;
        strides.push_back(stride);
      }
    }
    return strides;
  };
  return get_strides(a) == get_strides(b);
}
}  
}  