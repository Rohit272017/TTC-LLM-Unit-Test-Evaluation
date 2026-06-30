#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir {
namespace odml {
namespace {
class StridedArrayViewBase {
 protected:
  StridedArrayViewBase(ArrayRef<int64_t> shape, ArrayRef<int64_t> index,
                       int64_t axis) {
    assert(shape.size() == index.size());
    assert(axis < shape.size());
    assert(axis >= 0);
    assert(index[axis] == 0);
    offset_ = IndexToOffset(shape, index);
    stride_ = StrideForAxis(shape, axis);
    size_ = shape[axis];
  }
  int64_t size() const { return size_; }
  static std::optional<SmallVector<int64_t>> NextTensorIndex(
      SmallVector<int64_t> index, ArrayRef<int64_t> shape, int64_t fixed_axis) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    assert(fixed_axis < shape.size());
    assert(fixed_axis >= 0);
    assert(index[fixed_axis] == 0);
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      if (dim == fixed_axis) continue;
      ++index[dim];
      if (index[dim] < shape[dim]) return std::move(index);
      index[dim] = 0;
    }
    return std::nullopt;
  }
 protected:
  int64_t OffsetForIndex(int64_t i) const { return offset_ + i * stride_; }
 private:
  static int64_t StrideForAxis(ArrayRef<int64_t> shape, int64_t axis) {
    int64_t stride = 1;  
    for (int64_t dim = shape.size() - 1; dim > axis; --dim) {
      stride *= shape[dim];
    }
    return stride;
  }
  static int64_t IndexToOffset(ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> index) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  
    int64_t offset = 0;
    int64_t stride = 1;
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      offset += index[dim] * stride;
      stride *= shape[dim];
    }
    return offset;
  }
  int64_t offset_;
  int64_t stride_;
  int64_t size_;
};
template <typename T>
class StridedArrayView;  
template <>
class StridedArrayView<DenseIntElementsAttr> : StridedArrayViewBase {
 public:
  StridedArrayView(const DenseIntElementsAttr& data, ArrayRef<int64_t> shape,
                   ArrayRef<int64_t> index, int64_t axis)
      : StridedArrayViewBase(shape, index, axis), data_(data) {
    int64_t element_count = 1;
    for (int64_t i = 0, e = shape.size(); i < e; ++i) {
      element_count *= shape[i];
    }
    assert(data.getNumElements() == element_count);
  }
  using StridedArrayViewBase::NextTensorIndex;
  using StridedArrayViewBase::size;
  int64_t operator[](int64_t i) const {
    return data_.getValues<APInt>()[OffsetForIndex(i)].getSExtValue();
  }
 private:
  const DenseIntElementsAttr& data_;
};
bool MatchIotaBroadCastInDim(DenseIntElementsAttr dimensions, Value iota) {
  auto iota_broadcast =
      dyn_cast_or_null<mhlo::BroadcastInDimOp>(iota.getDefiningOp());
  if (!iota_broadcast || iota_broadcast.getBroadcastDimensions() != dimensions)
    return false;
  if (!isa_and_nonnull<mhlo::IotaOp>(
          iota_broadcast.getOperand().getDefiningOp()))
    return false;
  return true;
}
bool MatchIotaConst(DenseIntElementsAttr dimensions, Value iota) {
  DenseIntElementsAttr iota_const_attr;
  if (!matchPattern(iota, m_Constant(&iota_const_attr))) return false;
  auto iota_type = iota_const_attr.getType();
  auto iota_shape = iota_type.getShape();
  auto reduce_dim = (*dimensions.value_begin<APInt>()).getSExtValue();
  if (reduce_dim < 0) reduce_dim += iota_type.getRank();
  auto index =
      std::optional<SmallVector<int64_t>>(std::in_place, iota_type.getRank());
  while (index.has_value()) {
    StridedArrayView<DenseIntElementsAttr> array_view(
        iota_const_attr, iota_shape, *index, reduce_dim);
    for (int64_t i = 0; i < array_view.size(); ++i) {
      if (array_view[i] != i) return false;
    }
    index = StridedArrayView<DenseIntElementsAttr>::NextTensorIndex(
        std::move(*index), iota_shape, reduce_dim);
  }
  return true;
}
bool MatchReshapedIota(DenseIntElementsAttr dimensions, Value iota) {
  if (dimensions.getNumElements() != 1) return false;
  auto reshape_op = dyn_cast_or_null<mhlo::ReshapeOp>(iota.getDefiningOp());
  if (!reshape_op) return false;
  auto operand_type =
      mlir::dyn_cast<RankedTensorType>(reshape_op.getOperand().getType());
  if (!operand_type || !operand_type.hasStaticShape()) return false;
  auto reshape_type = mlir::cast<RankedTensorType>(reshape_op.getType());
  if (operand_type.getRank() != 1) return false;
  if (!dyn_cast_or_null<mhlo::IotaOp>(reshape_op.getOperand().getDefiningOp()))
    return false;
  int64_t iota_dim = (*dimensions.value_begin<APInt>()).getSExtValue();
  for (int64_t i = 0, e = reshape_type.getRank(); i < e; ++i) {
    if (i == iota_dim) {
      if (reshape_type.getDimSize(i) != operand_type.getDimSize(0))
        return false;
    } else if (reshape_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}
bool MatchSingleIota(DenseIntElementsAttr dimensions, Value iota) {
  auto iota_op = dyn_cast_or_null<mhlo::IotaOp>(iota.getDefiningOp());
  if (!iota_op || dimensions.getNumElements() != 1) return false;
  auto dim = *dimensions.value_begin<APInt>();
  return dim == iota_op.getIotaDimension();
}
bool MatchConstIotaBroadCastInDim(DenseIntElementsAttr dimensions, Value iota) {
  if (dimensions.getNumElements() != 1) return false;
  auto iota_broadcast =
      dyn_cast_or_null<mhlo::BroadcastInDimOp>(iota.getDefiningOp());
  if (!iota_broadcast || iota_broadcast.getBroadcastDimensions() != dimensions)
    return false;
  DenseElementsAttr range_const;
  if (!matchPattern(iota_broadcast.getOperand(), m_Constant(&range_const)))
    return false;
  int index = 0;
  for (auto value : range_const.getValues<APInt>()) {
    if (value != index++) return false;
  }
  return true;
}
}  
bool MatchIota(DenseIntElementsAttr dimensions, Value iota) {
  return MatchSingleIota(dimensions, iota) ||
         MatchIotaBroadCastInDim(dimensions, iota) ||
         MatchReshapedIota(dimensions, iota) ||
         MatchConstIotaBroadCastInDim(dimensions, iota) ||
         MatchIotaConst(dimensions, iota);
}
}  
}  