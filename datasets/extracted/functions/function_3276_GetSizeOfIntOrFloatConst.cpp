#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/const_op_size.h"
#include <climits>
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/Types.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
namespace mlir {
namespace quant {
namespace {
constexpr int64_t kAssumedNumBytesPerElem = 4;
int64_t GetSizeOfIntOrFloatConst(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();
  const ElementsAttr const_value = const_op.getValue();
  const auto bytes_per_elem =
      static_cast<int64_t>(dtype.getIntOrFloatBitWidth() / CHAR_BIT);
  return bytes_per_elem * const_value.getNumElements();
}
int64_t GetSizeOfStringConst(TF::ConstOp const_op) {
  const ElementsAttr const_value = const_op.getValue();
  const auto str_attr = cast<DenseStringElementsAttr>(const_value);
  return absl::c_accumulate(
      str_attr.getRawStringData(), 0,
      [](int64_t acc, const StringRef str_value) -> int64_t {
        return acc + str_value.size();
      });
}
int64_t GetSizeOfUnsupportedTypeConst(TF::ConstOp const_op) {
  return kAssumedNumBytesPerElem * const_op.getValue().getNumElements();
}
}  
int64_t GetSizeInBytes(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();
  if (dtype.isIntOrFloat()) {
    return GetSizeOfIntOrFloatConst(const_op);
  } else if (isa<TF::StringType>(dtype)) {
    return GetSizeOfStringConst(const_op);
  } else {
    return GetSizeOfUnsupportedTypeConst(const_op);
  }
}
}  
}  