#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PERMUTATION_H_
#include <cstdint>
#include <type_traits>
#include "llvm/ADT/ArrayRef.h"  
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"  
#include "mlir/Support/LLVM.h"  
namespace mlir::quant {
template <typename T,
          typename = std::enable_if_t<std::is_default_constructible_v<T>, void>>
SmallVector<T> Permute(const ArrayRef<T> values,
                       const ArrayRef<int64_t> permutation) {
  SmallVector<T> permuted_values(values.size(), T{});
  for (auto [i, permutation_idx] : llvm::enumerate(permutation)) {
    permuted_values[i] = std::move(values[permutation_idx]);
  }
  return permuted_values;
}
}  
#endif  