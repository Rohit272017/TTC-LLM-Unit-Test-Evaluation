#include "tensorflow/core/ir/utility.h"
#include <optional>
#include "mlir/IR/Block.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Region.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
namespace mlir {
namespace tfg {
Block::BlockArgListType GetLoopRegionDataArgs(Region &region) {
  Block::BlockArgListType args = region.getArguments();
  return args.drop_back(args.size() / 2);
}
Block::BlockArgListType GetLoopRegionControlTokens(Region &region) {
  Block::BlockArgListType args = region.getArguments();
  return args.drop_front(args.size() / 2);
}
BlockArgument GetLoopRegionControlOf(BlockArgument data) {
  Block &block = *data.getOwner();
  return block.getArgument(data.getArgNumber() + block.getNumArguments() / 2);
}
BlockArgument GetLoopRegionDataOf(BlockArgument ctl) {
  Block &block = *ctl.getOwner();
  return block.getArgument(ctl.getArgNumber() - block.getNumArguments() / 2);
}
Value LookupControlDependency(Value data) {
  assert(!mlir::isa<ControlType>(data.getType()) && "expected a data type");
  Value control_dep;
  if (auto result = mlir::dyn_cast<OpResult>(data)) {
    control_dep = *std::prev(result.getOwner()->result_end());
  } else {
    auto arg = mlir::cast<BlockArgument>(data);
    control_dep = cast<ControlArgumentInterface>(arg.getOwner()->getParentOp())
                      .getControlTokenOf(arg);
  }
  assert(mlir::isa<ControlType>(control_dep.getType()) &&
         "expected a control type");
  return control_dep;
}
std::optional<Value> LookupDataValue(Value ctl) {
  assert(mlir::isa<ControlType>(ctl.getType()) && "expected a control type");
  Value data;
  if (auto result = mlir::dyn_cast<OpResult>(ctl)) {
    if (result.getOwner()->getNumResults() == 1) return {};
    data = *result.getOwner()->result_begin();
  } else {
    auto arg = mlir::cast<BlockArgument>(ctl);
    data = cast<ControlArgumentInterface>(arg.getOwner()->getParentOp())
               .getDataValueOf(arg);
  }
  assert(!mlir::isa<ControlType>(data.getType()) && "expected a data type");
  return data;
}
}  
}  